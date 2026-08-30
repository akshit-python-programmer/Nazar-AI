"""
Live progress state for a running analysis, so the web page can show what the
engine is actually doing instead of a guessed animation.

The pipeline and the analyzers publish stage transitions here; app.py serves
them over /progress/<job_id> and the page polls. State is a plain in-memory
dict keyed by job id, thrown away when the process exits, which is all a
single-machine demo needs.

This lives in its own module rather than in pipeline.py because the analyzers
publish into it too, and importing pipeline from an analyzer would be circular.
"""

import time

# job_id -> live state. Written by the worker thread, read by request threads.
# Every write replaces a whole key rather than mutating nested state in place,
# which is enough to stay consistent under the GIL without a lock.
_JOBS = {}

# Bounds how many finished jobs stay in memory on a long-running server.
MAX_TRACKED = 40


def start(job_id, kind):
    """
    Begin tracking a job.

    job_id (str): the pipeline's job id.
    kind (str): "image", "video", "audio" or "unknown", so the page knows
        which stages to expect.

    Returns (dict): the fresh state record.
    """
    _evict()
    _JOBS[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "stages": {},          # stage id -> {state, label, note}
        "phase": {"icon": "📥", "text": "Starting up"},
        "pct": 0.02,
        "done": False,
        "error": None,
        "frames": None,        # {"done": int, "total": int} while scoring video
        "started": time.time(),
        "elapsed": 0.0,
    }
    return _JOBS[job_id]


def stage(job_id, stage_id, state, label=None, note=None):
    """
    Record one checkpoint's state.

    job_id (str): job being tracked. Unknown ids are ignored, so an analyzer
        running outside a tracked job (the CLI smoke test) costs nothing.
    stage_id (str): matches the step ids the page renders, e.g. "ingest",
        "frames", "voice_clone".
    state (str): "run", "ok", "err", "skip" or "wait".
    label (str|None): short pill text, e.g. "0.87 synthetic".
    note (str|None): one line of detail under the stage name.

    Returns: None.
    """
    job = _JOBS.get(job_id)
    if not job:
        return
    entry = {"state": state}
    if label is not None:
        entry["label"] = label
    if note is not None:
        entry["note"] = note
    # Replace the whole stages dict so a reader never sees a half-built entry.
    stages = dict(job["stages"])
    stages[stage_id] = entry
    job["stages"] = stages
    job["elapsed"] = round(time.time() - job["started"], 2)


def phase(job_id, icon, text, pct=None):
    """
    Set the one-line "what is happening now" banner.

    job_id (str): job being tracked.
    icon (str): a single emoji shown before the text.
    text (str): plain sentence, present tense.
    pct (float|None): overall completion 0..1. Never moves backwards, because
        a bar that retreats reads as a bug even when the estimate improved.

    Returns: None.
    """
    job = _JOBS.get(job_id)
    if not job:
        return
    job["phase"] = {"icon": icon, "text": text}
    if pct is not None:
        job["pct"] = max(job["pct"], min(1.0, float(pct)))
    job["elapsed"] = round(time.time() - job["started"], 2)


def set_band(job_id, lo, hi):
    """
    Declare the slice of the progress bar the current stage owns.

    job_id (str): job being tracked.
    lo (float), hi (float): start and end of the band, 0..1.

    Returns: None. Only the pipeline knows how the bar is divided between
    analyzers, so it sets this and frames() interpolates inside it.
    """
    job = _JOBS.get(job_id)
    if job:
        job["band"] = (float(lo), float(hi))


def frames(job_id, done, total):
    """
    Report video frame-scoring progress.

    job_id (str): job being tracked.
    done (int): frames scored so far.
    total (int): frames that will be scored.

    Returns: None. This is the slowest part of a video run, so it gets its own
    counter, and it also drives the bar across the current stage's band. Without
    that the bar would sit still for the whole of a 60-frame video, which is
    exactly the stretch a viewer most wants to see moving.
    """
    job = _JOBS.get(job_id)
    if not job:
        return
    job["frames"] = {"done": int(done), "total": int(total)}
    band = job.get("band")
    if band and total:
        lo, hi = band
        job["pct"] = max(job["pct"], min(hi, lo + (done / float(total)) * (hi - lo)))
    job["elapsed"] = round(time.time() - job["started"], 2)


def signal_done(job_id, sig):
    """
    Record a finished analyzer using its own signal dict.

    job_id (str): job being tracked.
    sig (dict): the standard signal dict the analyzer returned.

    Returns: None. Translates the signal into the page's stage vocabulary so
    the caller does not repeat this mapping: an errored analyzer shows as
    "err", a not-applicable one as "skip", and a scored one shows its score.
    """
    status = sig.get("status")
    name = sig.get("signal")
    if status == "error":
        stage(job_id, name, "err", "failed",
              str(sig.get("human_note") or "")[:160])
        return
    if status == "not_applicable":
        stage(job_id, name, "skip", "n/a",
              str(sig.get("human_note") or "")[:160])
        return

    score = sig.get("synthetic_score")
    label = "no score" if score is None else f"{round(score * 100)}% synthetic"
    stage(job_id, name, "ok", label, str(sig.get("human_note") or "")[:200])


def finish(job_id, result=None, error=None):
    """
    Mark a job complete and attach its final payload.

    job_id (str): job being tracked.
    result (dict|None): the finished API response, so the page can render
        without a second request.
    error (str|None): message when the run failed.

    Returns: None.
    """
    job = _JOBS.get(job_id)
    if not job:
        return
    job["done"] = True
    job["pct"] = 1.0
    job["error"] = error
    job["result"] = result
    job["elapsed"] = round(time.time() - job["started"], 2)


def get(job_id):
    """
    Read the current state of a job.

    job_id (str): job to look up.

    Returns (dict|None): the state record, or None if the id is unknown,
    which includes any job from before the last restart.
    """
    return _JOBS.get(job_id)


def _evict():
    """
    Drop the oldest finished jobs once MAX_TRACKED is exceeded.

    Returns: None. Only finished jobs are eligible, so a long video run is
    never evicted while the page is still polling it.
    """
    if len(_JOBS) <= MAX_TRACKED:
        return
    finished = sorted((j for j in _JOBS.values() if j.get("done")),
                      key=lambda j: j["started"])
    for job in finished[:len(_JOBS) - MAX_TRACKED]:
        _JOBS.pop(job["job_id"], None)
