"""
Orchestration: takes a saved file and produces the finished API response.

Validates the upload, builds the shared media context, runs every analyzer in
the config registry behind a crash guard, hands the signals to fusion, then
attaches heatmaps and the PDF report. app.py routes call in here and do nothing
else. Job results are kept in a plain in-memory dict, no database.
"""

import datetime
import os
import time
import uuid

import config
import media_utils

# job_id -> finished response dict. Survives only as long as the process, which
# is all the WhatsApp flow and the results page need.
JOBS = {}


def new_job_id():
    """
    Short unique id for one analysis run.

    Returns (str): 12 hex characters, used for scratch directory names, the
    report filename and the JOBS key.
    """
    return uuid.uuid4().hex[:12]


def validate_media(media):
    """
    Check an upload against the size, type and duration limits.

    media (dict): context from media_utils.build_media_context.

    Returns (tuple): (ok, message). ok is True when the file may be analyzed;
    when False, message is a sentence suitable for showing to the user.
    Duration is only enforced when ffprobe actually reported one.
    """
    if media["size"] > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        return False, f"File is larger than the {mb} MB limit. Please trim it and try again."

    if media["type"] == "unknown":
        return False, ("Could not tell what kind of file this is. "
                       "Upload an image, video or audio clip.")

    duration = media.get("duration")
    if duration and duration > config.MAX_MEDIA_SECONDS:
        return False, (f"Clip is {int(duration)}s long. The limit is "
                       f"{config.MAX_MEDIA_SECONDS}s, so please upload a shorter section.")

    return True, ""


def collect_media_assets(signals):
    """
    Pull the presentation assets out of the signal details into one place.

    signals (list): the signal dicts returned by the analyzers.

    Returns (dict): {"heatmaps": [url, ...], "ela_image": url|None,
    "timeline": [{"t": float, "score": float}, ...]}.
    The UI reads only this, so it never has to know which analyzer produced
    which file. Missing assets come back as [] or None rather than absent keys.
    """
    assets = {"heatmaps": [], "ela_image": None, "timeline": []}

    for sig in signals:
        details = sig.get("details") or {}

        for path in details.get("heatmaps", []) or []:
            url = path if str(path).startswith("/static/") else config.static_url(path)
            if url:
                assets["heatmaps"].append(url)

        if details.get("ela_image") and not assets["ela_image"]:
            raw = details["ela_image"]
            assets["ela_image"] = raw if str(raw).startswith("/static/") else config.static_url(raw)

        if details.get("timeline") and not assets["timeline"]:
            assets["timeline"] = details["timeline"]

    return assets


def run_analyzers(media):
    """
    Run every registered analyzer over one media file.

    media (dict): context from media_utils.build_media_context.

    Returns (list): one signal dict per analyzer, registry order. Each call is
    wrapped by media_utils.run_safely, so an analyzer that raises yields an
    error signal and the rest still run. An empty registry yields [].
    """
    signals = []
    for mod_name, func in config.load_analyzers():
        signals.append(media_utils.run_safely(func, media,
                                              signal_hint=mod_name,
                                              name_hint=mod_name))
    return signals


def analyze_media(saved, make_report=True):
    """
    Full pipeline for one file: validate, analyze, fuse, report.

    saved (dict): output of media_utils.save_upload or save_bytes.
    make_report (bool): generate the PDF. WhatsApp needs it, quick tests may not.

    Returns (dict): the API response documented in the README:
    {"file": {"name","type","size","sha256","analyzed_at","job_id"},
     "verdict": {...}, "signals": [...],
     "media": {"heatmaps","ela_image","timeline"},
     "report_url": str|None}
    On a validation failure it returns {"error": message, "file": {...}} with no
    verdict, so the caller can show the message without special-casing a crash.
    The result is also stored in JOBS under the job id.
    """
    job_id = new_job_id()
    started = time.time()

    media = media_utils.build_media_context(saved, job_id)
    file_block = {
        "name": media["name"],
        "type": media["type"],
        "size": media["size"],
        "sha256": media["sha256"],
        "duration": media.get("duration"),
        "analyzed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "job_id": job_id,
    }

    ok, message = validate_media(media)
    if not ok:
        result = {"error": message, "file": file_block,
                  "verdict": None, "signals": [], "media": {}, "report_url": None}
        JOBS[job_id] = result
        return result

    print(f"[pipeline] job {job_id} {media['type']} {media['name']} "
          f"({media['size'] / 1024:.0f} KB)")

    signals = run_analyzers(media)
    verdict = _fuse(signals, media)

    result = {
        "file": file_block,
        "verdict": verdict,
        "signals": signals,
        "media": collect_media_assets(signals),
        "report_url": None,
    }

    if make_report:
        result["report_url"] = _build_report(result, job_id)

    result["file"]["total_runtime_s"] = round(time.time() - started, 2)
    print(f"[pipeline] job {job_id} done in {result['file']['total_runtime_s']}s "
          f"-> {(verdict or {}).get('verdict')}")

    JOBS[job_id] = result
    return result


def _fuse(signals, media):
    """
    Call the fusion layer, tolerating its absence during early build phases.

    signals (list): analyzer signal dicts.
    media (dict): media context, passed through for type-aware wording.

    Returns (dict): the fusion verdict block, or a placeholder inconclusive
    verdict if fusion.py is missing or itself raised. The pipeline never dies
    because the verdict could not be computed.
    """
    try:
        import fusion
        return fusion.fuse_signals(signals, media)
    except Exception as exc:
        print(f"[pipeline] fusion unavailable: {exc}")
        return {
            "verdict": "inconclusive",
            "overall_score": None,
            "headline": "Not enough working signals to reach a verdict on this file.",
            "top_evidence": [],
        }


def _build_report(result, job_id):
    """
    Generate the PDF evidence report, tolerating its absence.

    result (dict): the assembled response, minus report_url.
    job_id (str): used for the PDF filename.

    Returns (str|None): "/static/reports/<file>.pdf" on success, None if
    report.py is missing or generation failed. A missing report degrades the
    response, it does not fail the analysis.
    """
    try:
        import report
        path = report.build_report(result, job_id)
        return config.static_url(path)
    except Exception as exc:
        print(f"[pipeline] report generation failed: {exc}")
        return None


def get_job(job_id):
    """
    Look up a finished analysis by id.

    job_id (str): id from a previous analyze_media call.

    Returns (dict|None): the stored response, or None if unknown or if the
    process restarted since it ran.
    """
    return JOBS.get(job_id)


def cleanup_job_files(job_id):
    """
    Delete the scratch frames and WAV for one job.

    job_id (str): the job whose static/work/<job_id> directory should go.

    Returns (bool): True if the directory was removed or never existed, False
    if deletion failed (a file still open on Windows, typically). Heatmaps,
    reports and the original upload are deliberately left alone because the
    results page and the PDF still reference them.
    """
    import shutil
    target = os.path.join(config.WORK_DIR, job_id)
    try:
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        return True
    except Exception:
        return False
