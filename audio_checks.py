"""
Audio signal: detect cloned or synthesised speech.

Runs a wav2vec2 anti-spoofing classifier over the audio track. Works on audio
uploads and on the audio inside a video, so a video automatically gets both a
visual and a vocal opinion. Anything longer than one window is scored in three
places (start, middle, end) because a scam clip often splices real and cloned
speech together, and a single average would hide that.
"""

import config
import media_utils
# Same label-to-score mapping as the image model: both return
# [{"label", "score"}] and both use fake/real wording, so the logic is shared
# rather than duplicated.
from image_checks import score_from_predictions

_AUDIO_PIPE = None
_MODEL_ID = None
_LOAD_ERROR = None


def get_audio_model():
    """
    Return the audio classification pipeline, loading it on first call.

    Returns (tuple): (pipeline, model_id), or (None, None) if neither the
    primary nor the fallback model could be loaded. The failure reason is
    cached in _LOAD_ERROR so a second upload does not retry a slow download.
    """
    global _AUDIO_PIPE, _MODEL_ID, _LOAD_ERROR

    if _AUDIO_PIPE is not None or _LOAD_ERROR is not None:
        return _AUDIO_PIPE, _MODEL_ID

    import time

    from transformers import pipeline as hf_pipeline

    device = config.get_device()
    errors = []
    seen = set()
    ordered = []
    for candidate in getattr(config, "AUDIO_MODEL_CANDIDATES", (config.AUDIO_MODEL_ID,)):
        resolved = config.resolve_model(candidate, config.AUDIO_MODEL_LOCAL_DIR)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    if config.AUDIO_MODEL_FALLBACK_ID not in seen:
        ordered.append(config.resolve_model(config.AUDIO_MODEL_FALLBACK_ID, config.AUDIO_MODEL_LOCAL_DIR))

    for model_id in ordered:
        try:
            started = time.time()
            _AUDIO_PIPE = hf_pipeline("audio-classification", model=model_id,
                                      device=device)
            # Report the published model name even when the weights were loaded
            # from a local folder, so reports do not show a machine path.
            _MODEL_ID = (config.AUDIO_MODEL_ID
                         if model_id == config.AUDIO_MODEL_LOCAL_DIR else model_id)
            print(f"[audio] loaded {_MODEL_ID} in {time.time() - started:.1f}s")
            return _AUDIO_PIPE, _MODEL_ID
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    _LOAD_ERROR = " | ".join(errors)
    print(f"[audio] no audio model available: {_LOAD_ERROR}")
    return None, None


def _windows(samples, rate):
    """
    Pick the slices of audio to score.

    samples (numpy.ndarray): 1-D waveform.
    rate (int): sample rate, expected 16000.

    Returns (list): [{"label": str, "start": float, "samples": ndarray}].
    Short clips give one window covering everything. Longer clips give three
    (start, middle, end), so spliced audio shows up as disagreement between
    windows instead of averaging away.
    """
    span = config.AUDIO_WINDOW_SECONDS * rate
    total = len(samples)

    if total <= span:
        return [{"label": "whole clip", "start": 0.0, "samples": samples}]

    middle_start = max(0, (total - span) // 2)
    end_start = max(0, total - span)
    picks = [("start", 0), ("middle", middle_start), ("end", end_start)]

    out = []
    for label, begin in picks:
        chunk = samples[begin:begin + span]
        if len(chunk) > rate * 0.5:      # ignore a scrap shorter than half a second
            out.append({"label": label, "start": round(begin / rate, 2),
                        "samples": chunk})
    return out


def _summarise(score, window_results):
    """
    Write the human-readable note for the audio result.

    score (float): overall synthetic probability.
    window_results (list): per-window results, each with "label" and "score".

    Returns (str): one or two plain sentences. When windows disagree strongly
    the note says so outright, because partially cloned audio is a real pattern
    in scam calls and a single average would bury it.
    """
    pct = int(round(score * 100))

    scores = [w["score"] for w in window_results if w["score"] is not None]
    if len(scores) > 1 and (max(scores) - min(scores)) > 0.4:
        worst = max(window_results, key=lambda w: w["score"] or 0)
        return (f"The voice scores differently across the clip. The {worst['label']} "
                f"section reads {int(round((worst['score'] or 0) * 100))}% synthetic "
                f"while other parts read much lower, which is what spliced or "
                f"partially cloned audio looks like.")

    if score >= 0.85:
        return (f"The voice shows strong signatures of AI cloning or synthesis "
                f"({pct}% synthetic).")
    if score >= 0.65:
        return f"The voice leans synthetic at {pct}%, with some cloning-like artefacts."
    if score >= 0.35:
        return (f"The voice sits in the uncertain range at {pct}% synthetic. Short or "
                f"noisy recordings often land here, so this is not evidence either way.")
    return (f"The voice reads as a genuine human recording, scoring only {pct}% "
            f"synthetic.")


def analyze_audio(media):
    """
    Signal: score speech for AI cloning or synthesis.

    media (dict): media context. Accepts type "audio", and type "video" whose
        audio track is extracted first. Images are not applicable.

    Returns (dict): standard signal dict, signal "voice_clone".
    details carries per-window scores, the model id and the source (whether the
    audio came from a video track).
    status is "not_applicable" for images and for silent video, and "error" if
    conversion failed or the model is unavailable. A silent video is not a
    failure, so it is reported as not applicable rather than as an error.
    """
    kind = media.get("type")
    if kind not in ("audio", "video"):
        return media_utils.na_signal("voice_clone", "AI Voice Cloning",
                                     "Applies to audio and to video with sound.")

    # Reuse the WAV if something already made one for this job.
    wav = media.get("wav")
    if not wav:
        if kind == "video":
            wav = media_utils.extract_audio_track(media["path"], media["job_id"])
            if wav is None:
                return media_utils.na_signal(
                    "voice_clone", "AI Voice Cloning",
                    "This video has no audio track, so there is no voice to check.")
        else:
            wav = media_utils.to_wav_16k(media["path"], media["job_id"])
        media["wav"] = wav

    if not wav:
        return media_utils.error_signal("voice_clone", "AI Voice Cloning",
                                        "audio could not be converted to 16 kHz WAV")

    samples, rate = media_utils.read_wav(wav)
    if samples is None or len(samples) < rate * 0.5:
        return media_utils.error_signal(
            "voice_clone", "AI Voice Cloning",
            "the audio is unreadable or shorter than half a second")

    pipe, model_id = get_audio_model()
    if pipe is None:
        return media_utils.error_signal("voice_clone", "AI Voice Cloning",
                                        _LOAD_ERROR or "audio model unavailable")

    window_results = []
    for window in _windows(samples, rate):
        try:
            preds = pipe({"array": window["samples"], "sampling_rate": rate})
            score = score_from_predictions(preds)
        except Exception as exc:
            print(f"[audio] window {window['label']} failed: {exc}")
            score = None
        window_results.append({"label": window["label"], "start": window["start"],
                               "score": round(score, 4) if score is not None else None})

    usable = [w["score"] for w in window_results if w["score"] is not None]
    if not usable:
        return media_utils.error_signal("voice_clone", "AI Voice Cloning",
                                        "no audio window could be scored")

    overall = sum(usable) / len(usable)

    # Short clips give these models very little to work with, so say so through
    # confidence rather than pretending the number is solid.
    seconds = len(samples) / rate
    length_factor = min(1.0, seconds / 8.0)
    confidence = round(min(0.95, (0.4 + abs(overall - 0.5) * 0.9) * length_factor), 3)

    return media_utils.make_signal(
        "voice_clone", "AI Voice Cloning",
        synthetic_score=overall,
        confidence=confidence,
        human_note=_summarise(overall, window_results),
        details={"model": model_id,
                 "windows": window_results,
                 "duration_s": round(seconds, 2),
                 "source": "video audio track" if kind == "video" else "audio file"},
    )
