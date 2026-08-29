"""
Filesystem and media plumbing shared by every analyzer.

Saving uploads, hashing them, working out whether a file is image / video /
audio, pulling frames out of a video and normalising audio to 16 kHz mono WAV.
Also holds the small constructors for the uniform signal dict, since every
analyzer needs those and they belong next to the contract they build.
Nothing in here loads a model.
"""

import hashlib
import json
import os
import subprocess
import time
import uuid

import config


# ---------------------------------------------------------------- signal contract

def make_signal(signal, display_name, status="ok", synthetic_score=None,
                confidence=0.0, human_note="", details=None):
    """
    Build the one dict shape that fusion, the UI, the PDF and WhatsApp all read.

    signal (str): machine name, unique across analyzers, e.g. "voice_clone".
    display_name (str): label shown to a human, e.g. "AI Voice Cloning".
    status (str): "ok", "error" or "not_applicable".
    synthetic_score (float|None): 0.0 authentic to 1.0 synthetic, None if the
        signal produced no score (metadata and provenance usually do this).
    confidence (float): 0..1, how much this signal should be trusted for this
        particular file. Fusion multiplies the score by it.
    human_note (str): one or two plain sentences a non-technical person can read.
    details (dict|None): signal-specific extras (timelines, paths, raw fields).

    Returns (dict): the signal dict. Scores and confidence are clamped into
    range here so a buggy analyzer cannot poison the fused average.
    """
    if synthetic_score is not None:
        synthetic_score = max(0.0, min(1.0, float(synthetic_score)))
    return {
        "signal": signal,
        "display_name": display_name,
        "status": status,
        "synthetic_score": synthetic_score,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "human_note": human_note,
        "details": details or {},
    }


def error_signal(signal, display_name, exc):
    """
    Signal dict for an analyzer that raised.

    signal (str), display_name (str): as in make_signal.
    exc (Exception|str): what went wrong.

    Returns (dict): status "error", no score, zero confidence, with the error
    text in details["error"] so the UI can show it without guessing.
    A broken signal is reported as broken, never as evidence either way.
    """
    return make_signal(
        signal, display_name,
        status="error",
        confidence=0.0,
        human_note="This check could not run on this file.",
        details={"error": f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)},
    )


def na_signal(signal, display_name, why):
    """
    Signal dict for a check that does not apply to this media type.

    signal (str), display_name (str): as in make_signal.
    why (str): human-readable reason, e.g. "Only applies to still images".

    Returns (dict): status "not_applicable", no score, zero confidence.
    """
    return make_signal(signal, display_name, status="not_applicable",
                       confidence=0.0, human_note=why)


def run_safely(func, media, signal_hint="unknown", name_hint="Check"):
    """
    Call one analyzer and never let it break the pipeline.

    func (callable): analyzer taking the media dict, returning a signal dict.
    media (dict): the media context built by build_media_context.
    signal_hint (str), name_hint (str): used only to label the error dict if
        func explodes before it can name itself.

    Returns (dict): whatever func returned, or an error_signal if it raised or
    returned something that is not a dict. Runtime in seconds is added to
    details["runtime_s"] and printed, so slow signals are visible in the console.
    """
    started = time.time()
    try:
        result = func(media)
        if not isinstance(result, dict) or "signal" not in result:
            result = error_signal(signal_hint, name_hint,
                                  "analyzer returned a malformed result")
    except Exception as exc:
        result = error_signal(signal_hint, name_hint, exc)

    elapsed = round(time.time() - started, 2)
    result.setdefault("details", {})["runtime_s"] = elapsed
    print(f"[signal] {result.get('signal', signal_hint):<26} "
          f"{result.get('status', '?'):<15} {elapsed:>6.2f}s")
    return result


# ---------------------------------------------------------------- files

def sha256_file(path, chunk=1024 * 1024):
    """
    Hash a file so the report can carry a tamper-evident fingerprint.

    path (str): file to hash.
    chunk (int): read size in bytes, default 1 MB.

    Returns (str): lowercase hex SHA-256 digest, or "" if the file cannot be
    read (the report then just shows an empty hash rather than failing).
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


def detect_media_type(path):
    """
    Decide whether a file is an image, video or audio clip.

    path (str): file on disk.

    Returns (str): "image", "video", "audio" or "unknown". Extension is checked
    first because it is instant; if the extension is unhelpful the file is
    probed with ffprobe and classified by the streams it actually contains.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in config.ALLOWED_IMAGE_EXT:
        return "image"
    if ext in config.ALLOWED_VIDEO_EXT:
        return "video"
    if ext in config.ALLOWED_AUDIO_EXT:
        return "audio"

    # Unknown extension: ask ffprobe what is actually inside.
    info = ffprobe_json(path)
    if not info:
        return "unknown"
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    if "video" in kinds:
        # A single-frame "video" stream is how ffprobe reports a still image.
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and s.get("nb_frames") in ("1", 1):
                return "image"
        return "video"
    if "audio" in kinds:
        return "audio"
    return "unknown"


def save_upload(file_storage):
    """
    Write an uploaded file to static/uploads under a collision-proof name.

    file_storage: Werkzeug FileStorage from request.files, or any object with
        .filename and .save(path).

    Returns (dict): {"path": abs path, "name": original filename,
    "stored_name": name on disk, "size": bytes}.
    Raises ValueError if the upload is empty or has no filename, because there
    is nothing sensible to analyze in that case.
    """
    original = getattr(file_storage, "filename", "") or ""
    if not original.strip():
        raise ValueError("No file was provided")

    ext = os.path.splitext(original)[1].lower()
    stored = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(config.UPLOAD_DIR, stored)
    file_storage.save(dest)

    size = os.path.getsize(dest)
    if size == 0:
        os.remove(dest)
        raise ValueError("The uploaded file is empty")

    return {"path": dest, "name": os.path.basename(original),
            "stored_name": stored, "size": size}


def save_bytes(data, suffix=".bin", original_name=None):
    """
    Same as save_upload but for bytes already in memory (the WhatsApp path).

    data (bytes): file content.
    suffix (str): extension to give the stored file, including the dot.
    original_name (str|None): name to report back; defaults to the stored name.

    Returns (dict): same shape as save_upload.
    Raises ValueError if data is empty.
    """
    if not data:
        raise ValueError("The downloaded file is empty")
    stored = f"{uuid.uuid4().hex}{suffix}"
    dest = os.path.join(config.UPLOAD_DIR, stored)
    with open(dest, "wb") as fh:
        fh.write(data)
    return {"path": dest, "name": original_name or stored,
            "stored_name": stored, "size": len(data)}


# ---------------------------------------------------------------- ffprobe

def ffprobe_json(path):
    """
    Run ffprobe and return the container/stream metadata as a dict.

    path (str): media file to probe.

    Returns (dict): parsed ffprobe JSON with "format" and "streams" keys.
    Returns {} if ffprobe is missing, times out after 30s, or the file is not
    media - callers treat an empty dict as "no metadata available".
    """
    cmd = [config.FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=30,
                             creationflags=_no_window())
        if out.returncode != 0:
            return {}
        return json.loads(out.stdout.decode("utf-8", "replace"))
    except Exception:
        return {}


def media_duration(path):
    """
    Length of a video or audio file in seconds.

    path (str): media file.

    Returns (float|None): duration in seconds, or None when the container has
    no duration field (some streamed MP4s) or ffprobe failed.
    """
    info = ffprobe_json(path)
    raw = (info.get("format") or {}).get("duration")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _no_window():
    """
    Flag that stops a console window flashing on every subprocess on Windows.

    Returns (int): subprocess.CREATE_NO_WINDOW on Windows, 0 elsewhere so the
    same call site works on Linux and macOS without branching.
    """
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


# ---------------------------------------------------------------- extraction

def extract_frames(video_path, job_id, fps=None, max_frames=None):
    """
    Sample frames out of a video at a fixed rate.

    video_path (str): source video.
    job_id (str): short id used to namespace the written frame files.
    fps (float|None): frames to sample per second, defaults to
        config.VIDEO_SAMPLE_FPS.
    max_frames (int|None): hard cap, defaults to config.VIDEO_MAX_FRAMES. The
        cap protects a 4 GB GPU from a long clip.

    Returns (list): [{"t": seconds into the video (float),
    "path": abs path to the written JPEG, "index": frame number}], in time
    order. Returns [] if the video cannot be opened or has no readable frames;
    the caller then reports the signal as an error rather than crashing.
    """
    import cv2

    fps = fps or config.VIDEO_SAMPLE_FPS
    max_frames = max_frames or config.VIDEO_MAX_FRAMES

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0 or src_fps > 240:
        src_fps = 25.0            # broken metadata, assume something sane
    step = max(1, int(round(src_fps / fps)))   # take every Nth decoded frame

    out_dir = os.path.join(config.WORK_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                dest = os.path.join(out_dir, f"frame_{len(frames):03d}.jpg")
                if cv2.imwrite(dest, frame):
                    frames.append({"t": round(index / src_fps, 2),
                                   "path": dest, "index": index})
            index += 1
    finally:
        cap.release()
    return frames


def extract_audio_track(video_path, job_id):
    """
    Pull the audio out of a video as 16 kHz mono WAV, ready for wav2vec2.

    video_path (str): source video.
    job_id (str): short id used to namespace the written file.

    Returns (str|None): path to the WAV, or None when the video has no audio
    stream, ffmpeg is missing, or conversion times out after 120s. None is a
    normal outcome (silent video), not an error.
    """
    info = ffprobe_json(video_path)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if not has_audio:
        return None
    return to_wav_16k(video_path, job_id)


def to_wav_16k(src_path, job_id):
    """
    Convert any audio or video input to 16 kHz mono WAV.

    src_path (str): source file, any format ffmpeg understands.
    job_id (str): short id used to namespace the written file.

    Returns (str|None): path to the converted WAV, or None if ffmpeg failed,
    timed out after 120s, or produced an empty file. The audio analyzer treats
    None as "cannot score this" and reports an error signal.
    """
    out_dir = os.path.join(config.WORK_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "audio_16k.wav")

    cmd = [config.FFMPEG_BIN, "-y", "-i", src_path,
           "-vn",                                   # drop any video stream
           "-ac", "1",                              # mono
           "-ar", str(config.AUDIO_SAMPLE_RATE),    # 16 kHz
           "-t", str(config.AUDIO_MAX_SECONDS),     # never longer than the cap
           "-f", "wav", dest]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=120,
                             creationflags=_no_window())
        if res.returncode != 0 or not os.path.exists(dest):
            return None
        return dest if os.path.getsize(dest) > 1024 else None
    except Exception:
        return None


def read_wav(path):
    """
    Load a 16 kHz mono WAV into a float array for the audio model.

    path (str): WAV file, expected to already be 16 kHz mono.

    Returns (tuple): (samples, sample_rate) where samples is a 1-D numpy float32
    array normalised to roughly -1..1. Returns (None, 0) if the file cannot be
    parsed, so the caller can report an error signal.
    """
    import wave

    import numpy as np

    try:
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
    except Exception:
        return None, 0

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        return None, 0

    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)   # fold down to mono
    peak = float(np.iinfo(dtype).max)
    return data / peak, rate


def build_media_context(saved, job_id):
    """
    Assemble the single dict every analyzer receives.

    saved (dict): output of save_upload or save_bytes.
    job_id (str): short id for this analysis run, used for scratch filenames.

    Returns (dict): {"path", "name", "size", "job_id", "type", "sha256",
    "duration" (float|None), "probe" (ffprobe dict), "wav" (str|None),
    "frames" (list, filled in later by the video analyzer)}.
    Frame and WAV extraction happen here once so the video and audio analyzers
    do not each pay for their own ffmpeg pass.
    """
    media = dict(saved)
    media["job_id"] = job_id
    media["type"] = detect_media_type(saved["path"])
    media["sha256"] = sha256_file(saved["path"])
    media["probe"] = ffprobe_json(saved["path"]) if media["type"] != "image" else {}
    media["duration"] = None
    media["wav"] = None
    media["frames"] = []

    if media["type"] in ("video", "audio"):
        raw = (media["probe"].get("format") or {}).get("duration")
        try:
            media["duration"] = float(raw)
        except (TypeError, ValueError):
            media["duration"] = None

    return media
