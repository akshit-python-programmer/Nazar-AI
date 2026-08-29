"""
Metadata forensics: what the file says about its own history.

Reads EXIF from images and the container tags from video/audio, then looks for
things that do not add up - no camera fields at all, an editor or AI generator
named in the software tag, timestamps that disagree with each other.
Metadata is circumstantial, so this signal usually returns no score and speaks
through confidence and its note instead. See the comment above _decide_score.
"""

import os
import re

import config
import media_utils

# Software strings that mean a human edited the file in a desktop editor.
EDITOR_HINTS = [
    "photoshop", "gimp", "lightroom", "affinity", "paint.net", "pixelmator",
    "snapseed", "picsart", "facetune", "canva", "capcut", "premiere",
    "after effects", "davinci", "vegas", "final cut",
]

# Strings that name an AI generator outright. Much stronger evidence than an
# editor tag, because cameras never write these.
AI_HINTS = [
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dalle",
    "firefly", "imagen", "flux", "comfyui", "automatic1111", "invokeai",
    "novelai", "leonardo.ai", "runway", "pika", "sora", "kling", "hailuo",
    "elevenlabs", "heygen", "synthesia", "d-id", "veo", "grok-imagine",
    "nano banana", "seedream", "wan2", "hunyuan",
]

# Encoder/handler names that show up in AI-generated or heavily reprocessed
# video containers. Lavf on its own is not suspicious, plenty of honest tools
# use ffmpeg, so it is scored as weak context rather than a red flag.
VIDEO_TOOL_HINTS = ["lavf", "lavc", "handbrake", "capcut", "kapwing", "veed"]


def _exif_dict(path):
    """
    Read EXIF from an image as readable tag names.

    path (str): image file.

    Returns (dict): {tag name (str): value}, empty if the image has no EXIF or
    cannot be parsed. Values are coerced to str and trimmed to 300 chars so a
    huge embedded blob cannot bloat the JSON response.
    """
    from PIL import ExifTags, Image

    try:
        with Image.open(path) as img:
            raw = img.getexif()
            if not raw:
                return {}
            out = {}
            for tag_id, value in raw.items():
                name = ExifTags.TAGS.get(tag_id, str(tag_id))
                text = str(value)
                out[name] = text[:300]
            return out
    except Exception:
        return {}


def _find_hint(haystack, needles):
    """
    First matching needle inside a blob of text.

    haystack (str): text to search, already lowercased by the caller.
    needles (list): lowercase substrings to look for.

    Returns (str|None): the needle that matched, or None if none did.
    """
    for needle in needles:
        if needle in haystack:
            return needle
    return None


def _image_findings(path):
    """
    Collect metadata observations about a still image.

    path (str): image file.

    Returns (tuple): (findings, fields) where findings is a list of
    {"text": str, "weight": float, "direction": "synthetic"|"authentic"} and
    fields is the EXIF dict surfaced for display. Weight is how much that one
    observation should move the reader, 0..1.
    """
    exif = _exif_dict(path)
    findings = []
    blob = " ".join(f"{k} {v}" for k, v in exif.items()).lower()

    ai_hit = _find_hint(blob, AI_HINTS)
    editor_hit = _find_hint(blob, EDITOR_HINTS)
    make = exif.get("Make", "").strip()
    model = exif.get("Model", "").strip()

    if ai_hit:
        findings.append({
            "text": f"The file's own metadata names an AI generator ({ai_hit}).",
            "weight": 0.95, "direction": "synthetic"})
    if editor_hit:
        findings.append({
            "text": f"Metadata shows the file passed through {editor_hit.title()}. "
                    f"That is common and innocent on its own, but it does mean the "
                    f"pixels are not straight out of a camera.",
            "weight": 0.4, "direction": "synthetic"})

    if not exif:
        findings.append({
            "text": "The image carries no EXIF metadata at all. Cameras always write "
                    "it, but so does almost every social platform strip it, so this "
                    "is only meaningful if the file is claimed to be an original.",
            "weight": 0.25, "direction": "synthetic"})
    elif make or model:
        # Join separately so a missing make or model does not leave a stray space.
        camera = " ".join(part for part in (make, model) if part)
        findings.append({
            "text": f"Camera fields are present and intact ({camera}), which is "
                    f"what an unmodified camera original looks like.",
            "weight": 0.5, "direction": "authentic"})
    else:
        findings.append({
            "text": "EXIF is present but the camera make and model are missing, "
                    "which usually means the file was re-saved by software.",
            "weight": 0.3, "direction": "synthetic"})

    # Creation vs modification timestamps. A real camera writes these within a
    # second of each other; an edit-and-resave leaves them far apart.
    original = exif.get("DateTimeOriginal", "")
    modified = exif.get("DateTime", "")
    if original and modified and original != modified:
        findings.append({
            "text": f"The photo was taken at {original} but last written at "
                    f"{modified}. A gap like that means it was re-saved after capture.",
            "weight": 0.45, "direction": "synthetic"})

    return findings, exif


def _container_findings(probe):
    """
    Collect metadata observations about a video or audio container.

    probe (dict): parsed ffprobe output from media_utils.ffprobe_json.

    Returns (tuple): (findings, fields) with the same shapes as
    _image_findings. fields carries encoder, creation time, codecs and handler
    names so the UI can show them whether or not anything looked wrong.
    """
    fmt = probe.get("format") or {}
    tags = {k.lower(): str(v) for k, v in (fmt.get("tags") or {}).items()}
    streams = probe.get("streams") or []

    handlers = []
    codecs = []
    for s in streams:
        codecs.append(f"{s.get('codec_type', '?')}:{s.get('codec_name', '?')}")
        stream_tags = {k.lower(): str(v) for k, v in (s.get("tags") or {}).items()}
        if stream_tags.get("handler_name"):
            handlers.append(stream_tags["handler_name"])

    fields = {
        "encoder": tags.get("encoder", ""),
        "creation_time": tags.get("creation_time", ""),
        "format": fmt.get("format_long_name", ""),
        "codecs": codecs,
        "handlers": handlers,
        "major_brand": tags.get("major_brand", ""),
    }

    findings = []
    blob = " ".join(list(tags.values()) + handlers).lower()

    ai_hit = _find_hint(blob, AI_HINTS)
    if ai_hit:
        findings.append({
            "text": f"The container metadata names an AI tool ({ai_hit}).",
            "weight": 0.95, "direction": "synthetic"})

    tool_hit = _find_hint(blob, VIDEO_TOOL_HINTS)
    if tool_hit and not ai_hit:
        findings.append({
            "text": f"The file was written by a re-encoding tool ({tool_hit}) rather "
                    f"than a camera. Most shared clips are re-encoded at some point, "
                    f"so this is context, not an accusation.",
            "weight": 0.2, "direction": "synthetic"})

    if not tags.get("creation_time"):
        findings.append({
            "text": "The container has no creation timestamp, which recording apps "
                    "normally write.",
            "weight": 0.2, "direction": "synthetic"})

    if not findings:
        findings.append({
            "text": "Container metadata looks ordinary. Nothing in the encoder or "
                    "timestamps stands out.",
            "weight": 0.35, "direction": "authentic"})

    return findings, fields


def _decide_score(findings):
    """
    Decide whether metadata should contribute a number at all.

    findings (list): the findings list built above.

    Returns (tuple): (synthetic_score or None, confidence).

    Metadata is circumstantial. A missing EXIF block proves nothing on its own,
    so in the normal case this returns None and the signal only informs the
    reader. A score is emitted in exactly two situations, both near-conclusive:
    a generator named in the file's own metadata, and an intact camera
    fingerprint. Everything in between stays unscored on purpose, so fusion
    cannot drift a verdict on weak circumstantial evidence.
    """
    strong_synth = [f for f in findings
                    if f["direction"] == "synthetic" and f["weight"] >= 0.9]
    if strong_synth:
        return 0.92, 0.85

    strong_auth = [f for f in findings
                   if f["direction"] == "authentic" and f["weight"] >= 0.5]
    if strong_auth:
        return 0.25, 0.45

    # Confidence still reflects how much there was to say, even with no score.
    suspicious = sum(f["weight"] for f in findings if f["direction"] == "synthetic")
    return None, round(min(0.5, 0.15 + suspicious * 0.25), 3)


def analyze_metadata(media):
    """
    Signal: read the file's own metadata and report what does not add up.

    media (dict): media context. Handles images via EXIF and video/audio via
        the ffprobe output already stored in media["probe"].

    Returns (dict): standard signal dict, signal "metadata_forensics".
    synthetic_score is usually None by design (see _decide_score); details
    carries every field read, so the UI and PDF can show the raw metadata
    regardless of whether anything was flagged.
    status is "error" only if the file could not be read at all.
    """
    kind = media.get("type")

    if kind == "image":
        findings, fields = _image_findings(media["path"])
        detail_key = "exif"
    elif kind in ("video", "audio"):
        probe = media.get("probe") or media_utils.ffprobe_json(media["path"])
        if not probe:
            return media_utils.error_signal(
                "metadata_forensics", "Metadata Forensics",
                "ffprobe returned nothing for this file")
        findings, fields = _container_findings(probe)
        detail_key = "container"
    else:
        return media_utils.na_signal("metadata_forensics", "Metadata Forensics",
                                     "Unsupported file type for metadata analysis.")

    score, confidence = _decide_score(findings)

    # The note is the findings joined, because metadata evidence is a list of
    # small observations rather than one number.
    note = " ".join(f["text"] for f in findings)

    return media_utils.make_signal(
        "metadata_forensics", "Metadata Forensics",
        synthetic_score=score,
        confidence=confidence,
        human_note=note,
        details={detail_key: fields,
                 "findings": findings,
                 "scored": score is not None},
    )
