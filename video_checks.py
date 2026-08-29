"""
Video signal: sample frames, find the face in each, score them, build a timeline.

Frames are taken at one per second and capped, so a long clip cannot exhaust a
4 GB card. Each frame is cropped to the detected face before classification,
because these detectors were trained on faces and a wide shot mostly feeds them
background. Frames with no detectable face are still scored, but count less.
The per-frame scores become details["timeline"], which drives the UI sparkline.
"""

import os

import config
import image_checks
import media_utils

# OpenCV's bundled Haar cascade, loaded once. Not a great face detector by
# modern standards, but it needs no download and no extra VRAM, which matters
# more here than raw accuracy.
_FACE_CASCADE = None


def get_face_detector():
    """
    Return the Haar cascade face detector, loading it on first call.

    Returns (cv2.CascadeClassifier|None): the detector, or None if OpenCV has no
    bundled cascade file. A None detector is not fatal: every frame is then
    scored full-frame at the reduced weight.
    """
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    import cv2

    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(path):
        return None
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    _FACE_CASCADE = cascade
    return _FACE_CASCADE


def crop_largest_face(bgr_frame):
    """
    Cut the biggest face out of a frame, with a margin around it.

    bgr_frame (numpy.ndarray): frame as OpenCV loads it, BGR channel order.

    Returns (tuple): (crop, found) where crop is a BGR ndarray and found is a
    bool. When no face is detected, crop is the untouched frame and found is
    False, so the caller can down-weight that frame rather than skip it.
    The margin matters because deepfake artefacts often sit on the jaw and
    hairline, just outside a tight detector box.
    """
    import cv2

    detector = get_face_detector()
    if detector is None:
        return bgr_frame, False

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                      minSize=(48, 48))
    if len(faces) == 0:
        return bgr_frame, False

    # Largest box wins: the subject of the clip, not a bystander.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    margin_x = int(w * config.FACE_CROP_MARGIN)
    margin_y = int(h * config.FACE_CROP_MARGIN)
    height, width = bgr_frame.shape[:2]
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(width, x + w + margin_x)
    y1 = min(height, y + h + margin_y)

    return bgr_frame[y0:y1, x0:x1], True


def _score_frames(frames):
    """
    Run the classifier over every sampled frame.

    frames (list): [{"t": float, "path": str, "index": int}] from
        media_utils.extract_frames.

    Returns (list): [{"t", "path", "score" (float|None), "has_face" (bool),
    "weight" (float)}] in time order. Frames the model could not score keep a
    None score and are dropped from the aggregate by the caller.
    Work happens in batches of config.VIDEO_BATCH_SIZE to bound VRAM.
    """
    import cv2
    from PIL import Image

    scored = []
    batch_images = []
    batch_meta = []

    def flush():
        """Score whatever is queued and append the results."""
        if not batch_images:
            return
        results = image_checks.classify_images(batch_images)
        for meta, score in zip(batch_meta, results):
            meta["score"] = score
            scored.append(meta)
        batch_images.clear()
        batch_meta.clear()

    for frame in frames:
        bgr = cv2.imread(frame["path"])
        if bgr is None:
            continue
        crop, has_face = crop_largest_face(bgr)
        # OpenCV is BGR, PIL wants RGB.
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        batch_images.append(pil)
        batch_meta.append({"t": frame["t"], "path": frame["path"],
                           "has_face": has_face,
                           "weight": 1.0 if has_face else config.NO_FACE_FRAME_WEIGHT})

        if len(batch_images) >= config.VIDEO_BATCH_SIZE:
            flush()
    flush()

    scored.sort(key=lambda f: f["t"])
    return scored


def _summarise(scored, face_frames, total):
    """
    Write the human-readable note for a scored video.

    scored (list): frames that got a score.
    face_frames (int): how many of them had a detectable face.
    total (int): frames sampled overall.

    Returns (str): one or two plain sentences naming how many frames looked
    synthetic, since "41 of 52 frames" is far more convincing to a reader than
    a bare percentage.
    """
    if not scored:
        return "No frames from this video could be scored."

    suspicious = [f for f in scored if (f["score"] or 0) >= 0.6]
    count = len(suspicious)
    n = len(scored)

    if face_frames == 0:
        where = ("No face was detected in any sampled frame, so the whole frame was "
                 "scored instead. That is a much noisier signal than a face crop.")
    else:
        where = f"A face was found and analysed in {face_frames} of {total} sampled frames."

    if count == 0:
        return (f"None of the {n} scored frames show generation artefacts. {where}")
    if count == n:
        return (f"All {n} scored frames show blending and texture artefacts typical "
                f"of AI-generated or face-swapped video. {where}")
    return (f"{count} of {n} scored frames show artefacts typical of AI-generated or "
            f"face-swapped video. {where}")


def analyze_video_frames(media):
    """
    Signal: score a video frame by frame and produce a suspicion timeline.

    media (dict): media context. Video only; images and audio are handled by
        their own analyzers.

    Returns (dict): standard signal dict, signal "video_face_manipulation".
    details carries:
      "timeline": [{"t": seconds, "score": 0..1}] for the UI sparkline
      "frames_sampled", "frames_scored", "frames_with_face"
      "top_frames": the most suspicious frames, for the explainer and PDF
      "heatmaps": overlay URLs, when the explainer managed to build them
    status is "not_applicable" for non-video, "error" if no frame could be read
    or the classifier is unavailable.
    """
    if media.get("type") != "video":
        return media_utils.na_signal("video_face_manipulation",
                                     "AI Face Manipulation (Video)",
                                     "Applies to video files only.")

    frames = media_utils.extract_frames(media["path"], media["job_id"])
    if not frames:
        return media_utils.error_signal("video_face_manipulation",
                                        "AI Face Manipulation (Video)",
                                        "no frames could be read from this video")

    # Hand the frame list back so other code (the report, cleanup) can find it.
    media["frames"] = frames

    scored = _score_frames(frames)
    usable = [f for f in scored if f["score"] is not None]
    if not usable:
        return media_utils.error_signal(
            "video_face_manipulation", "AI Face Manipulation (Video)",
            "the image classifier could not score any frame of this video")

    # Weighted mean: face crops count fully, full-frame guesses count less.
    total_weight = sum(f["weight"] for f in usable)
    overall = sum(f["score"] * f["weight"] for f in usable) / total_weight

    face_frames = sum(1 for f in usable if f["has_face"])
    timeline = [{"t": f["t"], "score": round(f["score"], 4)} for f in usable]

    # Most suspicious frames, for the explainer and the PDF.
    top_frames = sorted(usable, key=lambda f: f["score"], reverse=True)[:config.EXPLAIN_TOP_FRAMES]

    details = {
        "timeline": timeline,
        "frames_sampled": len(frames),
        "frames_scored": len(usable),
        "frames_with_face": face_frames,
        "model": image_checks._MODEL_ID,
        "top_frames": [{"t": f["t"], "score": round(f["score"], 4),
                        "url": config.static_url(f["path"])} for f in top_frames],
    }

    # Heatmaps are a nice-to-have. If the explainer is slow or broken the
    # verdict must still ship, so nothing here is allowed to raise.
    try:
        import explainer
        details["heatmaps"] = explainer.explain_frames(
            [f["path"] for f in top_frames], media["job_id"])
    except Exception as exc:
        print(f"[video] explainer skipped: {exc}")
        details["heatmaps"] = []

    # Confidence rises with how much face evidence there was. A clip where the
    # detector never found a face is a much weaker basis for a verdict.
    face_ratio = face_frames / max(1, len(usable))
    confidence = round(min(0.95, 0.35 + 0.45 * face_ratio + min(0.15, len(usable) / 100)), 3)

    return media_utils.make_signal(
        "video_face_manipulation", "AI Face Manipulation (Video)",
        synthetic_score=overall,
        confidence=confidence,
        human_note=_summarise(usable, face_frames, len(frames)),
        details=details,
    )
