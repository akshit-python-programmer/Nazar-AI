"""
Still-image signals: the neural deepfake classifier and Error Level Analysis.

The classifier is loaded once into a module-level singleton and shared with
video_checks.py, which feeds it face crops frame by frame. ELA is the classic
resave-and-diff trick for spotting locally edited regions in a JPEG.
Both entry points take the media context and return the standard signal dict.
"""

import os
import time

import config
import media_utils

# Loaded on first use, then reused for the life of the process. Plain module
# variables, so there is no class and no second copy of the weights on the 4 GB
# card. _MODEL_ID records which id actually loaded, primary or fallback.
_PROCESSOR = None
_MODEL = None
_MODEL_ID = None
_LOAD_ERROR = None


def get_image_model():
    """
    Return the image classifier, loading it on first call.

    Returns (tuple): (processor, model, model_id). All None if no model could
    be loaded, in which case the reason is cached in _LOAD_ERROR and every
    later call returns immediately instead of retrying a slow download.
    Tries config.IMAGE_MODEL_ID first (a local weights folder wins over the
    repo id when one is present), then IMAGE_MODEL_FALLBACK_ID.

    Loaded directly through AutoModel rather than the transformers pipeline
    because the active detector has a single-logit sigmoid head
    (num_labels == 1), which the image-classification pipeline mishandles:
    softmax over one logit is always 1.0. Direct loading serves both that
    head shape and ordinary two-label softmax models through one code path.
    """
    global _PROCESSOR, _MODEL, _MODEL_ID, _LOAD_ERROR

    if _MODEL is not None or _LOAD_ERROR is not None:
        return _PROCESSOR, _MODEL, _MODEL_ID

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "cuda" if config.get_device() == 0 else "cpu"
    errors = []
    seen = set()
    ordered = []
    for candidate in getattr(config, "IMAGE_MODEL_CANDIDATES", (config.IMAGE_MODEL_ID,)):
        resolved = config.resolve_model(candidate, config.IMAGE_MODEL_LOCAL_DIR)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    if config.IMAGE_MODEL_FALLBACK_ID not in seen:
        ordered.append(config.resolve_model(config.IMAGE_MODEL_FALLBACK_ID, config.IMAGE_MODEL_LOCAL_DIR))

    for model_id in ordered:
        try:
            started = time.time()
            _PROCESSOR = AutoImageProcessor.from_pretrained(model_id)
            _MODEL = AutoModelForImageClassification.from_pretrained(model_id)
            _MODEL.to(device).eval()
            # Report the published model name even when the weights were loaded
            # from a local folder, so reports do not show a machine path.
            _MODEL_ID = (config.IMAGE_MODEL_ID
                         if model_id == config.IMAGE_MODEL_LOCAL_DIR else model_id)
            print(f"[image] loaded {_MODEL_ID} on {device} "
                  f"in {time.time() - started:.1f}s "
                  f"(head: {_MODEL.config.num_labels} logit(s))")
            return _PROCESSOR, _MODEL, _MODEL_ID
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    _LOAD_ERROR = " | ".join(errors)
    print(f"[image] no image model available: {_LOAD_ERROR}")
    return None, None, None


def score_from_predictions(preds):
    """
    Collapse a label-dict classifier output into one synthetic score.

    preds (list): [{"label": str, "score": float}, ...] as returned by a
        transformers classification pipeline for a single input. The image
        path no longer produces this shape (it scores logits directly in
        classify_images), but the audio check still runs through the HF
        pipeline and shares this mapping.

    Returns (float|None): probability that the input is synthetic, 0..1.
    Labels are matched case-insensitively against config.FAKE_LABELS and
    REAL_LABELS. When both labels are present the score is normalised as
    fake / (fake + real), which is a no-op for softmax models and corrects
    sigmoid-head models whose per-label scores can sum past 1. Returns None
    when no label can be interpreted, which the caller reports rather than
    guessing a direction.
    """
    fake_p = None
    real_p = None
    for item in preds:
        label = str(item.get("label", "")).strip().lower()
        score = float(item.get("score", 0.0))
        if label in config.FAKE_LABELS:
            fake_p = score
        elif label in config.REAL_LABELS:
            real_p = score

    if fake_p is not None and real_p is not None:
        total = fake_p + real_p
        return fake_p / total if total > 0 else 0.5
    if fake_p is not None:
        return fake_p
    if real_p is not None:
        return 1.0 - real_p
    return None


def _fake_index(model_config):
    """
    Which output index of a multi-label head means "synthetic".

    model_config: the loaded model's config, carrying id2label.

    Returns (int|None): the index whose label matches config.FAKE_LABELS, or
    None when no label can be interpreted. Matching is case-insensitive so both
    {0: "Fake"} and {1: "fake"} orderings work without special casing.
    """
    id2label = getattr(model_config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).strip().lower() in config.FAKE_LABELS:
            return int(idx)
    return None


def blend_detection_scores(full_frame=None, face_crop=None, has_face=False):
    """
    Combine the full-frame and face-crop predictions without over-weighting one
    view. A face-swapped image is usually near-real across the whole frame but
    highly suspicious in the crop, while a fully generated image is suspicious in
    both views. We therefore blend the two when a face is present and otherwise
    fall back to the full-frame judgement.

    Returns (float|None): the blended synthetic probability, 0..1.
    """
    if full_frame is None and face_crop is None:
        return None
    if full_frame is None:
        return float(face_crop)
    if face_crop is None:
        return float(full_frame)

    full_frame = float(full_frame)
    face_crop = float(face_crop)

    if not has_face:
        return full_frame

    # The crop is more informative for identity swaps, especially when the rest of
    # the image is a real background. The full frame still matters for fully
    # synthetic portraits, so keep both in the mix rather than selecting the max.
    blended = 0.4 * full_frame + 0.6 * face_crop
    return max(0.0, min(1.0, blended))


def classify_images(images):
    """
    Score a batch of images for synthetic-ness. The single inference entry
    point: the still-image signal, the video frame loop and the occlusion
    explainer all come through here.

    images (list): list of PIL.Image objects, already RGB.

    Returns (list): one float or None per input image, same order, where the
    float is the probability the image is synthetic (0 authentic .. 1
    synthetic). Returns a list of Nones if the model is unavailable. Raises
    nothing; a failure degrades to Nones so a single bad frame cannot kill a
    whole video run.

    Head handling: a single-logit model (the active CommunityForensics
    detector) is scored as sigmoid(logit) = P(generated), which is how it was
    trained. A two-label model is softmaxed and read at the index whose label
    means fake.
    """
    processor, model, _ = get_image_model()
    if model is None or not images:
        return [None] * len(images)

    import torch

    device = next(model.parameters()).device
    scores = []
    batch_size = max(1, config.VIDEO_BATCH_SIZE)
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            with torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(device)
                logits = model(**inputs).logits
            if model.config.num_labels == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
                scores.extend(float(p) for p in probs)
            else:
                fake_idx = _fake_index(model.config)
                if fake_idx is None:
                    scores.extend([None] * len(batch))
                    continue
                probs = torch.softmax(logits, dim=-1)[:, fake_idx]
                scores.extend(float(p) for p in probs)
    except Exception as exc:
        print(f"[image] batch inference failed: {exc}")
        return [None] * len(images)

    return scores


def describe_image_score(score, model_id):
    """
    Turn a raw score into a sentence a non-technical reader can act on.

    score (float): synthetic probability 0..1.
    model_id (str): the model that produced it, named for transparency.

    Returns (str): one or two plain sentences. Deliberately hedged in the middle
    of the range, because that is where these classifiers are least reliable.
    """
    pct = int(round(score * 100))
    short_id = (model_id or "the detector").split("/")[-1]
    if score >= 0.85:
        return (f"The image classifier is {pct}% confident this was AI-generated or "
                f"face-swapped. Texture and edge patterns match known generator output.")
    if score >= 0.65:
        return (f"The classifier leans synthetic at {pct}%. There are generation-like "
                f"artefacts, though not the textbook pattern.")
    if score >= 0.35:
        return (f"The classifier is undecided at {pct}%. This range is genuinely "
                f"ambiguous, so it should not be read as evidence either way.")
    if score >= 0.15:
        return (f"The classifier leans authentic, scoring only {pct}% synthetic. "
                f"Noise and texture look camera-like.")
    return (f"The classifier sees no generation artefacts, scoring {pct}% synthetic "
            f"({short_id}).")


def analyze_image_model(media):
    """
    Signal: run the deepfake classifier on a still image.

    media (dict): media context. Skipped unless media["type"] == "image";
        video is handled by video_checks.py, which does its own frame loop.

    Returns (dict): standard signal dict, signal name "face_manipulation".
    status is "not_applicable" for non-images, "error" if the model could not
    load or the file could not be opened, otherwise "ok" with a score.
    details carries the model id and the raw label probabilities.
    """
    if media.get("type") != "image":
        return media_utils.na_signal("face_manipulation", "AI Face Manipulation",
                                     "Runs on still images. Video frames are "
                                     "scored by the video check instead.")

    from PIL import Image

    _, model, model_id = get_image_model()
    if model is None:
        return media_utils.error_signal("face_manipulation", "AI Face Manipulation",
                                        _LOAD_ERROR or "model unavailable")

    full_image = Image.open(media["path"]).convert("RGB")

    # Score the whole frame AND the face crop, and keep the worse (higher) of
    # the two. A fully generated image reads strongest full-frame, which is how
    # the detector was trained. A face swapped into a real photo is the
    # opposite: most pixels are genuine camera output and dilute the full-frame
    # score, so only the face crop exposes it. Taking the max covers both.
    # Imported here rather than at module level because video_checks imports
    # this module, and a top-level import would be circular.
    face_found = False
    face_image = None
    crop_path = None
    # Path of the image the heatmap should explain: whichever view produced
    # the score that is being reported.
    scored_path = media["path"]
    try:
        import cv2
        import numpy as np

        import video_checks
        bgr = cv2.cvtColor(np.asarray(full_image), cv2.COLOR_RGB2BGR)
        crop, face_found = video_checks.crop_largest_face(bgr)
        if face_found:
            face_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_path = os.path.join(config.WORK_DIR, f"{media['job_id']}_face.png")
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            if not cv2.imwrite(crop_path, crop):
                crop_path = None
    except Exception as exc:
        print(f"[image] face crop skipped: {exc}")

    views = [full_image] + ([face_image] if face_image is not None else [])
    results = classify_images(views)
    full_score = results[0]
    crop_score = results[1] if len(results) > 1 else None

    usable = [s for s in results if s is not None]
    if not usable:
        return media_utils.error_signal(
            "face_manipulation", "AI Face Manipulation",
            "the classifier could not score this image")
    score = blend_detection_scores(full_score, crop_score, face_found)

    # Point the heatmap at whichever view actually produced the reported score.
    if (crop_score is not None and score == crop_score
            and face_found and crop_path):
        scored_path = crop_path

    # Confidence is highest when the model commits. A score sitting near 0.5
    # means the model itself is unsure, so the signal should carry less weight.
    confidence = min(1.0, 0.45 + abs(score - 0.5) * 1.1)
    if not face_found:
        # With no face found there is only the full-frame view; still a valid
        # input for this detector, but one eye instead of two.
        confidence *= 0.8
    confidence = round(confidence, 3)

    details = {"model": model_id,
               "face_detected": face_found,
               "full_frame_score": None if full_score is None else round(full_score, 4),
               "face_crop_score": None if crop_score is None else round(crop_score, 4),
               "combined_score": round(score, 4)}

    # A heatmap is a nice-to-have. If the explainer is slow or broken the
    # verdict must still ship, so nothing here is allowed to raise.
    try:
        import explainer
        heatmap = explainer.explain_image(scored_path, media["job_id"])
        details["heatmaps"] = [heatmap] if heatmap else []
    except Exception as exc:
        print(f"[image] explainer skipped: {exc}")
        details["heatmaps"] = []

    return media_utils.make_signal(
        "face_manipulation", "AI Face Manipulation",
        synthetic_score=score,
        confidence=confidence,
        human_note=describe_image_score(score, model_id),
        details=details,
    )


# ---------------------------------------------------------------- ELA

def analyze_ela(media):
    """
    Signal: Error Level Analysis on a still image.

    media (dict): media context. Still images only.

    How it works: a JPEG that has been saved once compresses evenly, so
    re-saving it at quality 90 and subtracting produces a near-uniform
    difference. A region that was pasted or retouched has a different
    compression history and lights up brighter than its surroundings. The score
    is the unevenness of that difference, not its brightness.

    Returns (dict): standard signal dict, signal "error_level_analysis".
    details["ela_image"] is the URL of the saved visualisation.
    status is "not_applicable" for non-images and "error" if the image cannot
    be read or re-encoded.
    """
    if media.get("type") != "image":
        return media_utils.na_signal("error_level_analysis", "Error Level Analysis",
                                     "Applies to still images only.")

    import numpy as np
    from PIL import Image, ImageChops

    original = Image.open(media["path"]).convert("RGB")

    # Re-encode through JPEG in memory, then diff against the original.
    scratch = os.path.join(config.WORK_DIR, f"{media['job_id']}_ela_tmp.jpg")
    os.makedirs(os.path.dirname(scratch), exist_ok=True)
    original.save(scratch, "JPEG", quality=config.ELA_QUALITY)
    resaved = Image.open(scratch).convert("RGB")

    diff = ImageChops.difference(original, resaved)
    arr = np.asarray(diff).astype(np.float32).mean(axis=2)   # grayscale error map

    # Block means smooth out per-pixel noise so we measure regional unevenness,
    # which is what an edit actually produces.
    block = 16
    h, w = arr.shape
    bh, bw = h // block, w // block
    if bh < 2 or bw < 2:
        blocks = arr.reshape(1, -1).mean(axis=1)
    else:
        trimmed = arr[:bh * block, :bw * block]
        blocks = trimmed.reshape(bh, block, bw, block).mean(axis=(1, 3)).ravel()

    mean_err = float(blocks.mean())
    std_err = float(blocks.std())
    # Coefficient of variation: high when a few regions differ much more than
    # the rest, which is the ELA signature of a local edit.
    cv = std_err / (mean_err + 1e-6)

    # Empirical mapping. An untouched JPEG typically lands around 0.3-0.6.
    score = max(0.0, min(1.0, (cv - 0.35) / 0.9))

    # Save the amplified visualisation for the UI and the PDF.
    amplified = diff.point(lambda p: min(255, p * config.ELA_SCALE))
    out_path = os.path.join(config.ELA_DIR, f"{media['job_id']}_ela.png")
    amplified.save(out_path)
    try:
        os.remove(scratch)
    except OSError:
        pass

    source_ext = os.path.splitext(media["path"])[1].lower()
    if source_ext in (".png", ".bmp", ".webp"):
        # Lossless or re-encoded sources have no original JPEG history to read,
        # so the result means much less. Say so and drop the confidence.
        confidence = 0.2
        note = ("This file is not a camera JPEG, so error level analysis has little "
                "compression history to work with. Treat the result as weak context.")
    elif score >= 0.6:
        confidence = 0.5
        note = ("Compression error is uneven across the image. Some regions have a "
                "different save history than their surroundings, which is what "
                "pasting or retouching leaves behind.")
    elif score >= 0.35:
        confidence = 0.4
        note = ("Compression error is slightly uneven. That can mean a local edit, "
                "but heavy sharpening or a messaging app re-encode does it too.")
    else:
        confidence = 0.45
        note = ("Compression error is spread evenly across the frame, which is what "
                "an unedited single-save image looks like.")

    return media_utils.make_signal(
        "error_level_analysis", "Error Level Analysis",
        synthetic_score=score,
        confidence=confidence,
        human_note=note,
        details={"ela_image": config.static_url(out_path),
                 "mean_error": round(mean_err, 3),
                 "error_variation": round(cv, 3),
                 "quality": config.ELA_QUALITY},
    )
