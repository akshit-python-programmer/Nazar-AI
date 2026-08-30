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


# The ensemble. Loaded once, kept resident, scored one member at a time.
_ENSEMBLE = None
_ENSEMBLE_ERROR = None


def get_image_models():
    """
    Load the detector ensemble, once, on first call.

    Returns (list): one dict per member that loaded, each
        {"name": str, "id": str, "processor", "model", "weight": float,
         "single_logit": bool, "fake_idx": int|None}.
    Returns [] when nothing loaded, with the reason cached in _ENSEMBLE_ERROR
    so later calls do not retry a slow download.

    Members come from config.IMAGE_ENSEMBLE. A member whose weights are
    missing or broken is skipped with a warning: the ensemble degrades to
    whatever did load rather than failing the signal outright. If no member
    loads at all, this falls back to the single-model candidate chain in
    get_image_model(), so the signal weakens instead of disappearing.

    Loaded through AutoModel rather than the transformers pipeline because one
    member has a single-logit sigmoid head (num_labels == 1), which the
    image-classification pipeline mishandles: softmax over one logit is always
    1.0. Direct loading serves that head and ordinary softmax heads alike.
    """
    global _ENSEMBLE, _ENSEMBLE_ERROR, _MODEL_ID

    if _ENSEMBLE is not None or _ENSEMBLE_ERROR is not None:
        return _ENSEMBLE or []

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "cuda" if config.get_device() == 0 else "cpu"
    loaded, errors = [], []

    for spec in getattr(config, "IMAGE_ENSEMBLE", []):
        source = config.resolve_model(spec["id"],
                                      os.path.join(config.MODELS_DIR, spec["dir"]))
        try:
            started = time.time()
            processor = AutoImageProcessor.from_pretrained(source)
            model = AutoModelForImageClassification.from_pretrained(source)
            model.to(device).eval()

            # Resolve head shape once at load time rather than per batch.
            single_logit = model.config.num_labels == 1
            fake_idx = None if single_logit else _fake_index(model.config)
            if not single_logit and fake_idx is None:
                raise ValueError(f"labels not interpretable: {model.config.id2label}")

            loaded.append({"name": spec["name"], "id": spec["id"],
                           "processor": processor, "model": model,
                           "weight": float(spec.get("weight", 1.0)),
                           "operating_point": spec.get("operating_point"),
                           "single_logit": single_logit, "fake_idx": fake_idx})
            vram = (torch.cuda.memory_allocated() / 1e6) if device == "cuda" else 0.0
            print(f"[image] + {spec['name']} "
                  f"({model.config.num_labels} logit(s)) "
                  f"{time.time() - started:.1f}s, VRAM {vram:.0f}MB")
        except Exception as exc:
            errors.append(f"{spec['name']}: {exc}")
            print(f"[image] ! skipped {spec['name']}: {str(exc)[:130]}")

    if not loaded:
        # Nothing in the ensemble worked. Fall back to the single-model chain.
        processor, model, model_id = get_image_model()
        if model is None:
            _ENSEMBLE_ERROR = " | ".join(errors) or (_LOAD_ERROR or "no models")
            return []
        single_logit = model.config.num_labels == 1
        loaded.append({"name": (model_id or "fallback").split("/")[-1],
                       "id": model_id, "processor": processor, "model": model,
                       "weight": 1.0, "single_logit": single_logit,
                       "fake_idx": None if single_logit else _fake_index(model.config)})

    _ENSEMBLE = loaded
    _MODEL_ID = " + ".join(m["name"] for m in loaded)
    print(f"[image] ensemble ready on {device}: {_MODEL_ID}")
    return _ENSEMBLE


def calibrate_score(raw, operating_point):
    """
    Map a detector's raw output onto the shared verdict scale.

    raw (float): the model's own probability, 0..1.
    operating_point (float|None): the raw score at which this model should be
        considered to be calling the image synthetic, measured on the eval
        set. None means the model is already calibrated and raw is returned.

    Returns (float): the calibrated score, 0..1, where
    config.VERDICT_SYNTHETIC_ABOVE now means what it says for this model.

    Two straight lines meeting at the operating point: [0, op] is stretched
    onto [0, threshold] and [op, 1] onto [threshold, 1]. Monotonic, so it
    never reorders results and cannot turn a lower raw score into a higher
    calibrated one; it only moves where the decision line falls. Kept
    deliberately simple over a fitted curve, because it is anchored on one
    measured number and a curve would imply precision the eval set does not
    support.
    """
    if operating_point is None:
        return raw
    threshold = config.VERDICT_SYNTHETIC_ABOVE
    op = min(max(float(operating_point), 1e-6), 1.0 - 1e-6)
    if raw <= op:
        return (raw / op) * threshold
    return threshold + ((raw - op) / (1.0 - op)) * (1.0 - threshold)


def combine_scores(per_model, strategy=None):
    """
    Reduce one image's per-member scores to a single synthetic probability.

    per_model (dict): {member_name: float|None}. Nones are ignored, so a
        member that failed on this particular image does not drag the result.
    strategy (str|None): override for config.IMAGE_ENSEMBLE_STRATEGY.

    Returns (float|None): the combined score, or None if no member produced
    one. Strategies are described in config.IMAGE_ENSEMBLE_STRATEGY.
    """
    strategy = strategy or getattr(config, "IMAGE_ENSEMBLE_STRATEGY", "max")
    vals = [v for v in per_model.values() if v is not None]
    if not vals:
        return None

    if strategy == "max":
        return max(vals)
    if strategy == "mean":
        return sum(vals) / len(vals)
    if strategy == "noisy_or":
        acc = 1.0
        for v in vals:
            acc *= (1.0 - v)
        return 1.0 - acc
    if strategy == "weighted_mean":
        weights = {m["name"]: m["weight"] for m in get_image_models()}
        num = sum(v * weights.get(k, 1.0)
                  for k, v in per_model.items() if v is not None)
        den = sum(weights.get(k, 1.0)
                  for k, v in per_model.items() if v is not None)
        return num / den if den else None
    return max(vals)


def classify_images_per_model(images, calibrate=True):
    """
    Score a batch of images with every ensemble member.

    images (list): PIL.Image objects, already RGB.
    calibrate (bool): apply each member's calibration to its output. Pass
        False when the caller still has to combine several views of the same
        image and will calibrate the combined figure itself. Calibration is
        piecewise linear, so calibrating two views and then averaging is not
        the same number as averaging and then calibrating, and the operating
        point in config was measured on the second of those.

    Returns (list): one dict per input image, same order, mapping member name
    to that member's synthetic probability (0 authentic .. 1 synthetic) or
    None where that member failed. Returns a list of empty dicts when no model
    is available. Raises nothing: a member that throws contributes Nones and
    the others still report, so one broken detector cannot kill a video run.

    Head handling: a single-logit member is scored sigmoid(logit) =
    P(generated), which is how it was trained. A multi-label member is
    softmaxed and read at the index whose label means fake.
    """
    members = get_image_models()
    if not members or not images:
        return [{} for _ in images]

    import torch

    batch_size = max(1, config.VIDEO_BATCH_SIZE)
    out = [{} for _ in images]

    for member in members:
        model = member["model"]
        processor = member["processor"]
        device = next(model.parameters()).device
        try:
            for start in range(0, len(images), batch_size):
                batch = images[start:start + batch_size]
                with torch.no_grad():
                    inputs = processor(images=batch, return_tensors="pt").to(device)
                    logits = model(**inputs).logits
                if member["single_logit"]:
                    probs = torch.sigmoid(logits.squeeze(-1))
                else:
                    probs = torch.softmax(logits, dim=-1)[:, member["fake_idx"]]
                for offset, p in enumerate(probs.tolist()):
                    out[start + offset][member["name"]] = (
                        calibrate_score(p, member["operating_point"])
                        if calibrate else p)
        except Exception as exc:
            print(f"[image] {member['name']} inference failed: {str(exc)[:110]}")
            for slot in out:
                slot.setdefault(member["name"], None)

    return out


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

    Every ensemble member scores each image and the results are reduced by
    combine_scores(). Callers that want to show each detector's opinion
    separately should use classify_images_per_model() instead.
    """
    if not images:
        return []
    return [combine_scores(per) for per in classify_images_per_model(images)]


def describe_image_score(score, model_id, per_model=None):
    """
    Turn a raw score into a sentence a non-technical reader can act on.

    score (float): the combined synthetic probability, 0..1.
    model_id (str): the models that produced it, named for transparency.
    per_model (dict|None): {member name: score|None}. When given, the note
        names which detectors flagged the file and how many agreed, because
        "3 of 4 detectors agree" is far more use to someone filing a
        complaint than a bare percentage.

    Returns (str): one or two plain sentences. Deliberately hedged in the
    middle of the range, where these classifiers are least reliable.
    """
    pct = int(round(score * 100))
    short_id = (model_id or "the detector").split("/")[-1]

    tail = ""
    if per_model:
        scored = {k: v for k, v in per_model.items() if v is not None}
        if len(scored) > 1:
            flagged = [k for k, v in scored.items() if v >= 0.5]
            if flagged and len(flagged) < len(scored):
                tail = (f" {len(flagged)} of {len(scored)} detectors flagged it"
                        f" ({', '.join(sorted(flagged))}); the rest did not.")
            elif flagged:
                tail = f" All {len(scored)} detectors agree."
            else:
                tail = f" None of the {len(scored)} detectors flagged it."

    if score >= 0.85:
        return (f"The detectors are {pct}% confident this was AI-generated or "
                f"face-swapped. Texture and edge patterns match known generator "
                f"output.{tail}")
    if score >= 0.65:
        return (f"The detectors lean synthetic at {pct}%. There are generation-like "
                f"artefacts, though not the textbook pattern.{tail}")
    if score >= 0.35:
        return (f"The detectors are undecided at {pct}%. This range is genuinely "
                f"ambiguous, so it should not be read as evidence either way.{tail}")
    if score >= 0.15:
        return (f"The detectors lean authentic, scoring only {pct}% synthetic. "
                f"Noise and texture look camera-like.{tail}")
    return (f"No generation artefacts found, scoring {pct}% synthetic "
            f"({short_id}).{tail}")


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

    members = get_image_models()
    if not members:
        return media_utils.error_signal(
            "face_manipulation", "AI Face Manipulation",
            _ENSEMBLE_ERROR or _LOAD_ERROR or "model unavailable")
    model_id = _MODEL_ID

    with Image.open(media["path"]) as img:
        full_image = img.convert("RGB")

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

    # Score the whole frame and, when there is one, the face crop. Each view is
    # scored by every member, then the two views are blended per member before
    # the members are combined. Blending inside each member keeps a specialist's
    # two views together instead of letting one member's crop score race
    # another member's full-frame score.
    # Raw scores here: the two views are blended first and the blend is what
    # gets calibrated, matching how the operating point in config was measured.
    views = [full_image] + ([face_image] if face_image is not None else [])
    per_view = classify_images_per_model(views, calibrate=False)
    full_by_model = per_view[0]
    crop_by_model = per_view[1] if len(per_view) > 1 else {}

    per_model = {}
    for member in members:
        name = member["name"]
        blended = blend_detection_scores(
            full_by_model.get(name), crop_by_model.get(name), face_found)
        per_model[name] = (None if blended is None
                           else calibrate_score(blended, member["operating_point"]))

    score = combine_scores(per_model)
    if score is None:
        return media_utils.error_signal(
            "face_manipulation", "AI Face Manipulation",
            "the classifier could not score this image")

    full_score = combine_scores(full_by_model)
    crop_score = combine_scores(crop_by_model) if crop_by_model else None

    # Point the heatmap at whichever view drove the reported score.
    if (crop_score is not None and full_score is not None
            and crop_score > full_score and face_found and crop_path):
        scored_path = crop_path

    # Confidence is highest when the models commit. A score sitting near 0.5
    # means they are unsure, so the signal should carry less weight.
    confidence = min(1.0, 0.45 + abs(score - 0.5) * 1.1)
    if not face_found:
        # With no face found there is only the full-frame view; still a valid
        # input for these detectors, but one eye instead of two.
        confidence *= 0.8
    # Agreement is itself evidence. When the members split hard, say so through
    # confidence rather than presenting a averaged number as if it were solid.
    scored = [v for v in per_model.values() if v is not None]
    if len(scored) > 1 and (max(scored) - min(scored)) > 0.5:
        confidence *= 0.75
    confidence = round(confidence, 3)

    details = {"model": model_id,
               "face_detected": face_found,
               "ensemble_strategy": getattr(config, "IMAGE_ENSEMBLE_STRATEGY", "mean"),
               # Each detector's own calibrated verdict, so the report can show
               # who flagged what instead of one opaque number.
               "per_model": {k: (None if v is None else round(v, 4))
                             for k, v in per_model.items()},
               # The two views before calibration, kept for transparency: these
               # are the models' own raw numbers, not on the verdict scale.
               "full_frame_raw": None if full_score is None else round(full_score, 4),
               "face_crop_raw": None if crop_score is None else round(crop_score, 4),
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
        human_note=describe_image_score(score, model_id, per_model),
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

    import io

    import numpy as np
    from PIL import Image, ImageChops

    with Image.open(media["path"]) as img:
        original = img.convert("RGB")

    # Re-encode through JPEG in memory, then diff against the original.
    # BytesIO skips the scratch-file write/read/delete round trip to disk.
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=config.ELA_QUALITY)
    buf.seek(0)
    with Image.open(buf) as img:
        resaved = img.convert("RGB")

    diff_arr = np.asarray(ImageChops.difference(original, resaved)).astype(np.float32)
    arr = diff_arr.mean(axis=2)   # grayscale error map

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

    # Save the amplified visualisation for the UI and the PDF. Clipping in
    # numpy is far faster than PIL's point() with a Python lambda, which runs
    # per pixel in Python.
    amplified = Image.fromarray(
        np.clip(diff_arr * config.ELA_SCALE, 0, 255).astype(np.uint8))
    out_path = os.path.join(config.ELA_DIR, f"{media['job_id']}_ela.png")
    amplified.save(out_path)

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
