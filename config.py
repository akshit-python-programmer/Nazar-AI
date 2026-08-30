"""
Central configuration for NazarAI.

Holds every tunable constant in one place: filesystem paths, upload limits,
model IDs, fusion weights and verdict thresholds. Nothing here does real work,
so any module can import it without pulling in torch or Flask. The analyzer
registry at the bottom is the list the pipeline actually iterates over.
"""

import os

# This machine has TensorFlow installed alongside Keras 3, and transformers
# refuses to import its TF model classes against Keras 3. We only ever use the
# torch path, so tell transformers not to go looking for TensorFlow at all.
# Must run before transformers is imported anywhere, which is why it lives at
# the top of the module every other module imports first.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# A Windows console defaults to cp1252, which raises UnicodeEncodeError the
# moment anything prints an emoji - and the WhatsApp verdict lines are full of
# them. Force UTF-8 on the streams and never let logging kill a request.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------- paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")     # original files as received
HEATMAP_DIR = os.path.join(STATIC_DIR, "heatmaps")   # explainer overlays
REPORT_DIR = os.path.join(STATIC_DIR, "reports")     # generated PDFs
ELA_DIR = os.path.join(STATIC_DIR, "ela")            # error-level-analysis images
WORK_DIR = os.path.join(STATIC_DIR, "work")          # extracted frames / audio scratch

# Everything the app writes to. Created on import so no route has to check.
ALL_DIRS = [STATIC_DIR, UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR, ELA_DIR, WORK_DIR]
for _d in ALL_DIRS:
    os.makedirs(_d, exist_ok=True)


def static_url(abs_path):
    """
    Turn an absolute path inside static/ into a browser-usable URL path.

    abs_path (str): absolute filesystem path that must live under STATIC_DIR.

    Returns (str): URL beginning with "/static/", forward slashes only.
    Returns None when the path is empty or sits outside STATIC_DIR, so callers
    can drop it from the JSON instead of leaking a local filesystem path.
    """
    if not abs_path:
        return None
    try:
        rel = os.path.relpath(abs_path, STATIC_DIR)
    except ValueError:
        # Different drive letter on Windows, so it is not servable.
        return None
    if rel.startswith(".."):
        return None
    return "/static/" + rel.replace("\\", "/")


# ---------------------------------------------------------------- limits

MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB, also fed to Flask MAX_CONTENT_LENGTH
MAX_MEDIA_SECONDS = 60                # reject longer clips rather than stall the demo

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"}

FFMPEG_BIN = os.environ.get("NAZARAI_FFMPEG", "ffmpeg")
FFPROBE_BIN = os.environ.get("NAZARAI_FFPROBE", "ffprobe")


# ---------------------------------------------------------------- models

def get_device():
    """
    Pick the inference device once, importing torch lazily.

    Returns (int): 0 when a CUDA GPU is usable, -1 for CPU. The integer form is
    what the transformers pipeline() device argument expects.
    On any import or driver error this returns -1 so the app still runs on CPU.
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


# Image deepfake classifier: the Community Forensics detector (ViT, 87 MB,
# single sigmoid logit = P(generated), 440 px input, standard architecture, no
# remote code).
#
# Chosen by measurement, not by reputation. Benchmarked 2026-08-29 against 10
# known AI-generated faces and 12 real portraits, scoring separation (mean AI
# minus mean real) and AUC (chance a random AI face outranks a random real
# photo; 0.5 is a coin flip). Face-crop input unless noted:
#
#   buildborderless/CommunityForensics-DeepfakeDet-ViT
#                                    full-frame        sep +0.842  AUC 0.958
#                                    face crop         sep +0.692  AUC 0.950
#   Purnachander-Konda/deepfake-detection-swin         sep +0.362  AUC 0.633
#   dima806/deepfake_vs_real_image_detection           sep +0.192  AUC 0.658
#   prithivMLmods/deepfake-detector-model-v1           sep -0.047  AUC 0.433
#
# The last one is anti-correlated (rates real photos as more synthetic than AI
# faces) and must not be used. The Swin flagged half the real set - mostly
# retouched studio portraits - which the CF model scores at 0.03 or less.
# The still-image analyzer scores both the full frame and the face crop and
# blends them rather than picking just the maximum, because genuinely synthetic
# images are high in both views while face-swapped images tend to be subtle in
# the full frame but obvious in the crop.
#
# The single-logit head is also why image_checks loads the model directly
# instead of using the transformers pipeline: softmax over one logit is
# always 1.0, so the pipeline would score everything as certain.
IMAGE_MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
IMAGE_MODEL_CANDIDATES = (
    IMAGE_MODEL_ID,
    "Purnachander-Konda/deepfake-detection-swin",
    "dima806/deepfake_vs_real_image_detection",
)

# --- the detector ensemble -------------------------------------------------
#
# One model is not enough. Each of these was trained on a different slice of
# the generator landscape, and each has blind spots the others cover:
# CommunityForensics is strongest on GAN faces but scored a diffusion-made
# fashion photo at 0.002, a miss that a diffusion-trained detector catches
# outright. Members are scored independently and combined by
# IMAGE_ENSEMBLE_STRATEGY.
#
# All members stay resident on the GPU at once. Their combined weight is about
# 1.6 GB against the 4 GB card, and inference runs one member at a time, so
# peak memory is the resident total plus a single model's activations.
#
# "dir" is a folder under MODELS_DIR; a local copy is preferred over the repo
# id so the app starts offline. A member that fails to load is skipped with a
# warning rather than breaking the signal.
# Measured 2026-08-30 on 13 photorealistic AI images and 19 real photos, which
# is the population that matters here: media passing itself off as a real
# photograph. AUC, and the best recall reachable without flagging any real
# photo:
#
#   solo CommunityForensics   AUC 0.984   92% recall at 0% FP    87 MB
#   CF + SwinV2-Large mean    AUC 0.980   92% recall at 0% FP   832 MB
#   CF + SigLIP mean          AUC 0.988   77% recall at 0% FP   459 MB
#   all four, mean            AUC 0.883   62% recall at 0% FP   1.6 GB
#
# Bigger did not mean better. The 745 MB SwinV2-Large scores AUC 0.800 alone
# against the 87 MB CommunityForensics at 0.944, and adding it to the ensemble
# moves nothing. Organika/sdxl-detector is worse than a coin flip on this set
# (AUC 0.463, it fires on 47% of real photos) and drags any ensemble
# containing it down, which is why "combine everything" scored worst of all.
#
# So the ensemble is deliberately a single member. The machinery below is kept
# because it is measured, tested and config-driven: adding a detector is one
# entry in this list, and each member's own opinion is reported in the signal
# details. The other three are still on disk under models/cand_* if you want
# to re-run the comparison.
IMAGE_ENSEMBLE = [
    {"name": "CommunityForensics", "weight": 1.0,
     "id": "buildborderless/CommunityForensics-DeepfakeDet-ViT",
     "dir": "image_detector",
     # See IMAGE_SCORE_OPERATING_POINT below.
     "operating_point": 0.20},
]

# How members' scores combine when there is more than one. Measured: "max" and
# "noisy_or" look attractive but flag 58% and 63% of real photos respectively,
# because a member that fires spuriously is never outvoted. "mean" is the safe
# default. Irrelevant while the ensemble has one member.
IMAGE_ENSEMBLE_STRATEGY = "mean"

# Raw detector output is not a calibrated probability. This model is decisive:
# real photos sit at a median of 0.0005 and AI images pile up above 0.99, so
# almost nothing lands near the 0.65 verdict line, and the genuinely hard
# cases sit between 0.13 and 0.45 where the default threshold silently drops
# them. Measured on the eval set:
#
#   raw threshold   0.65 -> 77% recall, 0% FP     (the old, uncalibrated line)
#   raw threshold   0.20 -> 92% recall, 0% FP     <- chosen
#   raw threshold   0.13 -> 92% recall, 0% FP     (but see below)
#
# 0.13 gives the same recall and is NOT used: the highest-scoring real photo
# in the set is 0.1290, so a 0.13 line clears it by 0.001 and is fitted to one
# image rather than to a real margin. 0.20 keeps the same recall with 55x that
# margin. A member's raw score is mapped so its operating point lands on
# VERDICT_SYNTHETIC_ABOVE, which keeps one set of verdict thresholds
# meaningful across every signal instead of special-casing this one.
IMAGE_SCORE_OPERATING_POINT = 0.20

# Backup if the primary stops resolving. Swin, labels {0: "Fake", 1: "Real"};
# usable but far weaker (numbers above), kept only so the signal degrades
# instead of disappearing.
IMAGE_MODEL_FALLBACK_ID = "Purnachander-Konda/deepfake-detection-swin"

# Avoid the KoreaPeter ms-eff-gcvit models. They are only 36 MB, but ship a
# custom architecture needing trust_remote_code and have no preprocessor config.

# Audio anti-spoofing. Checked 2026-08-29: Wav2Vec2ForSequenceClassification,
# labels {0: "fake", 1: "real"}. Keep the best model first but try a ranked
# fallback chain so a single repo outage does not kill the signal.
AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
AUDIO_MODEL_CANDIDATES = (
    AUDIO_MODEL_ID,
    "motheecreator/Deepfake-audio-detection",
)
AUDIO_MODEL_FALLBACK_ID = "motheecreator/Deepfake-audio-detection"

# Weights can also be kept in the project instead of the HuggingFace cache.
# Drop config.json, preprocessor_config.json and model.safetensors into
# models/image_detector or models/audio_detector and they are used in place of
# the ids above. Useful on a slow connection, and it makes the app start
# offline once the files are there.
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGE_MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "image_detector")
AUDIO_MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "audio_detector")


def resolve_model(model_id, local_dir):
    """
    Choose between a local weights folder and a HuggingFace repo id.

    model_id (str): the HuggingFace repo id to fall back on.
    local_dir (str): folder that may hold a downloaded copy of the same model.

    Returns (str): local_dir when it contains both a config and a weights file,
    otherwise model_id unchanged. Only a complete folder wins, so a half
    finished download is ignored rather than loaded and crashed on.
    """
    try:
        has_config = os.path.exists(os.path.join(local_dir, "config.json"))
        has_weights = any(
            os.path.exists(os.path.join(local_dir, name))
            for name in ("model.safetensors", "pytorch_model.bin"))
        if has_config and has_weights:
            return local_dir
    except Exception:
        pass
    return model_id

# Label text meaning "synthetic" / "authentic". Compared lowercased so both
# model families above work without per-model special casing.
FAKE_LABELS = {"fake", "ai", "artificial", "spoof", "deepfake", "generated",
               "ai_generated", "label_1"}
REAL_LABELS = {"real", "authentic", "human", "hum", "bonafide", "genuine",
               "label_0"}


# ---------------------------------------------------------------- video

VIDEO_SAMPLE_FPS = 1.0    # one frame per second of source
VIDEO_MAX_FRAMES = 60     # hard cap, keeps a 4 GB card comfortable
VIDEO_BATCH_SIZE = 8      # frames pushed through the model at once

FACE_CROP_MARGIN = 0.25   # grow the detected face box by this fraction per side

# Frames with no detectable face still get scored, but count for less, because a
# full-frame score is much noisier than a face crop.
NO_FACE_FRAME_WEIGHT = 0.4


# ---------------------------------------------------------------- audio

AUDIO_SAMPLE_RATE = 16000       # wav2vec2 expects 16 kHz mono
AUDIO_MAX_SECONDS = 60
# Clips longer than one window are scored in three places (start, middle, end)
# so spliced audio shows up as disagreement rather than averaging away.
AUDIO_WINDOW_SECONDS = 20


# ---------------------------------------------------------------- ELA

ELA_QUALITY = 90       # re-save quality, 90 is the classic choice
ELA_SCALE = 20         # brightness multiplier applied to the difference image


# ---------------------------------------------------------------- explainer

EXPLAIN_TOP_FRAMES = 3       # heatmaps for the N most suspicious video frames
OCCLUSION_PATCH = 48         # grey square size in px for occlusion saliency
OCCLUSION_STRIDE = 24        # step between patch positions


# ---------------------------------------------------------------- fusion

# How much each signal counts in the weighted average. Model-based signals
# dominate. ELA is a weak prior on its own. Metadata and provenance normally
# carry no score and act through the special rules in fusion.py instead.
SIGNAL_WEIGHTS = {
    "face_manipulation": 1.0,
    "video_face_manipulation": 1.0,
    "voice_clone": 1.0,
    "error_level_analysis": 0.45,
    "metadata_forensics": 0.30,
    "provenance": 0.50,
}
DEFAULT_SIGNAL_WEIGHT = 0.5   # used if a new signal forgets to register a weight

VERDICT_AUTHENTIC_BELOW = 0.35   # fused score under this reads as authentic
VERDICT_SYNTHETIC_ABOVE = 0.65   # over this reads as synthetic
MIN_SCOREABLE_SIGNALS = 2        # fewer than this and we refuse to commit
DISAGREEMENT_SPREAD = 0.55       # top signals differing by more than this = inconclusive

# A contradicting signal only forces inconclusive if its influence is at least
# this fraction of the leading signal's. Stops a weak circumstantial check from
# vetoing a confident model result. See the comment in fusion.py.
DISAGREEMENT_MIN_RATIO = 0.5


# ---------------------------------------------------------------- registry

# The pipeline runs exactly these, in order. Each entry is (module, function).
# They are strings, imported on demand, so config.py stays free of heavy imports
# and there is no circular import back from the analyzers into config.
# Adding a check later means writing one function that returns the standard
# signal dict and adding one line here.
# (module, function, signal id, display name). The signal id is carried here
# rather than only inside the returned dict because the pipeline needs to name
# a check before it runs it: to announce it as started, and to label the error
# signal if the analyzer raises before it can name itself. Two entries share a
# module, so the module name alone cannot identify a check.
ANALYZER_REGISTRY = [
    ("image_checks", "analyze_image_model", "face_manipulation",
     "AI Face Manipulation"),
    ("image_checks", "analyze_ela", "error_level_analysis",
     "Error Level Analysis"),
    ("video_checks", "analyze_video_frames", "video_face_manipulation",
     "AI Face Manipulation (Video)"),
    ("audio_checks", "analyze_audio", "voice_clone",
     "AI Voice Cloning"),
    ("forensics", "analyze_metadata", "metadata_forensics",
     "Metadata Forensics"),
    ("provenance", "analyze_provenance", "provenance",
     "Content Credentials"),
]


def load_analyzers():
    """
    Resolve ANALYZER_REGISTRY into callables.

    Returns (list): one dict per usable analyzer, registry order, each
    {"module": str, "signal": str, "display_name": str, "func": callable}.
    An entry whose module or attribute cannot be imported is skipped and
    printed to stdout rather than raised, so one broken file cannot stop the
    app booting.
    """
    import importlib

    resolved = []
    for mod_name, func_name, signal, display_name in ANALYZER_REGISTRY:
        try:
            mod = importlib.import_module(mod_name)
            resolved.append({"module": mod_name, "signal": signal,
                             "display_name": display_name,
                             "func": getattr(mod, func_name)})
        except Exception as exc:
            print(f"[config] analyzer {mod_name}.{func_name} unavailable: {exc}")
    return resolved
