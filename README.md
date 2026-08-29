# NazarAI

Multi-signal media verification. Upload an image, video or audio clip and
NazarAI runs several independent forensic checks over it, fuses them into a
three-way verdict, and produces a downloadable PDF evidence report.

It is deliberately **not** a binary real/fake classifier. Each check is one
signal. No single model failure changes the answer on its own, and when the
signals disagree or too few of them worked, the verdict is **Inconclusive**
rather than a confident guess. For someone filing a complaint about a
manipulated video of themselves, a wrong confident answer is worse than no
answer.

## What it checks

| Signal | What it looks at |
|---|---|
| AI Face Manipulation | Transformer image classifier, run on the photo or on each video frame |
| Error Level Analysis | JPEG compression unevenness, the classic locally-edited-region test |
| AI Face Manipulation (Video) | Frames sampled at 1 fps, cropped to the detected face, plus a suspicion timeline |
| AI Voice Cloning | wav2vec2 anti-spoofing model over the audio track |
| Metadata Forensics | EXIF and container tags: editor software, AI generator names, timestamp mismatches |
| Content Credentials (C2PA) | Signed provenance manifests, when the file carries any |

A video automatically gets both the visual and the voice check, because its
audio track is extracted and analysed too.

## Requirements

- Python 3.10+ (built and tested on 3.13)
- **ffmpeg and ffprobe on PATH** — system binaries, not pip packages.
  Windows: `winget install Gyan.FFmpeg`, or download a build and add its `bin`
  folder to PATH. Verify with `ffmpeg -version`.
- An NVIDIA GPU is optional. Everything falls back to CPU on its own, just
  slower.

## Setup

Install PyTorch first, because the CUDA build comes from PyTorch's own index
rather than PyPI:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
```

CPU-only machines can use plain `pip install torch torchaudio torchvision`.

Then the rest:

```bash
pip install -r requirements.txt
```

Run it:

```bash
python app.py
```

Open http://127.0.0.1:5000.

The first analysis downloads the model weights (about 110 MB for the image model
and 378 MB for the audio one) and will be slow. Everything after that is cached,
and weights already in `models/` are used instead of downloading. `GET /health`
reports whether the GPU was picked up and how many analyzers loaded.

## Checking it works

```bash
python smoke_test.py
```

Runs the whole pipeline over everything in `samples/` and prints a verdict per
file. Pass paths to test specific files instead.

## Limits

Uploads are capped at 25 MB and 60 seconds. Longer or larger files are rejected
with a message rather than silently truncated. Videos are sampled at 1 frame per
second up to 60 frames, which is what keeps a 4 GB card from running out of
memory.

## WhatsApp (Twilio sandbox)

Twilio drops a webhook after roughly 15 seconds and video analysis takes longer,
so `/whatsapp` replies immediately, runs the pipeline on a background thread,
and pushes the verdict back as a separate outbound message when it finishes.

Credentials are read from the environment. Nothing is hardcoded.

```bash
export TWILIO_ACCOUNT_SID=ACxxxxxxxx
export TWILIO_AUTH_TOKEN=xxxxxxxx
export TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
export PUBLIC_BASE_URL=https://your-id.ngrok-free.app
```

On Windows PowerShell use `$env:TWILIO_ACCOUNT_SID = "ACxxxxxxxx"` instead. A
`.env` file in the project root also works.

Setup steps:

1. Start the app: `python app.py`
2. Expose it: `ngrok http 5000`
3. Copy the https URL ngrok prints, and set `PUBLIC_BASE_URL` to it so report
   links in WhatsApp messages resolve from a phone.
4. In the Twilio console, go to Messaging → Try it out → Send a WhatsApp
   message → Sandbox settings, and set **"When a message comes in"** to
   `https://your-id.ngrok-free.app/whatsapp`, method POST.
5. Join the sandbox from your phone with the `join <code>` message Twilio shows,
   then send it a photo or voice note.

`PUBLIC_BASE_URL` matters because the report link has to be reachable from the
phone, and the request's own host header is not that URL behind ngrok.

## Architecture

```
app.py           Flask routes only, no logic
pipeline.py      orchestration: validate, run analyzers, fuse, report
config.py        every constant, threshold, model id, and the analyzer registry
media_utils.py   saving, hashing, type detection, frame and audio extraction
image_checks.py  image classifier + Error Level Analysis
video_checks.py  frame sampling, face cropping, per-frame scoring, timeline
audio_checks.py  voice clone / anti-spoofing inference
forensics.py     EXIF and container metadata anomalies
provenance.py    C2PA content credentials
explainer.py     occlusion saliency heatmaps
fusion.py        signal fusion and the three-way verdict
report.py        PDF evidence report
whatsapp.py      Twilio webhook and async reply
```

### The analyzer contract

Every analyzer takes the media context and returns one dict shape:

```python
{
  "signal": "voice_clone",
  "display_name": "AI Voice Cloning",
  "status": "ok",              # ok | error | not_applicable
  "synthetic_score": 0.87,     # 0 authentic .. 1 synthetic, or None
  "confidence": 0.7,           # how much to trust this signal for this file
  "human_note": "...",         # plain language, this is what people read
  "details": {...}             # timelines, heatmap paths, raw metadata
}
```

Fusion, the web UI, the PDF and WhatsApp all consume only that shape. Adding a
new check means writing one function that returns this dict and adding one line
to `ANALYZER_REGISTRY` in `config.py`.

Analyzers are wrapped so an exception becomes `status: "error"` and the rest of
the pipeline continues. A broken signal is reported as broken, never as
evidence in either direction.

### How the verdict is reached

Signals that produced a score are averaged, weighted by their own confidence
and their configured weight in `config.py`. The result is thresholded at 0.35
and 0.65, with the band between them being inconclusive.

Two rules override that:

- A **valid C2PA manifest** is the creating tool's own signed statement, so it
  outranks the statistical signals in either direction.
- A manifest whose **signature does not verify** means the file changed after
  signing, and forces the verdict toward synthetic.

Two rules force inconclusive regardless of the average:

- Fewer than 2 scoreable signals succeeded.
- The two most influential signals disagree by more than 0.55 **and** are of
  comparable influence. The second condition matters: without it a low-weight
  circumstantial check like ELA cancels a confident classifier result just by
  pointing the other way, and every AI-generated image comes back inconclusive,
  because a freshly generated file has a clean single-save compression history
  and so reads as authentic to ELA exactly when the classifier reads it as
  synthetic. Measured on the eval set (with the Swin model then active),
  adding the influence test moved AI-face detection from 30% to 90% with no AI
  image wrongly cleared.

### API

`POST /analyze` with multipart `file` returns:

```json
{
  "file": {"name": "...", "type": "video", "sha256": "...", "analyzed_at": "..."},
  "verdict": {"verdict": "likely_synthetic", "overall_score": 0.82,
              "headline": "...", "top_evidence": [...]},
  "signals": [ ... ],
  "media": {"heatmaps": ["/static/..."], "ela_image": "/static/...", "timeline": [...]},
  "report_url": "/static/reports/....pdf"
}
```

`GET /job/<id>` returns a stored result. `GET /health` reports device and model
config. Job state is in memory only, so it does not survive a restart.

## Notes on the models

Model ids live in `config.py` with the label mapping each one uses. Both the
image and audio checks have a fallback id and will try it if the primary fails
to load.

### The image model was chosen by measurement

Four candidates were benchmarked on 10 known AI-generated faces and 12 real
portraits. AUC is the chance a random AI face outranks a random real photo, so
0.5 is a coin flip:

| Model | Separation | AUC | Size |
|---|---|---|---|
| `buildborderless/CommunityForensics-DeepfakeDet-ViT` (full frame) | **+0.842** | **0.958** | 87 MB |
| `buildborderless/CommunityForensics-DeepfakeDet-ViT` (face crop) | +0.692 | 0.950 | 87 MB |
| `Purnachander-Konda/deepfake-detection-swin` | +0.362 | 0.633 | 110 MB |
| `dima806/deepfake_vs_real_image_detection` | +0.192 | 0.658 | 343 MB |
| `prithivMLmods/deepfake-detector-model-v1` | -0.047 | 0.433 | 372 MB |

The winner is the detector from the Community Forensics work, trained across
thousands of generators. It is also the smallest. The last model is
**anti-correlated** - it rates real photos as more synthetic than AI faces -
despite being the obvious pick by download count. Popularity is not accuracy;
measure before trusting any of these.

Two implementation details matter for this model. It has a **single sigmoid
logit** (P(generated)), which the transformers pipeline mishandles (softmax
over one logit is always 1.0), so `image_checks.py` loads and scores it
directly. And the still-image analyzer scores **both the full frame and the
face crop, keeping the higher**: full-frame is how the detector was trained
and is strongest on wholly generated images, while the crop is what exposes a
face swapped into an otherwise real photo.

Re-run the benchmark yourself after adding samples to `samples/eval_ai` and
`samples/eval_real`.

### Known accuracy, measured end to end

On that same set, through the whole pipeline:

- Real photos: **14 of 14 correctly cleared, zero false positives** -
  including heavily retouched studio portraits that fooled the weaker models.
- AI-generated faces: **80% flagged** as likely synthetic, 10% inconclusive,
  one wrongly cleared (a diffusion-generated fashion shot the model scores
  0.002 - a genuine blind spot, documented rather than tuned around).

A low score is still "no evidence found", not proof of authenticity: a
generator this model has not seen can score low. That is why the verdict is
three-way and why no single signal decides it. The audio model is equally
strong on its own ground: three genuine human speech recordings scored 0.000
synthetic across every window, against 1.000 for a synthetic control.

The explainer uses **occlusion saliency** rather than Grad-CAM: a grey patch is
slid across the image and the whole thing re-scored at each position, so
wherever hiding a region drops the synthetic score most is what the model was
reacting to. It needs nothing but forward passes, which means it works against
any classifier architecture. Grad-CAM's ViT reshape transform does not apply
cleanly to the SigLIP model in use here.

## Interpreting results honestly

- "No provenance credentials" is the normal case for almost every file on the
  internet. It is not evidence of fakery, and the tool never treats it as such.
- Missing EXIF usually means a platform stripped it, not that someone faked the
  photo. It only means something if the file is claimed to be an original.
- Error Level Analysis is weak evidence on its own and near-meaningless on a
  PNG or a screenshot. It is weighted accordingly.
- These classifiers are trained on particular generators. A newer generator can
  score low. A low score is "no evidence found", not proof of authenticity.

The PDF footer says the same thing: automated multi-signal analysis for triage
and complaint filing, not a certified forensic conclusion.

## Model files on disk

Weights live in `models/`:

```
models/image_detector/        CommunityForensics ViT, the active classifier (87 MB)
models/audio_detector/        wav2vec2 anti-spoofing (378 MB)
models/image_detector_swin_alt/   Swin alternative, kept for comparison (110 MB)
models/image_detector_vit_alt/    ViT alternative, kept for comparison (343 MB)
models/rejected_siglip_unusable/  benchmarked at AUC 0.433, safe to delete (355 MB)
```

`config.resolve_model()` prefers a local folder over the HuggingFace repo id, so
once these exist the app starts offline and never re-downloads. Delete the
`_alt` and `rejected_` folders to reclaim ~800 MB.

Evaluation samples used for the benchmark are in `samples/eval_ai/` (known
AI-generated faces), `samples/eval_real/` (real portraits) and
`samples/eval_audio/` (genuine human speech recordings).

Note: `app.py` runs with `debug=False` so the reloader does not load every model
twice on a 4 GB card. Template and code edits therefore need a server restart.
