"""
Explainability: show which parts of the picture drove the model's suspicion.

Uses occlusion saliency. A grey patch is slid across the image and the whole
thing re-scored at every position; wherever hiding a region makes the synthetic
score drop the most, that region is what the model was reacting to. The result
is drawn as a colour overlay at about 40% opacity.

Occlusion was chosen over Grad-CAM deliberately. The active detector is a
SigLIP model, not the plain ViT that pytorch-grad-cam's reshape transform is
written for, and occlusion needs nothing but forward passes, so it works
against any classifier without touching internals. It costs more compute, which
is why it only runs on a still image or the three most suspicious frames.
"""

import os

import config
import image_checks

# Everything is scored at this size. Occlusion cost grows with the square of the
# side length, so the working canvas stays small and the overlay is scaled back
# up to the original at the end.
WORK_SIZE = 224


def _saliency_map(pil_image):
    """
    Build a coarse map of which regions push the synthetic score up.

    pil_image (PIL.Image): RGB image, already resized to WORK_SIZE.

    Returns (numpy.ndarray|None): 2-D float array, one cell per patch position,
    where a higher value means hiding that region lowered the synthetic score
    more, so the region mattered more. Returns None if the classifier is
    unavailable or the baseline could not be scored.
    """
    import numpy as np
    from PIL import Image

    baseline = image_checks.classify_images([pil_image])[0]
    if baseline is None:
        return None

    patch = config.OCCLUSION_PATCH
    stride = config.OCCLUSION_STRIDE
    positions = []
    variants = []

    for top in range(0, WORK_SIZE - patch + 1, stride):
        for left in range(0, WORK_SIZE - patch + 1, stride):
            covered = pil_image.copy()
            # Neutral grey rather than black: black is itself an unusual input
            # and would confuse the score for reasons unrelated to the content.
            covered.paste((128, 128, 128), (left, top, left + patch, top + patch))
            variants.append(covered)
            positions.append((top // stride, left // stride))

    if not variants:
        return None

    # Score in batches so VRAM stays flat regardless of how many positions.
    scores = []
    batch = max(1, config.VIDEO_BATCH_SIZE)
    for start in range(0, len(variants), batch):
        scores.extend(image_checks.classify_images(variants[start:start + batch]))

    rows = max(p[0] for p in positions) + 1
    cols = max(p[1] for p in positions) + 1
    grid = np.zeros((rows, cols), dtype=np.float32)

    for (row, col), score in zip(positions, scores):
        if score is None:
            continue
        # Positive where covering the region reduced the synthetic score.
        grid[row, col] = baseline - score

    return grid


def _render_overlay(source_path, grid, out_path):
    """
    Draw the saliency grid over the original image and save it.

    source_path (str): the image the grid was computed from.
    grid (numpy.ndarray): 2-D saliency values from _saliency_map.
    out_path (str): where to write the PNG.

    Returns (str|None): out_path on success, None if the image could not be
    read or written. The overlay sits at 40% so the underlying face stays
    readable, which is the whole point of showing it to a person.
    """
    import cv2
    import numpy as np

    image = cv2.imread(source_path)
    if image is None:
        return None

    # Clip negatives: regions that made the score go up are not what we are
    # explaining here, and keeping them would wash out the scale.
    grid = np.clip(grid, 0, None)
    span = float(grid.max()) - float(grid.min())
    if span <= 1e-6:
        # Perfectly flat map, so there is nothing meaningful to highlight.
        normalised = np.zeros_like(grid)
    else:
        normalised = (grid - float(grid.min())) / span

    height, width = image.shape[:2]
    heat = cv2.resize((normalised * 255).astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
    return out_path if cv2.imwrite(out_path, blended) else None


def explain_image(image_path, job_id, tag="image"):
    """
    Generate one heatmap overlay for a single image.

    image_path (str): the image to explain.
    job_id (str): analysis id, used in the output filename.
    tag (str): short suffix distinguishing multiple overlays in one job.

    Returns (str|None): the "/static/heatmaps/..." URL, or None if the model was
    unavailable, the saliency map came out flat, or the file could not be
    written. Callers treat None as "no heatmap", never as a failure.
    """
    from PIL import Image

    try:
        original = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    working = original.resize((WORK_SIZE, WORK_SIZE))
    grid = _saliency_map(working)
    if grid is None:
        return None

    out_path = os.path.join(config.HEATMAP_DIR, f"{job_id}_{tag}.png")
    written = _render_overlay(image_path, grid, out_path)
    return config.static_url(written) if written else None


def explain_frames(frame_paths, job_id):
    """
    Generate heatmaps for the most suspicious video frames.

    frame_paths (list): image paths, already ordered most suspicious first.
    job_id (str): analysis id, used in the output filenames.

    Returns (list): "/static/heatmaps/..." URLs, one per frame that produced a
    map. Frames that fail are skipped silently, so a partial set still reaches
    the UI. Never raises, and never runs on more than
    config.EXPLAIN_TOP_FRAMES frames.
    """
    urls = []
    for position, path in enumerate(frame_paths[:config.EXPLAIN_TOP_FRAMES]):
        try:
            url = explain_image(path, job_id, tag=f"frame{position + 1}")
            if url:
                urls.append(url)
        except Exception as exc:
            print(f"[explainer] frame {position} failed: {exc}")
    return urls
