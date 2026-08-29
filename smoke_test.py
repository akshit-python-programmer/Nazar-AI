"""
Run the whole pipeline over everything in samples/ and print the verdicts.

Use this before a demo to confirm the models are cached, the GPU is picked up
and every signal still returns something sane. It talks to pipeline.py directly,
so no server needs to be running.

    python smoke_test.py            # everything in samples/
    python smoke_test.py a.jpg b.mp4  # specific files
"""

import os
import shutil
import sys
import time

import config
import media_utils
import pipeline

SAMPLES_DIR = os.path.join(config.BASE_DIR, "samples")


class LocalFile:
    """
    Minimal stand-in for a Werkzeug FileStorage.

    Lets the smoke test reuse media_utils.save_upload, which expects an object
    with .filename and .save(dest), without starting Flask. This is the one
    class in the project, and only because the upload helper needs that shape.
    """

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)

    def save(self, dest):
        """Copy the sample into the upload directory. dest (str): target path."""
        shutil.copy(self.path, dest)


def collect_samples(args):
    """
    Decide which files to run.

    args (list): command line arguments after the script name.

    Returns (list): absolute paths. Explicit arguments win; otherwise every
    file in samples/ is used. Returns [] if nothing was found, and the caller
    prints a hint rather than crashing.
    """
    if args:
        return [os.path.abspath(a) for a in args if os.path.isfile(a)]
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return [os.path.join(SAMPLES_DIR, name)
            for name in sorted(os.listdir(SAMPLES_DIR))
            if os.path.isfile(os.path.join(SAMPLES_DIR, name))]


def run_one(path):
    """
    Analyze a single file and print a compact summary.

    path (str): file to analyze.

    Returns (dict|None): the pipeline response, or None if the file could not
    even be saved. Exceptions are caught and printed so one bad sample does not
    stop the run.
    """
    print("\n" + "=" * 72)
    print(f"FILE  {os.path.basename(path)}")
    print("=" * 72)

    try:
        saved = media_utils.save_upload(LocalFile(path))
    except Exception as exc:
        print(f"  could not read: {exc}")
        return None

    started = time.time()
    result = pipeline.analyze_media(saved)
    elapsed = time.time() - started

    if result.get("error"):
        print(f"  rejected: {result['error']}")
        return result

    verdict = result.get("verdict") or {}
    score = verdict.get("overall_score")
    print(f"\n  VERDICT  {verdict.get('verdict', '?').upper()}"
          f"   score={score if score is None else round(score, 3)}"
          f"   ({elapsed:.1f}s)")
    print(f"  {verdict.get('headline', '')}")

    print("\n  signals:")
    for sig in result.get("signals", []):
        score_text = ("  -  " if sig.get("synthetic_score") is None
                      else f"{sig['synthetic_score']:.2f}")
        print(f"    {sig['signal']:<26} {sig['status']:<16} "
              f"score={score_text} conf={sig.get('confidence', 0):.2f}")

    if result.get("report_url"):
        print(f"\n  report: {result['report_url']}")
    return result


def main():
    """
    Run every selected sample and print a tally.

    Returns (int): process exit code. 0 when at least one file produced a
    verdict, 1 when there was nothing to test or everything failed, so this can
    be used in a shell check.
    """
    device = "GPU (cuda)" if config.get_device() == 0 else "CPU"
    print(f"NazarAI smoke test on {device}")
    print(f"  image model: {config.IMAGE_MODEL_ID}")
    print(f"  audio model: {config.AUDIO_MODEL_ID}")

    paths = collect_samples(sys.argv[1:])
    if not paths:
        print(f"\nNo files found. Put a few test files in {SAMPLES_DIR} "
              f"or pass paths on the command line.")
        return 1

    verdicts = {}
    for path in paths:
        result = run_one(path)
        if result and result.get("verdict"):
            key = result["verdict"].get("verdict", "unknown")
            verdicts[key] = verdicts.get(key, 0) + 1

    print("\n" + "=" * 72)
    print(f"TALLY over {len(paths)} file(s): "
          + (", ".join(f"{k}={v}" for k, v in verdicts.items()) or "nothing scored"))
    return 0 if verdicts else 1


if __name__ == "__main__":
    sys.exit(main())
