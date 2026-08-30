"""
Flask entry point for NazarAI.

Routes only: parse the request, hand off to pipeline.py or whatsapp.py, return
JSON. No analysis logic lives here. Run with `python app.py` and open
http://127.0.0.1:5000.
"""

import os

from dotenv import load_dotenv

from flask import Flask, jsonify, render_template, request

load_dotenv()

import config
import media_utils
import pipeline

app = Flask(__name__)

# Flask rejects anything bigger before it reaches a route, which keeps a 2 GB
# upload from filling the disk. The friendly wording comes from the 413 handler.
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES


@app.route("/")
def index():
    """
    Serve the upload page.

    Returns: rendered templates/index.html. The page holds no results markup of
    its own; everything is drawn client side from the /analyze JSON.
    """
    return render_template("index.html",
                           max_mb=config.MAX_UPLOAD_BYTES // (1024 * 1024),
                           max_seconds=config.MAX_MEDIA_SECONDS)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze one uploaded file.

    Expects multipart/form-data with a "file" field.

    Returns: the pipeline response as JSON with HTTP 200, or
    {"error": message} with HTTP 400 when no file was attached or the file
    failed validation (wrong type, too long, empty).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file was attached to the request."}), 400

    try:
        saved = media_utils.save_upload(request.files["file"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = pipeline.analyze_media(saved)
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/analyze/start", methods=["POST"])
def analyze_start():
    """
    Begin an analysis on a background thread and return its id immediately.

    Expects multipart/form-data with a "file" field, same as /analyze.

    Returns: {"job_id": str, "kind": str} with HTTP 202, or {"error": message}
    with HTTP 400 if the file is missing or fails validation.

    This exists so the page can show real progress. /analyze still runs
    synchronously for API clients, the CLI and WhatsApp, which have no use for
    a progress feed and would rather have the result in one call.
    """
    import threading

    import progress

    if "file" not in request.files:
        return jsonify({"error": "No file was attached to the request."}), 400

    try:
        saved = media_utils.save_upload(request.files["file"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = pipeline.new_job_id()
    kind = media_utils.detect_media_type(saved["path"])
    progress.start(job_id, kind)
    progress.phase(job_id, "📥", "Upload received. Fingerprinting the file…", 0.10)

    def work():
        """Run the pipeline, recording any crash against the job."""
        try:
            pipeline.analyze_media(saved, job_id=job_id)
        except Exception as exc:                     # never leave the page hanging
            print(f"[app] job {job_id} crashed: {exc}")
            progress.finish(job_id, error=f"The analysis crashed: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id, "kind": kind}), 202


@app.route("/progress/<job_id>")
def job_progress(job_id):
    """
    Report what a running analysis is doing right now.

    job_id (str): id from /analyze/start.

    Returns: JSON with the stage table, the current phase line, overall
    percentage, video frame counts while frames are being scored, and the
    finished result once done is true. HTTP 404 for an unknown id, which
    includes any job from before a restart.
    """
    import progress

    state = progress.get(job_id)
    if state is None:
        return jsonify({"error": "No analysis is running under that id."}), 404
    return jsonify(state)


@app.route("/job/<job_id>")
def job(job_id):
    """
    Fetch a previously computed result.

    job_id (str): id from a prior /analyze response.

    Returns: the stored response as JSON, or {"error": ...} with HTTP 404 when
    the id is unknown (including after a server restart, since jobs live in
    memory only).
    """
    result = pipeline.get_job(job_id)
    if result is None:
        return jsonify({"error": "No analysis found for that id."}), 404
    return jsonify(result)


@app.route("/whatsapp", methods=["POST"])
def whatsapp_hook():
    """
    Twilio webhook for inbound WhatsApp messages.

    Expects Twilio's form-encoded webhook body. Delegates entirely to
    whatsapp.py, which replies immediately and finishes the analysis on a
    background thread so Twilio does not time out.

    Returns: TwiML XML with content type text/xml. If whatsapp.py is missing or
    raises, a plain TwiML apology is returned so Twilio still gets valid XML.
    """
    try:
        import whatsapp
        body = whatsapp.handle_webhook(request.form, request.url_root)
    except Exception as exc:
        print(f"[app] whatsapp handler failed: {exc}")
        body = ("<?xml version='1.0' encoding='UTF-8'?><Response><Message>"
                "NazarAI could not process that message. Please try again."
                "</Message></Response>")
    return app.response_class(body, mimetype="text/xml")


@app.route("/health")
def health():
    """
    Quick check that the process is alive and what hardware it will use.

    Returns: JSON with the device (cuda or cpu), the configured model ids and
    the number of analyzers that imported successfully. Useful before a demo to
    confirm the GPU was picked up.
    """
    return jsonify({
        "status": "up",
        "device": "cuda" if config.get_device() == 0 else "cpu",
        "image_model": config.IMAGE_MODEL_ID,
        "audio_model": config.AUDIO_MODEL_ID,
        "analyzers_loaded": len(config.load_analyzers()),
    })


@app.errorhandler(413)
def too_large(_exc):
    """
    Turn Flask's raw 413 into the same JSON error shape as everything else.

    Returns: {"error": message} with HTTP 413.
    """
    mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"error": f"That file is over the {mb} MB limit."}), 413


if __name__ == "__main__":
    device = "GPU (cuda)" if config.get_device() == 0 else "CPU"
    print(f"NazarAI starting on {device}")
    print(f"  image model: {config.IMAGE_MODEL_ID}")
    print(f"  audio model: {config.AUDIO_MODEL_ID}")
    # debug=False: the reloader would load every model twice and eat the 4 GB card.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
