"""
WhatsApp entry point via the Twilio sandbox.

Twilio drops a webhook after about 15 seconds, and scoring a video takes longer
than that, so this module never analyzes inside the request. It replies at once
with a short acknowledgement, runs the pipeline on a background thread, then
pushes the verdict back as a fresh outbound message through the REST API.

Credentials come from environment variables only. Nothing is hardcoded.
See the README for the ngrok and sandbox setup.
"""

import mimetypes
import os
import threading
from xml.sax.saxutils import escape

import media_utils
import pipeline

# Read at call time rather than import time, so a .env loaded later still works.
ENV_SID = "TWILIO_ACCOUNT_SID"
ENV_TOKEN = "TWILIO_AUTH_TOKEN"
ENV_FROM = "TWILIO_WHATSAPP_FROM"        # e.g. whatsapp:+14155238886
ENV_PUBLIC = "PUBLIC_BASE_URL"           # the ngrok https URL, for report links

VERDICT_EMOJI = {
    "likely_synthetic": "⚠️",   # warning sign
    "likely_authentic": "✅",         # check mark
    "inconclusive": "\U0001f9ed",         # compass
}


def _credentials():
    """
    Read the Twilio credentials from the environment.

    Returns (tuple): (account_sid, auth_token, from_number). Any missing value
    comes back as None, and callers degrade to "cannot send" rather than
    raising, so a misconfigured demo still answers the webhook with valid XML.
    """
    return (os.environ.get(ENV_SID), os.environ.get(ENV_TOKEN),
            os.environ.get(ENV_FROM))


def twiml(message):
    """
    Wrap a line of text as a TwiML reply document.

    message (str): plain text to send back on the open webhook.

    Returns (str): TwiML XML. The text is XML-escaped, so a filename containing
    an ampersand cannot produce a malformed document.
    """
    return ("<?xml version='1.0' encoding='UTF-8'?>"
            f"<Response><Message>{escape(message)}</Message></Response>")


def _extension_for(content_type, fallback=".bin"):
    """
    Work out a file extension from a MIME type.

    content_type (str): e.g. "image/jpeg" from Twilio's MediaContentType0.
    fallback (str): used when the type is unknown.

    Returns (str): extension including the dot. The extension matters because
    media_utils.detect_media_type checks it before falling back to ffprobe.
    """
    if not content_type:
        return fallback
    # mimetypes prefers ".jpe" for image/jpeg, which confuses everything downstream.
    overrides = {"image/jpeg": ".jpg", "audio/ogg": ".ogg", "audio/mpeg": ".mp3",
                 "video/mp4": ".mp4", "image/png": ".png", "audio/amr": ".amr"}
    if content_type in overrides:
        return overrides[content_type]
    return mimetypes.guess_extension(content_type.split(";")[0].strip()) or fallback


def download_media(url):
    """
    Fetch a media file from Twilio's CDN.

    url (str): the MediaUrl from the webhook payload.

    Returns (bytes|None): the file content, or None if credentials are missing
    or the download failed. Twilio media URLs need HTTP basic auth with the
    account SID and auth token, which is why this cannot be a plain GET.
    """
    sid, token, _ = _credentials()
    if not sid or not token:
        print("[whatsapp] cannot download media: Twilio credentials not set")
        return None

    try:
        import requests
        response = requests.get(url, auth=(sid, token), timeout=60)
        if response.status_code != 200:
            print(f"[whatsapp] media download returned {response.status_code}")
            return None
        return response.content
    except Exception as exc:
        print(f"[whatsapp] media download failed: {exc}")
        return None


def send_message(to_number, body):
    """
    Send an outbound WhatsApp message through the Twilio REST API.

    to_number (str): recipient in Twilio form, e.g. "whatsapp:+9198...".
    body (str): message text.

    Returns (bool): True if Twilio accepted the message. False when credentials
    are missing or the API call failed; the failure is logged rather than
    raised, because this runs on a background thread with nobody to catch it.
    """
    sid, token, from_number = _credentials()
    if not (sid and token and from_number):
        print("[whatsapp] cannot send: Twilio credentials or sender not set")
        return False

    try:
        from twilio.rest import Client
        Client(sid, token).messages.create(from_=from_number, to=to_number, body=body)
        return True
    except Exception as exc:
        print(f"[whatsapp] send failed: {exc}")
        return False


def format_verdict(result, base_url):
    """
    Turn an analysis into a short WhatsApp message.

    result (dict): the pipeline response.
    base_url (str): public root URL used to make the report link absolute.

    Returns (str): a few short lines - verdict with score, the headline, the
    top two evidence notes, and the report link. Kept tight because this is
    read on a phone.
    """
    if result.get("error"):
        return f"❌ NazarAI could not analyze that file.\n{result['error']}"

    verdict_block = result.get("verdict") or {}
    key = verdict_block.get("verdict", "inconclusive")
    emoji = VERDICT_EMOJI.get(key, "\U0001f9ed")
    label = key.replace("likely_", "").replace("_", " ").upper()

    score = verdict_block.get("overall_score")
    score_text = f" ({int(round(score * 100))}%)" if score is not None else ""

    lines = [f"{emoji} Likely {label}{score_text}", verdict_block.get("headline", "")]

    # Top two pieces of evidence, trimmed so the message stays readable.
    for sig in (verdict_block.get("top_evidence") or [])[:2]:
        note = str(sig.get("human_note", ""))
        if len(note) > 150:
            note = note[:147] + "..."
        lines.append(f"• {note}")

    if result.get("report_url"):
        lines.append(f"\U0001f4c4 Full report: {base_url.rstrip('/')}{result['report_url']}")

    return "\n".join(line for line in lines if line)


def _analyze_and_reply(payload, suffix, original_name, to_number, base_url):
    """
    Background worker: run the pipeline and push the result back.

    payload (bytes): the downloaded media.
    suffix (str): file extension to store it under.
    original_name (str): name to show in the report.
    to_number (str): who to reply to.
    base_url (str): public root URL for the report link.

    Returns: nothing. Every failure is caught and reported to the user as a
    message, because an exception on this thread would otherwise be silent and
    the sender would just never hear back.
    """
    try:
        saved = media_utils.save_bytes(payload, suffix=suffix,
                                       original_name=original_name)
        result = pipeline.analyze_media(saved)
        send_message(to_number, format_verdict(result, base_url))
    except Exception as exc:
        print(f"[whatsapp] analysis thread failed: {exc}")
        send_message(to_number,
                     "❌ NazarAI hit an error analyzing that file. "
                     "Please try another one.")


def handle_webhook(form, url_root):
    """
    Handle one inbound Twilio webhook.

    form (dict-like): Twilio's form-encoded payload, i.e. request.form.
    url_root (str): request.url_root, used as the report link base unless
        PUBLIC_BASE_URL is set (behind ngrok the env var is the reliable one).

    Returns (str): TwiML to return on the open request, always within Twilio's
    timeout. When the message carries media, analysis continues on a daemon
    thread and the verdict arrives later as a separate outbound message.
    A message with no media gets usage instructions instead.
    """
    num_media = 0
    try:
        num_media = int(form.get("NumMedia", 0))
    except (TypeError, ValueError):
        num_media = 0

    sender = form.get("From", "")

    if num_media < 1:
        return twiml("Send NazarAI a photo, video or voice note and it will check "
                     "whether the media shows signs of AI generation or editing.")

    media_url = form.get("MediaUrl0")
    content_type = form.get("MediaContentType0", "")
    if not media_url:
        return twiml("That message had no downloadable media attached.")

    base_url = os.environ.get(ENV_PUBLIC) or url_root

    payload = download_media(media_url)
    if payload is None:
        return twiml("❌ NazarAI could not download that file from WhatsApp. "
                     "Check that the Twilio credentials are configured.")

    suffix = _extension_for(content_type)
    original_name = f"whatsapp_upload{suffix}"

    # Daemon thread: the reply goes out through the REST API when it finishes,
    # long after this webhook response has been sent.
    worker = threading.Thread(
        target=_analyze_and_reply,
        args=(payload, suffix, original_name, sender, base_url),
        daemon=True)
    worker.start()

    return twiml("\U0001f50d NazarAI is analyzing your file, results in about a minute.")
