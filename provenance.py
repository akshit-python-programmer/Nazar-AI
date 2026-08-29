"""
Provenance: read C2PA Content Credentials if the file carries any.

Content Credentials are a signed record of how a file was made. When one is
present and its signature verifies, it is far stronger evidence than any
classifier guess, in either direction, which is why fusion.py lets this signal
override the models. Most files in the wild have no manifest at all, and that
is reported as a neutral fact, never as evidence of fakery.
"""

import json

import media_utils

DISPLAY = "Content Credentials (C2PA)"

# IPTC digital source types that appear inside a manifest. The first two are the
# standard way a tool declares "a model made this".
AI_SOURCE_TYPES = [
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
]
CAPTURE_SOURCE_TYPES = ["digitalcapture", "negativefilm", "positivefilm", "print"]

# Signing tools that only ever sign generated content.
AI_TOOL_HINTS = ["firefly", "dall-e", "dalle", "openai", "midjourney",
                 "stable diffusion", "imagen", "veo", "sora", "gemini", "flux"]


def _read_with_c2pa(path):
    """
    Read a manifest using whichever c2pa-python API version is installed.

    path (str): media file to inspect.

    Returns (tuple): (raw_manifest_json_string, error_string). Exactly one is
    non-None. The API changed shape between c2pa-python releases, so both the
    modern Reader class and the older read_file function are attempted before
    giving up.
    """
    try:
        import c2pa
    except ImportError:
        return None, "c2pa package is not installed"

    # Modern API (c2pa-python 0.6+): Reader as a context manager.
    reader_cls = getattr(c2pa, "Reader", None)
    if reader_cls is not None:
        try:
            with reader_cls(path) as reader:
                return reader.json(), None
        except Exception as exc:
            # A file with no manifest raises here rather than returning empty.
            return None, str(exc)

    # Older API: module-level read_file returning a JSON string.
    read_file = getattr(c2pa, "read_file", None)
    if read_file is not None:
        try:
            return read_file(path, None), None
        except Exception as exc:
            return None, str(exc)

    return None, "installed c2pa package exposes no known read API"


def _looks_like_no_manifest(error_text):
    """
    Decide whether a c2pa error just means "this file has no credentials".

    error_text (str): the exception text from the reader.

    Returns (bool): True when the error is the ordinary no-manifest case, which
    must be reported neutrally, not as a failure of the check.
    """
    lowered = (error_text or "").lower()
    # c2pa 0.37 raises "ManifestNotFound: no JUMBF data found" for a clean file,
    # so "jumbf" on its own is the reliable marker rather than any exact phrase.
    return any(marker in lowered for marker in
               ("no claim", "manifestnotfound", "manifest not found", "no manifest",
                "jumbf", "not found", "no c2pa"))


def _inspect_manifest(raw_json):
    """
    Work out what a manifest actually says.

    raw_json (str): the JSON string returned by the c2pa reader.

    Returns (dict): {"declares_ai": bool, "issuer": str, "tool": str,
    "source_type": str, "validation_errors": [str], "active": dict}.
    Parsing is defensive because manifest layout varies by signing tool; any
    field that cannot be found comes back as "" rather than raising.
    """
    out = {"declares_ai": False, "issuer": "", "tool": "",
           "source_type": "", "validation_errors": [], "active": {}}

    try:
        data = json.loads(raw_json) if isinstance(raw_json, str) else (raw_json or {})
    except Exception:
        return out

    manifests = data.get("manifests") or {}
    active_id = data.get("active_manifest")
    active = manifests.get(active_id) or (next(iter(manifests.values()), {}) if manifests else {})
    out["active"] = {"label": active.get("label", ""),
                     "title": active.get("title", ""),
                     "format": active.get("format", "")}

    sig = active.get("signature_info") or {}
    out["issuer"] = sig.get("issuer", "") or sig.get("common_name", "")

    claim_gen = active.get("claim_generator_info") or []
    if isinstance(claim_gen, list) and claim_gen:
        out["tool"] = str(claim_gen[0].get("name", ""))
    if not out["tool"]:
        out["tool"] = str(active.get("claim_generator", ""))

    # Validation problems are reported alongside the manifest, not as an error.
    for key in ("validation_status", "validation_results"):
        for item in (data.get(key) or []):
            if isinstance(item, dict) and item.get("code"):
                out["validation_errors"].append(str(item["code"]))

    # Walk the assertions looking for how the content was made.
    blob = json.dumps(active.get("assertions") or []).lower()
    for marker in AI_SOURCE_TYPES:
        if marker in blob:
            out["declares_ai"] = True
            out["source_type"] = marker
            break
    if not out["source_type"]:
        for marker in CAPTURE_SOURCE_TYPES:
            if marker in blob:
                out["source_type"] = marker
                break

    if not out["declares_ai"]:
        tool_blob = (out["tool"] + " " + out["issuer"]).lower()
        if any(hint in tool_blob for hint in AI_TOOL_HINTS):
            out["declares_ai"] = True
            out["source_type"] = out["source_type"] or "ai signing tool"

    return out


def analyze_provenance(media):
    """
    Signal: check the file for C2PA Content Credentials.

    media (dict): media context; works on any file type c2pa can open.

    Returns (dict): standard signal dict, signal "provenance", where
    details["manifest_status"] is one of:
      "valid"              a manifest is present and its signature verified
      "invalid_signature"  a manifest is present but does not verify
      "none"               no credentials, the ordinary case
      "unavailable"        the c2pa library is missing or errored
    details["declares_ai"] tells fusion which way a valid manifest points.
    Those two fields are the contract fusion.py reads, so keep them stable.
    A missing manifest is status "ok" with no score, never an error.
    """
    raw, error = _read_with_c2pa(media["path"])

    # ---- library missing or broken
    if raw is None and error and not _looks_like_no_manifest(error):
        if "not installed" in error:
            return media_utils.make_signal(
                "provenance", DISPLAY,
                status="error", confidence=0.0,
                human_note="Content Credentials could not be checked because the "
                           "C2PA reader is not installed on this machine.",
                details={"manifest_status": "unavailable", "declares_ai": False,
                         "error": error})
        return media_utils.make_signal(
            "provenance", DISPLAY,
            status="error", confidence=0.0,
            human_note="The Content Credentials reader failed on this file.",
            details={"manifest_status": "unavailable", "declares_ai": False,
                     "error": error})

    # ---- no manifest: neutral, and the common case
    if raw is None:
        return media_utils.make_signal(
            "provenance", DISPLAY,
            synthetic_score=None, confidence=0.0,
            human_note="No provenance credentials found. Most files do not carry "
                       "them, so this is not a sign of fakery. Verification relies "
                       "on the forensic signals instead.",
            details={"manifest_status": "none", "declares_ai": False})

    info = _inspect_manifest(raw)

    # ---- manifest present but signature does not verify
    if info["validation_errors"]:
        return media_utils.make_signal(
            "provenance", DISPLAY,
            synthetic_score=0.85, confidence=0.9,
            human_note=("This file carries Content Credentials, but the signature "
                        "does not verify: " + ", ".join(info["validation_errors"][:3]) +
                        ". That means the file was changed after it was signed."),
            details={"manifest_status": "invalid_signature",
                     "declares_ai": info["declares_ai"],
                     "issuer": info["issuer"], "tool": info["tool"],
                     "validation_errors": info["validation_errors"]})

    # ---- valid manifest declaring AI generation
    if info["declares_ai"]:
        return media_utils.make_signal(
            "provenance", DISPLAY,
            synthetic_score=0.97, confidence=0.95,
            human_note=(f"The file carries valid Content Credentials stating it was "
                        f"generated by AI"
                        + (f" using {info['tool']}" if info["tool"] else "") +
                        ". This is the creating tool's own signed record, so it is "
                        "about as certain as this gets."),
            details={"manifest_status": "valid", "declares_ai": True,
                     "issuer": info["issuer"], "tool": info["tool"],
                     "source_type": info["source_type"]})

    # ---- valid manifest describing an ordinary capture or edit
    return media_utils.make_signal(
        "provenance", DISPLAY,
        synthetic_score=0.08, confidence=0.85,
        human_note=(f"The file carries valid Content Credentials"
                    + (f" signed by {info['issuer']}" if info["issuer"] else "") +
                    " with no AI generation declared. The signature is intact, so "
                    "its recorded history can be trusted."),
        details={"manifest_status": "valid", "declares_ai": False,
                 "issuer": info["issuer"], "tool": info["tool"],
                 "source_type": info["source_type"]})
