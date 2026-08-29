"""
Signal fusion: many independent checks in, one honest verdict out.

Takes the weighted average of every signal that produced a score, then applies
the two provenance rules that can outrank the models. The output is three-way
on purpose. When the signals disagree, or too few of them worked, the answer is
"inconclusive" rather than a confident guess, because a wrong confident answer
is worse than no answer for the people this tool is for.
"""

import config


def _weight_for(signal_name):
    """
    Look up how much a signal counts in the average.

    signal_name (str): the machine name from the signal dict.

    Returns (float): the configured weight, or config.DEFAULT_SIGNAL_WEIGHT if
    the signal is not listed, so a newly added check still participates.
    """
    return config.SIGNAL_WEIGHTS.get(signal_name, config.DEFAULT_SIGNAL_WEIGHT)


def _influence(sig):
    """
    How much one signal actually moved the result.

    sig (dict): a signal dict with a score.

    Returns (float): distance from the undecided midpoint, scaled by confidence
    and configured weight. A signal screaming 0.95 at high confidence outranks
    one mumbling 0.55, which is what "top evidence" should mean.
    """
    score = sig.get("synthetic_score")
    if score is None:
        return 0.0
    return abs(score - 0.5) * sig.get("confidence", 0.0) * _weight_for(sig["signal"])


def _scoreable(signals):
    """
    Filter to the signals that can take part in the average.

    signals (list): all signal dicts.

    Returns (list): those with status "ok", a non-None score and confidence
    above zero. Errors and not-applicable checks are excluded, so a broken
    model never silently drags the number toward 0.5.
    """
    return [s for s in signals
            if s.get("status") == "ok"
            and s.get("synthetic_score") is not None
            and s.get("confidence", 0) > 0]


def _weighted_score(scoreable):
    """
    Confidence- and weight-weighted mean of the signal scores.

    scoreable (list): output of _scoreable, must be non-empty.

    Returns (float|None): the fused score 0..1, or None if the weights summed
    to zero (every signal had zero confidence).
    """
    total = 0.0
    divisor = 0.0
    for sig in scoreable:
        w = sig["confidence"] * _weight_for(sig["signal"])
        total += sig["synthetic_score"] * w
        divisor += w
    if divisor <= 0:
        return None
    return total / divisor


def _find_provenance(signals):
    """
    Pull out the C2PA signal if it ran.

    signals (list): all signal dicts.

    Returns (dict|None): the provenance signal, or None when it is absent or
    errored. Only a status of "ok" counts, because a crashed C2PA reader must
    never be allowed to override the models.
    """
    for sig in signals:
        if sig.get("signal") == "provenance" and sig.get("status") == "ok":
            return sig
    return None


def _headline(verdict, score, signals, media):
    """
    One plain sentence a non-technical person can act on.

    verdict (str): the three-way verdict.
    score (float|None): fused score.
    signals (list): all signals, used to name the strongest piece of evidence.
    media (dict): media context, for saying "video" or "photo" correctly.

    Returns (str): the headline. Short words on purpose, since this line is
    also what gets sent over WhatsApp.
    """
    kind = {"image": "photo", "video": "video", "audio": "audio clip"}.get(
        media.get("type"), "file")
    pct = int(round((score or 0) * 100))

    if verdict == "likely_synthetic":
        return (f"This {kind} shows strong signs of being AI-generated or edited "
                f"({pct}% synthetic across the checks that ran).")
    if verdict == "likely_authentic":
        return (f"This {kind} looks genuine. The checks found no meaningful signs "
                f"of AI generation or editing ({pct}% synthetic).")

    working = len(_scoreable(signals))
    if working < config.MIN_SCOREABLE_SIGNALS:
        return (f"Not enough checks completed on this {kind} to give a reliable "
                f"answer. Treat it as unverified, not as fake.")
    return (f"The checks disagree about this {kind}, so NazarAI will not call it "
            f"either way. Look at the individual signals below before deciding.")


def fuse_signals(signals, media):
    """
    Combine every signal into the final three-way verdict.

    signals (list): all signal dicts from the analyzers, including errored ones.
    media (dict): media context, used only for wording the headline.

    Returns (dict): {"verdict": "likely_synthetic"|"likely_authentic"|
    "inconclusive", "overall_score": float|None, "headline": str,
    "top_evidence": [up to 3 signal dicts, most influential first],
    "reasoning": str explaining which rule decided it,
    "signals_used": int, "signals_failed": int}.
    Never raises: with no usable signals it returns an inconclusive verdict and
    a null score.
    """
    scoreable = _scoreable(signals)
    failed = [s for s in signals if s.get("status") == "error"]

    score = _weighted_score(scoreable) if scoreable else None
    ranked = sorted(scoreable, key=_influence, reverse=True)
    top_evidence = ranked[:3]

    verdict = "inconclusive"
    reasoning = ""

    # ---- normal path: threshold the weighted average
    if score is not None:
        if score >= config.VERDICT_SYNTHETIC_ABOVE:
            verdict = "likely_synthetic"
            reasoning = (f"Fused score {score:.2f} is above the "
                         f"{config.VERDICT_SYNTHETIC_ABOVE} synthetic threshold.")
        elif score <= config.VERDICT_AUTHENTIC_BELOW:
            verdict = "likely_authentic"
            reasoning = (f"Fused score {score:.2f} is below the "
                         f"{config.VERDICT_AUTHENTIC_BELOW} authentic threshold.")
        else:
            reasoning = (f"Fused score {score:.2f} sits in the inconclusive band "
                         f"between {config.VERDICT_AUTHENTIC_BELOW} and "
                         f"{config.VERDICT_SYNTHETIC_ABOVE}.")

    # ---- honesty rule 1: too little evidence to commit
    if len(scoreable) < config.MIN_SCOREABLE_SIGNALS:
        verdict = "inconclusive"
        reasoning = (f"Only {len(scoreable)} scoreable signal(s) succeeded, and at "
                     f"least {config.MIN_SCOREABLE_SIGNALS} are required before "
                     f"NazarAI will commit to a verdict.")

    # ---- honesty rule 2: two comparably strong signals contradict each other
    elif len(ranked) >= 2:
        spread = abs(ranked[0]["synthetic_score"] - ranked[1]["synthetic_score"])
        top_influence = _influence(ranked[0])
        second_influence = _influence(ranked[1])

        # Only a signal of comparable standing gets to veto the leader. Without
        # this, a low-weight circumstantial check like ELA could cancel a
        # confident model result purely by pointing the other way, and every
        # AI-generated image would come back inconclusive: a freshly generated
        # file has a clean single-save compression history, so ELA calls it
        # authentic exactly when the classifier calls it synthetic.
        comparable = (top_influence > 0 and
                      second_influence >= top_influence * config.DISAGREEMENT_MIN_RATIO)

        if spread > config.DISAGREEMENT_SPREAD and comparable:
            verdict = "inconclusive"
            reasoning = (f"The two most influential signals disagree by {spread:.2f} "
                         f"({ranked[0]['display_name']} vs {ranked[1]['display_name']}), "
                         f"which is more than the {config.DISAGREEMENT_SPREAD} allowed.")

    # ---- provenance overrides, applied last so they outrank everything above
    prov = _find_provenance(signals)
    if prov:
        state = (prov.get("details") or {}).get("manifest_status")

        # A cryptographically valid Content Credential is signed by the tool that
        # made the file. If it declares AI generation, that is the maker's own
        # admission and beats any classifier guess. If it is valid and declares a
        # camera capture, that is equally strong the other way.
        if state == "valid":
            if (prov.get("details") or {}).get("declares_ai"):
                verdict = "likely_synthetic"
                score = max(score or 0.0, 0.95)
                reasoning = ("A valid C2PA Content Credential on the file declares it "
                             "as AI-generated. That is the creating tool's own signed "
                             "statement, so it overrides the model signals.")
            else:
                verdict = "likely_authentic"
                score = min(score if score is not None else 0.1, 0.15)
                reasoning = ("A valid C2PA Content Credential traces this file to its "
                             "capture tool with an intact signature, which outweighs "
                             "the statistical signals.")

        # A manifest that fails signature verification means someone altered the
        # file after it was signed. That is a much louder alarm than any model.
        elif state == "invalid_signature":
            verdict = "likely_synthetic"
            score = max(score or 0.0, 0.85)
            reasoning = ("The file carries C2PA Content Credentials whose signature "
                         "does not verify. The content was altered after it was "
                         "signed, so the verdict is forced toward synthetic.")

        if prov not in top_evidence and state in ("valid", "invalid_signature"):
            top_evidence = ([prov] + top_evidence)[:3]

    return {
        "verdict": verdict,
        "overall_score": round(score, 4) if score is not None else None,
        "headline": _headline(verdict, score, signals, media),
        "top_evidence": top_evidence,
        "reasoning": reasoning,
        "signals_used": len(scoreable),
        "signals_failed": len(failed),
    }
