"""Pure speech-recognition text handling: TARS's name, mode names, addressing.
No audio and no I/O. Safe to import and test anywhere.
"""

import re


# A vocabulary hint for faster-whisper. In the October 8 offline probe, tiny.en
# wrote the name as "Tars", "Taurus", "Tarz", and "Tarzan" without it, and as
# "TARS" in every phrase with it, for about 0.05 s more decoding per clip.
STT_HOTWORDS = "TARS, buddy mode, know-it-all, humor, sarcasm, honesty, intellect"

_NAME = re.compile(r"\b(?:tars|tarz)\b", re.IGNORECASE)
_BODY_MODE = re.compile(r"\bbody(?=[\s-]+mode\b)", re.IGNORECASE)
_FILLERS = r"(?:ok|okay|alright|all right|yeah|well|so|now|hey|hi|hello|oh|yo|um|uh)"


def clean_transcript(text):
    """Write the name as TARS and fix the "body mode" mishear of "buddy mode".

    "Your body" and other uses of the word stay untouched: only "body mode",
    which names no TARS feature, is corrected.
    """
    return _BODY_MODE.sub("buddy", _NAME.sub("TARS", text))


def mentions_tars(text):
    return _NAME.search(text) is not None


def _words(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text.lower())).strip()


def is_bare_address(text):
    """True for "Hey TARS." or "TARS?" alone: listen for the question, answer nothing."""
    return re.fullmatch(rf"(?:{_FILLERS} )*tars", _words(clean_transcript(text))) is not None


def is_sleep_command(text):
    """True for a whole-utterance "go to sleep", with an optional address or filler."""
    return re.fullmatch(
        rf"(?:(?:{_FILLERS}|tars|you can|please|just) )*"
        # "up": tiny.en heard "go to sleep" as "go up to sleep" in the October 8 probe.
        r"go (?:back |up )?to sleep(?: (?:now|please|tars|then|for now))*",
        _words(clean_transcript(text)),
    ) is not None
