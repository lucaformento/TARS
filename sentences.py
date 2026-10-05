"""Sentence boundaries shared by speech playback and the brain's reply filter.

Both sides must split identically, or a filtered reply could hold text back
longer than playback would.
"""

import re


def split_sentences(buffer, final=False):
    """Extract speech chunks; wait for whitespace to resolve split-token decimals.
    Common abbreviations are conservative: they may keep two sentences together.
    """
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
    chunks, start = [], 0
    for match in re.finditer(r"""[.!?]+["'\u201d\u2019)\]]*(?=\s)""", buffer):
        if match.group().startswith("."):
            prefix = buffer[:match.start() + 1]
            word = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", prefix)
            if word and (word[1].lower() in abbreviations or len(word[1]) == 1):
                continue
            if re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix):
                continue
        chunk = buffer[start:match.end()].strip()
        if chunk:
            chunks.append(chunk)
        start = match.end()
    remainder = buffer[start:].lstrip()
    if final and remainder.strip():
        chunks.append(remainder.strip())
        remainder = ""
    return chunks, remainder
