"""Persistent memory for TARS: what he knows about Luca between restarts.

Storage is a single human-readable JSON file so Luca can open, audit, and
hand-edit it. Every entry records where it came from, so a guess can never
quietly become a fact.

Parsing and inference are pure functions with no I/O, matching personality.py,
so they can be tested without touching the filesystem.
"""

from functools import wraps
import json
import os
import re
import tempfile
import threading
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().with_name("memory.json")

# Entry shape: id, text, kind, source, created, updated
KINDS = ("fact", "preference", "bit")
SOURCES = ("stated", "inferred")

# How much memory reaches the model. Confirmed entries win ties; the character
# cap is the real guard because a long entry costs more than a short one.
MAX_ENTRIES_IN_PROMPT = 20
MAX_BITS_IN_PROMPT = 2
MAX_PROMPT_CHARS = 1200
MAX_ENTRY_CHARS = 240
MAX_TOTAL_ENTRIES = 500


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# Luca, October 10: a check-in asks about older guesses only. In the live test
# TARS asked about something said two minutes earlier, which felt like an echo.
CHECKIN_MIN_AGE = timedelta(days=1)


def _created_before(entry, cutoff):
    try:
        created = datetime.fromisoformat(entry["created"])
    except (KeyError, TypeError, ValueError):
        return False
    if created.tzinfo is None:
        return False
    return created <= cutoff


def _normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().rstrip(" .!?")).strip()


_ADDRESS_PREFIX = re.compile(
    r"^(?:hey\s*[,;:]?\s*)?(?:tars|tarts)\s*[,;:!-]?\s+", re.IGNORECASE
)


def _strip_address(text):
    """Remove one transcribed wake/name prefix before local intent parsing."""
    return _ADDRESS_PREFIX.sub("", _normalize(text), count=1)


class AmbiguousMemory(ValueError):
    """A deletion request matches several entries; nothing has been deleted."""


# Only simple preference phrasing is canonicalized. The entire object,
# including comparisons and qualifiers, must match. Bits never use this key.
_PREFERENCE = re.compile(
    r"^(?:i\s+)?(?:really\s+)?(?P<verb>like|likes|love|loves|prefer|prefers|"
    r"enjoy|enjoys|hate|hates|dislike|dislikes|can't stand)\s+(?P<object>.+)$", re.I
)


def _identity(text, kind):
    normalized = _normalize(text).casefold()
    if kind != "bit":
        match = _PREFERENCE.fullmatch(normalized)
        if match:
            negative = match["verb"] in {"hate", "hates", "dislike", "dislikes", "can't stand"}
            return ("preference", "negative" if negative else "positive", match["object"])
    return (kind, normalized)


def _tokens(text):
    return re.findall(r"[^\W_]+", text.casefold())


def _contains_tokens(haystack, needle):
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def _timestamp(value, fallback):
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
        except ValueError:
            pass
    return fallback


# --------------------------------------------------------------------------
# Pure command parsing
# --------------------------------------------------------------------------

# Only leading, delimited words are fillers; never search inside a report,
# quotation or conditional, and never strip these words from the payload.
_LEAD_WORDS = r"can you|could you|please|hey|ok|okay|yeah|yes|just|so|um|actually"
_LEAD = rf"(?:(?:{_LEAD_WORDS})(?:\s*[,;:]\s*|\s+))*"
_LEADING_WORDS = re.compile(rf"^{_LEAD}", re.IGNORECASE)
_MEMORY_WORD = re.compile(r"\b(?:remember|forget)\b", re.IGNORECASE)

# "remember that I prefer PETG" / "remember I work nights"
_REMEMBER_WITH_OBJECT = re.compile(
    rf"^{_LEAD}remember\s+(?:that\s+|this[:,]\s+|the fact that\s+)?(?P<text>.+)$",
    re.IGNORECASE,
)
# Bare "remember that" / "remember this" -> keep what TARS just said
_REMEMBER_BARE = re.compile(
    rf"^{_LEAD}remember\s+(?:that|this|it|that one)$", re.IGNORECASE
)
_REMEMBER_REQUEST = re.compile(rf"^{_LEAD}remember\b", re.IGNORECASE)
_REMEMBER_REMINISCENCE = re.compile(
    rf"^{_LEAD}remember\s+(?:(?:when|where|who|what|why|how|which)\b|"
    rf"(?:the|that)\s+time\b)",
    re.IGNORECASE,
)
_FORGET = re.compile(
    rf"^{_LEAD}forget\s+(?:that\s+|about\s+|the\s+)*(?P<text>.+)$", re.IGNORECASE
)
_FORGET_BARE = re.compile(rf"^{_LEAD}forget\s+(?:that|this)$", re.IGNORECASE)
_FORGET_IT = re.compile(
    rf"^(?:(?:{_LEAD_WORDS}|never mind)(?:\s*[,;:]\s*|\s+))*forget\s+it$",
    re.IGNORECASE,
)
# "Them" is an ambiguous transcription, not an alias for "that" or a bulk
# deletion command. "Forget that <exact text>" can still target any wording.
_FORGET_UNSUPPORTED = re.compile(rf"^{_LEAD}forget\s+(?:it|them)\b", re.IGNORECASE)
_RECALL = re.compile(
    rf"^{_LEAD}(?:what do you remember(?:\s+about me)?|what do you know about me|"
    rf"list your memories|what have you remembered)$",
    re.IGNORECASE,
)


def parse_memory_command(text):
    """Return (intent, payload) for an explicit memory command, else None.

    Intents: 'remember' (payload=text), 'remember_last' (payload=None),
    'remember_rejected' (payload=None), 'forget' (payload=text or None),
    'recall' (payload=None).
    Pure: no I/O, no state.
    """
    command = _strip_address(text)
    # A transcribed name may follow the fillers: "yeah, TARS, just forget
    # that". Remove one leading address, never a name inside the memory text.
    command = _strip_address(_LEADING_WORDS.sub("", command, count=1))
    if not command:
        return None
    if _RECALL.match(command):
        return ("recall", None)
    if _FORGET_IT.fullmatch(command) or _FORGET_UNSUPPORTED.match(command):
        return None
    if (_REMEMBER_REQUEST.match(command)
            and ((text or "").rstrip().endswith("?")
                 or _REMEMBER_REMINISCENCE.match(command))):
        return ("remember_rejected", None)
    if _REMEMBER_BARE.match(command):
        return ("remember_last", None)
    if _FORGET_BARE.match(command):
        return ("forget", None)
    match = _REMEMBER_WITH_OBJECT.match(command)
    if match:
        return ("remember", _normalize(match.group("text")))
    match = _FORGET.match(command)
    if match:
        return ("forget", _normalize(match.group("text")))
    return None


def is_memory_request(text):
    """Classify parsed commands and guarded attempts, including no-ops.

    This does not authorize a mutation. The note-taker integration must use
    this broader classification to skip quiet notes and invalidate pending
    jobs even when parse_memory_command returns None.
    """
    return bool(_MEMORY_WORD.search(text or "") or parse_memory_command(text))


# --------------------------------------------------------------------------
# Pure inference
# --------------------------------------------------------------------------

# Deliberately narrow. These fire on first-person statements of durable fact
# only, and everything they produce is tagged 'inferred' so it can never be
# mistaken for something Luca asked TARS to keep.
_INFERENCE_PATTERNS = (
    (re.compile(r"\bi (?:really )?(?:like|love|prefer|enjoy) (?P<v>[^,.;!?]{3,80})", re.I),
     "preference", "likes {v}"),
    (re.compile(r"\bi (?:really )?(?:hate|dislike|can't stand) (?P<v>[^,.;!?]{3,80})", re.I),
     "preference", "dislikes {v}"),
    (re.compile(r"\bi(?:'m| am) (?:currently )?(?:working on|building) (?P<v>[^,.;!?]{3,80})", re.I),
     "fact", "is working on {v}"),
    (re.compile(r"\bi (?:use|run|drive) (?P<v>[^,.;!?]{3,80})", re.I),
     "fact", "uses {v}"),
    (re.compile(r"\bmy (?P<k>[a-z ]{2,24}?) is (?P<v>[^,.;!?]{2,80})", re.I),
     "fact", "{k} is {v}"),
)

# Never infer from a sentence that is asking rather than telling.
_QUESTIONISH = re.compile(r"^(?:what|who|when|where|why|how|do|does|did|can|could|"
                          r"should|would|is|are|was|were|will)\b", re.I)

# Spoken preferences often arrive with harmless first-person framing. Strip a
# small allowlist before matching instead of searching anywhere in a sentence;
# searching would also capture quotes and reports such as "my friend said I
# like PETG".
_INFERENCE_LEAD = re.compile(
    r"^(?:(?:i\s+)?(?:just\s+)?(?:want(?:ed)?\s+to\s+)?"
    r"(?:let you know|tell you)(?:\s+that)?|(?:just\s+)?so you know|"
    r"for (?:the )?record|by the way)[,;:]?\s+",
    re.I,
)


def infer_memories(text):
    """Return candidate inferred entries from one of Luca's turns. Pure.

    Conservative by design: a missed memory costs nothing, a wrong one that
    looks confirmed costs trust.
    """
    found = []
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        if "?" in sentence:
            continue
        sentence = _strip_address(sentence)
        if not sentence or _QUESTIONISH.match(sentence):
            continue
        if parse_memory_command(sentence):
            continue  # explicit commands are handled elsewhere
        sentence = _INFERENCE_LEAD.sub("", sentence, count=1)
        for pattern, kind, template in _INFERENCE_PATTERNS:
            match = pattern.match(sentence)
            if not match:
                continue
            values = {k: _normalize(v) for k, v in match.groupdict().items() if v}
            if not all(values.values()):
                continue
            phrase = template.format(**values)[:MAX_ENTRY_CHARS]
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"text": phrase, "kind": kind})
            break  # one inference per sentence keeps this honest
    return found


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

def entry_revision(entry):
    """A check-in may only change the exact entry version it asked about."""
    return f"{entry['updated']}|{entry['text']}"


def _locked(method):
    """Hold the store lock for the whole call; never hand out live entries."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            result = method(self, *args, **kwargs)
        return dict(result) if isinstance(result, dict) else result
    return wrapper


class MemoryStore:
    """Durable, auditable memory. One JSON file, atomic writes.

    The background note-taker writes while the next turn may read or write, so
    every public method holds a reentrant lock across its whole read, modify,
    and persist sequence. `epoch` counts explicit memory commands; a note job
    from an older epoch is dropped instead of applied.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.lock = threading.RLock()
        self.epoch = 0
        self.entries = []
        self.load_error = None
        # Process-local by design: these IDs represent memories added or
        # upgraded since this MemoryStore was created (TARS startup).
        self._saved_this_run = set()
        self.load()

    # ---- persistence ----

    @_locked
    def load(self):
        self.entries = []
        self.load_error = None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, UnicodeError):
            self._quarantine()
            return
        except OSError:
            # Unreadable is not the same as corrupt. Never replace a file
            # whose contents we could not inspect.
            self.load_error = "Memory file could not be read."
            warnings.warn(self.load_error, RuntimeWarning)
            return
        if not isinstance(raw, list) or not all(self._valid(e) for e in raw):
            self._quarantine()
            return
        by_identity = {}
        for original in raw:
            entry = self._repair_metadata(original)
            key = _identity(entry["text"], entry["kind"])
            previous = by_identity.get(key)
            if previous is None:
                by_identity[key] = entry
            elif previous["source"] == "inferred" and entry["source"] == "stated":
                entry["id"] = previous["id"]
                entry["created"] = min(previous["created"], entry["created"])
                by_identity[key] = entry
        self.entries = list(by_identity.values())

    def _quarantine(self):
        broken = self.path.with_suffix(".corrupt.json")
        if broken.exists():
            broken = self.path.with_suffix(f".corrupt.{uuid.uuid4().hex}.json")
        try:
            self.path.replace(broken)
        except OSError:
            self.load_error = "Invalid memory file could not be preserved; writes are disabled."
            warnings.warn(self.load_error, RuntimeWarning)
        else:
            warnings.warn("Invalid memory file preserved separately; starting with empty memory.",
                          RuntimeWarning)

    @staticmethod
    def _valid(entry):
        return (isinstance(entry, dict) and isinstance(entry.get("text"), str)
                and bool(_normalize(entry["text"]))
                and entry.get("kind") in KINDS and entry.get("source") in SOURCES)

    @staticmethod
    def _repair_metadata(entry):
        # Unknown historical dates stay visibly old; do not invent a recent
        # confirmation when migrating a hand-edited or older memory file.
        created = _timestamp(entry.get("created"), "1970-01-01T00:00:00.000000+00:00")
        text = _normalize(entry["text"])
        identity = _identity(text, entry["kind"])
        return {
            "id": entry.get("id") if isinstance(entry.get("id"), str) and entry["id"]
                  else f"m_{uuid.uuid4().hex[:8]}",
            "text": text,
            "kind": identity[0],
            "source": entry["source"],
            "created": created,
            "updated": _timestamp(entry.get("updated"), created),
        }

    def save(self, entries=None):
        if self.load_error:
            raise OSError(self.load_error)
        values = self.entries if entries is None else entries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                      prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(values, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _commit(self, entries):
        # A failed write must leave both the live state and previous disk
        # contents intact, so callers cannot claim an unsaved change succeeded.
        self.save(entries)
        self.entries = entries

    # ---- mutation ----

    @_locked
    def add(self, text, kind="fact", source="stated"):
        """Add or update one entry. Returns the entry, or None if unusable."""
        text = _normalize(text)
        if not text or len(text) > MAX_ENTRY_CHARS or kind not in KINDS or source not in SOURCES:
            return None
        key = _identity(text, kind)
        kind = key[0]
        existing = next((e for e in self.entries if _identity(e["text"], e["kind"]) == key), None)
        if existing:
            # Upgrading a guess to a confirmed fact is the point of tagging.
            if existing["source"] == "inferred" and source == "stated":
                upgraded = dict(existing, text=text, kind=kind, source="stated", updated=_now())
                self._commit([upgraded if e is existing else e for e in self.entries])
                self._saved_this_run.add(upgraded["id"])
                return upgraded
            return existing
        entry = {
            "id": f"m_{uuid.uuid4().hex[:8]}",
            "text": text,
            "kind": kind,
            "source": source,
            "created": _now(),
            "updated": _now(),
        }
        entries = self.entries + [entry]
        if len(entries) > MAX_TOTAL_ENTRIES:
            # Drop the oldest inferred entries first; never silently drop
            # something Luca explicitly asked TARS to keep.
            inferred = sorted((e for e in entries if e["source"] == "inferred"),
                              key=lambda e: e["updated"])
            for stale in inferred[:len(entries) - MAX_TOTAL_ENTRIES]:
                entries.remove(stale)
        if entry not in entries:
            return None
        self._commit(entries)
        self._saved_this_run.intersection_update(e["id"] for e in self.entries)
        self._saved_this_run.add(entry["id"])
        return entry

    @_locked
    def forget(self, text):
        """Remove one unambiguous match; raise AmbiguousMemory otherwise."""
        entry = self._match(text)
        if entry is None:
            return None
        self._commit([e for e in self.entries if e is not entry])
        self._saved_this_run.discard(entry["id"])
        return entry

    @_locked
    def forget_last(self):
        if not self.entries:
            return None
        _, entry = max(enumerate(self.entries), key=lambda pair: (pair[1]["updated"], pair[0]))
        self._commit([e for e in self.entries if e is not entry])
        self._saved_this_run.discard(entry["id"])
        return entry

    @_locked
    def forget_last_saved_this_run(self):
        candidates = [(index, entry) for index, entry in enumerate(self.entries)
                      if entry["id"] in self._saved_this_run]
        if not candidates:
            return None
        _, entry = max(candidates, key=lambda pair: (pair[1]["updated"], pair[0]))
        self._commit([e for e in self.entries if e is not entry])
        self._saved_this_run.discard(entry["id"])
        return entry

    @_locked
    def add_note(self, text, kind="fact"):
        """Quietly save one model-judged guess as UNCONFIRMED.

        A note that matches any existing entry changes nothing: it never
        promotes a guess, refreshes a timestamp, or counts as saved this run.
        """
        text = _normalize(text)
        if not text or len(text) > MAX_ENTRY_CHARS or kind not in ("fact", "preference"):
            return None
        key = _identity(text, kind)
        if any(_identity(e["text"], e["kind"]) == key for e in self.entries):
            return None
        return self.add(text, kind=kind, source="inferred")

    @_locked
    def checkin_candidates(self, limit, now=None):
        """Unconfirmed, non-joke entries at least CHECKIN_MIN_AGE old, newest first."""
        cutoff = (now or datetime.now(timezone.utc)) - CHECKIN_MIN_AGE
        guesses = [e for e in self.entries if e["source"] == "inferred" and e["kind"] != "bit"
                   and _created_before(e, cutoff)]
        guesses.sort(key=lambda e: e["updated"], reverse=True)
        return [dict(e) for e in guesses[:limit]]

    def _checked_entry(self, entry_id, revision):
        entry = next((e for e in self.entries if e["id"] == entry_id), None)
        if entry is None or entry["source"] != "inferred" or entry_revision(entry) != revision:
            return None
        return entry

    @_locked
    def confirm_checked(self, entry_id, revision):
        """Upgrade the exact guess a check-in asked about. False if it changed."""
        entry = self._checked_entry(entry_id, revision)
        if entry is None:
            return False
        upgraded = dict(entry, source="stated", updated=_now())
        self._commit([upgraded if e is entry else e for e in self.entries])
        self._saved_this_run.add(upgraded["id"])
        return True

    @_locked
    def retract_checked(self, entry_id, revision):
        """Delete the exact guess a check-in asked about. False if it changed."""
        entry = self._checked_entry(entry_id, revision)
        if entry is None:
            return False
        self._commit([e for e in self.entries if e is not entry])
        self._saved_this_run.discard(entry["id"])
        return True

    @_locked
    def latest(self):
        if not self.entries:
            return None
        return max(enumerate(self.entries),
                   key=lambda pair: (pair[1]["updated"], pair[0]))[1]

    # ---- retrieval ----

    def _match(self, text):
        needle = _normalize(text).casefold()
        if not needle:
            return None
        key = _identity(needle, "fact")
        matches = [e for e in self.entries if e["text"].casefold() == needle or
                   (key[0] == "preference" and _identity(e["text"], e["kind"]) == key)]
        if not matches and key[0] != "preference":
            # References may omit sentence framing, but all reference words
            # must appear together as whole words. No shared-word scoring.
            words = _tokens(needle)
            while words and words[0] in {"the", "a", "an"}:
                words.pop(0)
            while words and words[-1] in {"thing", "memory", "fact", "preference"}:
                words.pop()
            generic = {"i", "my", "me", "like", "likes", "prefer", "prefers",
                       "the", "that", "this", "it", "about", "memory", "fact"}
            if words and any(w not in generic for w in words):
                matches = [e for e in self.entries if _contains_tokens(_tokens(e["text"]), words)]
        if len(matches) > 1:
            raise AmbiguousMemory("More than one memory matches that description.")
        return matches[0] if matches else None

    @_locked
    def all(self):
        return [dict(entry) for entry in self.entries]

    @_locked
    def for_prompt(self):
        """Bounded, prioritised memory block. Confirmed first, bits last."""
        stated = [e for e in self.entries if e["source"] == "stated" and e["kind"] != "bit"]
        inferred = [e for e in self.entries if e["source"] == "inferred" and e["kind"] != "bit"]
        bits = [e for e in self.entries if e["kind"] == "bit"]
        stated.sort(key=lambda e: e["updated"], reverse=True)
        inferred.sort(key=lambda e: e["updated"], reverse=True)
        bits.sort(key=lambda e: e["updated"], reverse=True)
        chosen = ((stated + inferred)[:MAX_ENTRIES_IN_PROMPT] + bits[:MAX_BITS_IN_PROMPT])[:MAX_ENTRIES_IN_PROMPT]
        if not chosen:
            return None

        lines, used = [], 0
        for entry in chosen:
            if entry["kind"] == "bit":
                label = "RUNNING JOKE"
            elif entry["source"] == "stated":
                label = "CONFIRMED"
            else:
                label = "UNCONFIRMED"
            line = f"- [{label}] {entry['text'][:MAX_ENTRY_CHARS]}"
            cost = len(line) + bool(lines)
            if used + cost > MAX_PROMPT_CHARS:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines) if lines else None

    @_locked
    def spoken_summary(self, limit=8):
        """A short list for when Luca asks what TARS remembers."""
        if not self.entries:
            return "nothing yet"
        ordered = sorted(self.entries, key=lambda e: e["updated"], reverse=True)[:limit]
        parts = []
        for entry in ordered:
            if entry["kind"] == "bit":
                parts.append(f"a running joke: {entry['text']}")
            elif entry["source"] == "inferred":
                parts.append(f"{entry['text']} (not confirmed)")
            else:
                parts.append(entry["text"])
        return "; ".join(parts)


# --------------------------------------------------------------------------
# Turn handling — the single entry point brain.py needs
# --------------------------------------------------------------------------

def handle_memory_turn(store, user_input, last_reply=None, infer=True):
    """Apply memory effects for one turn. Returns a control note, or None.

    Explicit commands win. Inference runs only when the turn was not itself a
    memory command, so "remember that I like X" is stored once, as confirmed.
    """
    if store.load_error:
        raise OSError(store.load_error)
    parsed = parse_memory_command(user_input)

    if parsed:
        intent, payload = parsed
        if intent == "recall":
            return (f"Luca asked what you remember. Say briefly: {store.spoken_summary()}. "
                    "Flag anything marked not confirmed as a guess.")
        if intent == "remember_rejected":
            return ("Nothing was saved. Luca's wording was a question or reminiscence, "
                    "not a memory command. Do not claim you remembered it; answer him "
                    'normally. Nothing was deleted. For an intended save, suggest '
                    '"remember that <fact>" as a statement, not a question.')
        if intent == "remember":
            entry = store.add(payload, kind="fact", source="stated")
            if entry is None:
                return "No memory was saved. Ask Luca for a shorter, nonempty memory."
            return f"Luca asked you to remember: {entry['text']}. Confirm it in a few words."
        if intent == "remember_last":
            if not last_reply:
                return "Luca asked you to remember your last line, but there isn't one yet."
            entry = store.add(last_reply, kind="bit", source="stated")
            if entry is None:
                return "Luca asked you to remember your last line but it could not be saved."
            return "Luca saved that line of yours as a running joke. Acknowledge it briefly."
        if intent == "forget":
            try:
                entry = (store.forget(payload) if payload
                         else store.forget_last_saved_this_run())
            except AmbiguousMemory:
                return ("Several memories match. Nothing was deleted. Ask Luca to repeat "
                        "the forget request with the exact memory text or a unique description.")
            if entry is not None:
                if payload is not None:
                    return (f"Luca asked you to forget: {entry['text']}. It is deleted. "
                            "Confirm briefly.")
                return ("The memory was deleted. Deleted memory text: "
                        f"\"{entry['text']}\". Tell Luca exactly what you forgot.")
            if payload is not None:
                return ("Nothing was deleted. The request did not match a stored memory. "
                        'Do not claim "forgotten", "already done", or that it was '
                        'previously removed, and do not redirect Luca to settings. '
                        'Ask him to say "forget that <exact memory text>" or use a '
                        'unique description.')
            latest = store.latest()
            if latest is None:
                return "No memory was deleted because there are no stored memories. Say so plainly."
            updated = datetime.fromisoformat(latest["updated"]).astimezone()
            date = f"{updated.strftime('%B')} {updated.day}, {updated.year}"
            return ("Nothing was deleted because no memory was saved during this run. "
                    f"The most recent stored memory is \"{latest['text']}\", from {date}. "
                    f"Tell Luca to say exactly \"forget that {latest['text']}\" if he "
                    "wants it removed.")

    if is_memory_request(user_input):
        # Never let inference save a fact from the same guarded turn while
        # telling the model that nothing changed. The guard is a control note,
        # not a claim that the model's eventual spoken answer is guaranteed.
        return ("Nothing was saved or deleted this turn. No supported memory command "
                'was recognized. Do not claim "got it", "forgotten", "already done", '
                'or that a memory was previously removed; do not redirect Luca to '
                'settings. "Forget it" only drops the conversation topic; "forget '
                'them" is not a supported deletion command. If Luca intended a memory '
                'change, give the exact phrasing: "remember that <fact>" as a statement '
                'to save, or "forget that <exact memory text>" to remove. Otherwise '
                'answer normally without claiming a memory action.')

    if not infer:
        # Model-judged notes replace pattern inference after a completed turn.
        return None
    saved = [store.add(item["text"], kind=item["kind"], source="inferred")
             for item in infer_memories(user_input)]
    if any(saved):
        # Silent by design: an unconfirmed guess should not interrupt the
        # conversation to announce itself. It is visible via "what do you
        # remember" and in memory.json.
        return None
    return None
