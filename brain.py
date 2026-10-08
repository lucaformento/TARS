"""TARS brain: owns live state, conversation history, and Claude requests."""

import json
import os
from pathlib import Path
import tempfile

from anthropic import Anthropic
from dotenv import load_dotenv

from memory import MemoryStore, entry_revision, handle_memory_turn, is_memory_request
from memory_notes import (MAX_CHECKIN_CANDIDATES, NOTE_TIMEOUT_SECONDS, CheckinState,
                          MarkerFilter, NoteTaker, PendingCheckin, TurnRecord, notes_enabled,
                          store_applier)
from personality import BASELINE, apply_command, build_personality, describe_mode


DEFAULT_MODEL = "claude-sonnet-4-6"
SUPPORTED_MODELS = (
    DEFAULT_MODEL,
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)
MAX_HISTORY_MESSAGES = 24
# Before an explicit memory command, give a running note job this long to
# finish so "forget that" sees the newest quiet note.
MEMORY_COMMAND_WAIT_SECONDS = 2.0
CHECKIN_ANSWER_NOTE = (
    "Your previous line asked Luca to confirm a saved guess. If he answered it, "
    "acknowledge briefly. The application updates memory afterward, so do not say "
    "it was saved, confirmed, or deleted."
)
# Personality dials kept across restarts until Luca resets them; lives next to
# memory.json and stays out of Git.
SETTINGS_FILE = "personality-state.json"


def read_settings(path):
    """Return saved dials, or None without a file. Invalid contents raise ValueError."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError("not valid JSON") from exc
    dials = raw.get("dials") if isinstance(raw, dict) and raw.get("version") == 1 else None
    if (not isinstance(dials, dict) or set(dials) != set(BASELINE)
            or not all(type(value) is int and 0 <= value <= 100 for value in dials.values())):
        raise ValueError("unexpected contents")
    return dials


def write_settings(path, settings):
    """Replace the settings file atomically so a crash never leaves half a file."""
    path = Path(path)
    payload = {"version": 1, "dials": {dial: settings[dial] for dial in BASELINE}}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class TARS:
    def __init__(self, model=None, max_tokens=None, voice=False, client=None, memory=None):
        load_dotenv()
        selected = (model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL).strip()
        if selected not in SUPPORTED_MODELS:
            choices = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"Unsupported Anthropic model {selected!r}. Choose: {choices}")
        self.client = client or Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=20.0,
            max_retries=1,
        )
        self.model = selected
        self.voice = voice
        self.max_tokens = max_tokens if max_tokens is not None else (160 if voice else 300)
        self.settings = BASELINE.copy()
        self.conversation = []
        self.memory = memory if memory is not None else MemoryStore()
        self.last_reply = None
        # Memory v2: off until enable_memory_notes(), so tests and tools that
        # build a brain never start a worker or make a second request.
        self.notes = None
        self.checkins = None
        self._session = 0
        self._pending = None    # (session, PendingCheckin) asked by the last reply
        self._completed = None  # inputs for one note job, set only by a finished turn
        # Saved settings: off until enable_saved_settings(), for the same reason.
        self._settings_path = None
        self._log = print

    def enable_saved_settings(self, log=print):
        """Load dials kept from an earlier run; later changes are saved as they happen."""
        self._log = log
        path = self.memory.path.with_name(SETTINGS_FILE)
        try:
            saved = read_settings(path)
        except (OSError, ValueError) as exc:
            log(f"[settings] {path.name} could not be read ({exc}); starting at baseline. "
                "The next personality change replaces it.")
            saved = None
        if saved:
            self.settings.update(saved)
        self._settings_path = path
        return describe_mode(self.settings)

    def _save_settings(self):
        if self._settings_path is None:
            return
        try:
            write_settings(self._settings_path, self.settings)
        except OSError:
            self._log("[settings] could not save the change; it lasts until TARS restarts.")

    # ---- memory v2 lifecycle ----

    def enable_memory_notes(self, log=print):
        """Start the note-taker and daily check-ins unless TARS_MEMORY_NOTES is off."""
        if not notes_enabled() or self.notes is not None:
            return self.notes is not None
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"),
                           timeout=NOTE_TIMEOUT_SECONDS, max_retries=0)
        self.notes = NoteTaker(client, store_applier(self.memory), log=log)
        self.checkins = CheckinState(self.memory.path.with_name("memory-state.json"), warn=log)
        return True

    def begin_session(self):
        """A new wake. A check-in answer only counts within the same session."""
        self._session += 1
        self._pending = None

    def submit_note(self):
        """Queue the last fully completed turn. Call only after playback succeeded."""
        completed, self._completed = self._completed, None
        if completed is None or self.notes is None:
            return False
        with self.memory.lock:
            epoch = self.memory.epoch
        return self.notes.submit(TurnRecord(epoch=epoch, **completed))

    def close(self):
        if self.notes is not None:
            self.notes.stop(1.0)

    def respond(self, user_input):
        """Collect streamed text for front-ends that need a complete reply."""
        return "".join(self.respond_stream(user_input))

    def _request_options(self):
        # Sonnet 5 enables adaptive thinking by default. TARS voice turns are
        # short and latency-sensitive, so preserve the previous no-thinking
        # behavior and reserve the full output budget for spoken text.
        if self.model == "claude-sonnet-5":
            return {"thinking": {"type": "disabled"}}
        return {}

    def _finish_turn(self, user_message, parts):
        if parts:
            self.last_reply = "".join(parts).strip()
            self.conversation.append({"role": "assistant", "content": self.last_reply})
            if len(self.conversation) > MAX_HISTORY_MESSAGES:
                excess = len(self.conversation) - MAX_HISTORY_MESSAGES
                # Completed history is user/assistant pairs, so prune pairs.
                del self.conversation[:excess + (excess % 2)]
        elif self.conversation and self.conversation[-1] is user_message:
            # A request that failed before returning text must not leave an
            # orphan user turn that contaminates the next API request.
            self.conversation.pop()

    def _offer_checkins(self, memory_request):
        """Reserve today's check-in allowance before generation, or offer nothing."""
        if memory_request or self.notes is None or self.checkins is None:
            return {}
        if not self.checkins.available():
            return {}
        entries = self.memory.checkin_candidates(MAX_CHECKIN_CANDIDATES)
        offered = {f"c{index}": entry for index, entry in enumerate(entries, start=1)}
        candidates = [{"label": label, "entry_id": entry["id"], "revision": entry_revision(entry)}
                      for label, entry in offered.items()]
        return offered if self.checkins.reserve(candidates) else {}

    def _settle_checkin(self, offered, marker_filter, completed):
        if not offered:
            return
        label = marker_filter.recorded_label
        self.checkins.settle(completed, asked=label is not None)
        if completed and label is not None:
            entry = offered[label]
            self._pending = (self._session, PendingCheckin(
                entry["id"], entry_revision(entry), entry["text"],
                marker_filter.recorded_question))

    def respond_stream(self, user_input):
        """Yield reply text sentence by sentence; keep history and memory consistent.

        Check-in markers are removed or suppressed before any text is yielded.
        A note job is prepared only when the model stream finishes on its own;
        the caller submits it with submit_note() after playback succeeds.
        """
        self._completed = None
        pending = self._pending[1] if self._pending and self._pending[0] == self._session else None
        self._pending = None  # an answer counts for exactly this turn
        before = dict(self.settings)
        note = apply_command(user_input, self.settings)
        if self.settings != before:
            self._save_settings()
        memory_request = is_memory_request(user_input)
        if memory_request and self.notes is not None:
            self.notes.wait_idle(MEMORY_COMMAND_WAIT_SECONDS)
        try:
            with self.memory.lock:
                if memory_request:
                    # Invalidate outstanding note jobs before the command runs,
                    # even when it ends up changing nothing.
                    self.memory.epoch += 1
                memory_note = handle_memory_turn(self.memory, user_input, self.last_reply,
                                                 infer=self.notes is None)
        except OSError:
            memory_note = ("Persistent memory could not be read or updated. Do not claim "
                           "anything was saved or deleted. If Luca asked to manage memory, "
                           "briefly explain the failure; otherwise continue normally.")
        memory_block = self.memory.for_prompt()
        offered = self._offer_checkins(memory_request)
        notes = (note, memory_note, CHECKIN_ANSWER_NOTE if pending is not None else None)
        note = "\n".join(n for n in notes if n) or None
        user_message = {"role": "user", "content": user_input}
        self.conversation.append(user_message)
        marker_filter = MarkerFilter(offered)
        parts = []
        completed = False
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=build_personality(
                    self.settings, note, self.voice, memory_block,
                    [(label, entry["text"]) for label, entry in offered.items()]),
                messages=self.conversation,
                **self._request_options(),
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        for sentence in marker_filter.feed(text):
                            parts.append(sentence)
                            yield sentence
                for sentence in marker_filter.finish():
                    parts.append(sentence)
                    yield sentence
            completed = True
        finally:
            self._finish_turn(user_message, parts)
            self._settle_checkin(offered, marker_filter, completed)
            if completed and not memory_request:
                self._completed = {"utterance": user_input, "reply": marker_filter.reply,
                                   "checkin": pending, "memory_block": memory_block}
