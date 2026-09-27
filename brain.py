"""TARS brain: owns live state, conversation history, and Claude requests."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from memory import MemoryStore, handle_memory_turn
from personality import BASELINE, apply_command, build_personality


DEFAULT_MODEL = "claude-sonnet-4-6"
SUPPORTED_MODELS = (
    DEFAULT_MODEL,
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)
MAX_HISTORY_MESSAGES = 24


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
            self.last_reply = "".join(parts)
            self.conversation.append({"role": "assistant", "content": self.last_reply})
            if len(self.conversation) > MAX_HISTORY_MESSAGES:
                excess = len(self.conversation) - MAX_HISTORY_MESSAGES
                # Completed history is user/assistant pairs, so prune pairs.
                del self.conversation[:excess + (excess % 2)]
        elif self.conversation and self.conversation[-1] is user_message:
            # A request that failed before returning text must not leave an
            # orphan user turn that contaminates the next API request.
            self.conversation.pop()

    def respond_stream(self, user_input):
        """Yield text deltas and keep history consistent across interruptions."""
        note = apply_command(user_input, self.settings)
        try:
            memory_note = handle_memory_turn(self.memory, user_input, self.last_reply)
        except OSError:
            memory_note = ("Persistent memory could not be read or updated. Do not claim "
                           "anything was saved or deleted. If Luca asked to manage memory, "
                           "briefly explain the failure; otherwise continue normally.")
        note = "\n".join(n for n in (note, memory_note) if n) or None
        user_message = {"role": "user", "content": user_input}
        self.conversation.append(user_message)
        parts = []
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=build_personality(self.settings, note, self.voice, self.memory.for_prompt()),
                messages=self.conversation,
                **self._request_options(),
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        parts.append(text)
                        yield text
        finally:
            self._finish_turn(user_message, parts)
