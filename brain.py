"""TARS brain: owns live state (dials + conversation) and talks to Claude.
Front-ends (text, voice) drive this class and know nothing about personality logic."""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

from personality import BASELINE, build_personality, apply_command


class TARS:
    def __init__(self, model="claude-sonnet-4-6", max_tokens=None, voice=False):
        load_dotenv()
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.voice = voice
        # voice replies are short by design; text can afford to be longer
        self.max_tokens = max_tokens if max_tokens else (120 if voice else 200)
        self.settings = BASELINE.copy()     # source of truth for personality
        self.conversation = []              # dies with the process (for now)

    def respond(self, user_input):
        """Commands mutate state and ride the system prompt (control plane).
        The conversation only ever holds what Luca actually said (data plane)."""
        note = apply_command(user_input, self.settings)
        self.conversation.append({"role": "user", "content": user_input})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=build_personality(self.settings, note, self.voice),
            messages=self.conversation,
        )
        reply = response.content[0].text
        self.conversation.append({"role": "assistant", "content": reply})
        return reply
