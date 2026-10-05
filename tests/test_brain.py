"""Claude brain tests with a fake SDK; no network or paid calls."""

import importlib
from pathlib import Path
import sys
import types
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

anthropic_stub = types.ModuleType("anthropic")
anthropic_stub.Anthropic = MagicMock()
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = MagicMock()
with patch.dict(sys.modules, {"anthropic": anthropic_stub, "dotenv": dotenv_stub}):
    sys.modules.pop("brain", None)
    brain = importlib.import_module("brain")
from memory import MemoryStore


class FakeStream:
    def __init__(self, chunks=None, enter_error=None):
        self.text_stream = iter(chunks or [])
        self.enter_error = enter_error

    def __enter__(self):
        if self.enter_error:
            raise self.enter_error
        return self

    def __exit__(self, *_args):
        return False


class FakeMessages:
    def __init__(self, streams):
        self.streams = iter(streams)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.streams)


class FakeClient:
    def __init__(self, streams):
        self.messages = FakeMessages(streams)


class BrainTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.memory_path = Path(temporary.name) / "memory.json"
        self.memory = MemoryStore(self.memory_path)
        # Existing brain tests must never read or write the real default store.
        factory = patch.object(brain, "MemoryStore", return_value=self.memory)
        factory.start()
        self.addCleanup(factory.stop)

    def test_explicit_save_reaches_prompt_and_survives_new_brain(self):
        first = brain.TARS(client=FakeClient([FakeStream(["Remembered."])]), memory=self.memory)
        first.respond("remember that I prefer PETG Basic")
        reopened = MemoryStore(self.memory_path)
        client = FakeClient([FakeStream(["You prefer PETG Basic."])])
        second = brain.TARS(client=client, memory=reopened)
        second.respond("what do you remember about me")
        self.assertIn("[CONFIRMED] I prefer PETG Basic", client.messages.calls[0]["system"])
        self.assertEqual(second.conversation[0]["content"], "what do you remember about me")

    def test_voice_wake_prefix_save_and_recall_survive_restart(self):
        first = brain.TARS(client=FakeClient([FakeStream(["Got it."])]),
                           memory=self.memory, voice=True)
        first.respond("Hey, Tars, remember that I like my eggs scrambled")
        reopened = MemoryStore(self.memory_path)
        client = FakeClient([FakeStream(["You like your eggs scrambled."])])
        second = brain.TARS(client=client, memory=reopened, voice=True)
        second.respond("Hey Tars, what do you remember about me?")
        prompt = client.messages.calls[0]["system"]
        self.assertIn("[CONFIRMED] I like my eggs scrambled", prompt)
        self.assertIn("Say briefly: I like my eggs scrambled", prompt)

    def test_natural_inference_survives_restart_as_unconfirmed(self):
        first = brain.TARS(client=FakeClient([FakeStream(["Critical operational data."])]),
                           memory=self.memory, voice=True)
        first.respond("Just want to let you know that I like my mac and cheese really cheesy")
        reopened = MemoryStore(self.memory_path)
        client = FakeClient([FakeStream(["You like your mac and cheese really cheesy."])])
        second = brain.TARS(client=client, memory=reopened, voice=True)
        second.respond("What do you remember about me?")
        prompt = client.messages.calls[0]["system"]
        self.assertIn("[UNCONFIRMED] likes my mac and cheese really cheesy", prompt)
        self.assertIn("likes my mac and cheese really cheesy (not confirmed)", prompt)

    def test_failed_save_keeps_conversation_working_without_false_confirmation(self):
        client = FakeClient([FakeStream(["Memory storage is unavailable."])])
        tars = brain.TARS(client=client, memory=self.memory)
        with patch.object(self.memory, "save", side_effect=OSError("disk full")):
            tars.respond("remember that I like coffee")
        self.assertIn("Do not claim", client.messages.calls[0]["system"])
        self.assertNotIn("[CONFIRMED]", client.messages.calls[0]["system"])
        self.assertEqual(self.memory.all(), [])
        self.assertEqual(len(tars.conversation), 2)

    def test_last_reply_is_saved_as_joke_without_confirming_existing_fact(self):
        self.memory.add("I am made of gold", source="inferred")
        client = FakeClient([FakeStream(["I am made of gold"]), FakeStream(["Saved the joke."])])
        tars = brain.TARS(client=client, memory=self.memory)
        tars.respond("Tell me a joke")
        tars.respond("remember that")
        prompt = client.messages.calls[-1]["system"]
        self.assertIn("[UNCONFIRMED] I am made of gold", prompt)
        self.assertIn("[RUNNING JOKE] I am made of gold", prompt)
        self.assertNotIn("[CONFIRMED] I am made of gold", prompt)

    def test_sonnet_five_disables_thinking_for_short_voice_turns(self):
        client = FakeClient([FakeStream(["Hello", "."])])
        tars = brain.TARS(model="claude-sonnet-5", voice=True, client=client)

        self.assertEqual(tars.respond("Hi"), "Hello.")

        call = client.messages.calls[0]
        self.assertEqual(call["thinking"], {"type": "disabled"})
        self.assertEqual(call["max_tokens"], 160)
        self.assertEqual(tars.conversation[-1]["content"], "Hello.")

    def test_sonnet_four_sends_no_thinking_override(self):
        client = FakeClient([FakeStream(["Ready."])])
        tars = brain.TARS(model="claude-sonnet-4-6", client=client)
        tars.respond("Status?")
        self.assertNotIn("thinking", client.messages.calls[0])

    def test_failed_request_removes_orphan_user_message(self):
        client = FakeClient([FakeStream(enter_error=RuntimeError("offline"))])
        tars = brain.TARS(client=client)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            tars.respond("This request failed")
        self.assertEqual(tars.conversation, [])

    def test_interrupted_partial_reply_is_retained_as_a_complete_pair(self):
        client = FakeClient([FakeStream(["First sentence.", " Unheard remainder."])])
        tars = brain.TARS(client=client)
        response = tars.respond_stream("Begin")
        # Replies are yielded sentence by sentence, with a separating space.
        self.assertEqual(next(response), "First sentence. ")
        response.close()
        self.assertEqual(
            tars.conversation,
            [
                {"role": "user", "content": "Begin"},
                {"role": "assistant", "content": "First sentence."},
            ],
        )

    def test_history_is_pruned_in_complete_pairs(self):
        streams = [FakeStream([f"answer {index}"]) for index in range(20)]
        tars = brain.TARS(client=FakeClient(streams))
        for index in range(20):
            tars.respond(f"question {index}")
        self.assertLessEqual(len(tars.conversation), brain.MAX_HISTORY_MESSAGES)
        self.assertEqual(tars.conversation[0]["role"], "user")
        self.assertEqual(tars.conversation[-1]["role"], "assistant")

    def test_unknown_model_is_rejected_before_a_request(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Anthropic model"):
            brain.TARS(model="made-up-model", client=FakeClient([]))


if __name__ == "__main__":
    unittest.main()
