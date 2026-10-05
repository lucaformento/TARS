"""Memory v2 phase 2: store locking, the brain's note and check-in flow, and
stale-result protection. Fake Anthropic clients and a fake clock; no paid calls."""

from datetime import date
import importlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

anthropic_stub = types.ModuleType("anthropic")
anthropic_stub.Anthropic = MagicMock()
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = MagicMock()
with patch.dict(sys.modules, {"anthropic": anthropic_stub, "dotenv": dotenv_stub}):
    sys.modules.pop("brain", None)
    brain = importlib.import_module("brain")
import memory_notes as notes
from memory import MemoryStore, entry_revision
from memory_notes import Action, CheckinState, NoteTaker, PendingCheckin, TurnRecord, store_applier


class FakeStream:
    def __init__(self, chunks=(), error=None):
        self.chunks = list(chunks)
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def text_stream(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class ReplyClient:
    """Fake brain client: one scripted stream per turn; records each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def stream(self, **request):
        self.calls.append(request)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, FakeStream) else FakeStream([reply])


class NoteClient:
    """Fake note-taker model: returns scripted actions, optionally after a gate."""

    def __init__(self, *payloads, gate=None):
        self.payloads = list(payloads)
        self.requests = []
        self.gate = gate
        self.started = threading.Event()
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(2)
        return SimpleNamespace(stop_reason="tool_use", content=[
            SimpleNamespace(type="tool_use", name=notes.TOOL_NAME, input=self.payloads.pop(0))])


class Clock:
    def __init__(self, day=date(2026, 10, 5)):
        self.day = day

    def __call__(self):
        return self.day


class StoreTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = MemoryStore(Path(tmp.name) / "memory.json")

    def test_methods_return_copies_never_live_entries(self):
        entry = self.store.add("I prefer PETG")
        entry["text"] = "tampered"
        self.assertEqual(self.store.all()[0]["text"], "I prefer PETG")
        latest = self.store.latest()
        latest["source"] = "tampered"
        self.assertEqual(self.store.all()[0]["source"], "stated")

    def test_quiet_note_is_unconfirmed_and_never_promotes_or_refreshes(self):
        stated = self.store.add("Luca prefers PETG", kind="preference")
        self.assertIsNone(self.store.add_note("Luca prefers PETG", "preference"))
        self.assertEqual(self.store.all(), [stated])
        guess = self.store.add_note("Luca is getting a 3D printer", "fact")
        self.assertEqual(guess["source"], "inferred")
        self.assertIsNone(self.store.add_note("Luca is getting a 3D printer", "fact"))
        self.assertEqual(self.store.all()[1]["updated"], guess["updated"])
        self.assertIsNone(self.store.add_note("x" * 241, "fact"))
        self.assertIsNone(self.store.add_note("Luca tells jokes", "bit"))

    def test_quiet_note_counts_as_saved_this_run_for_forget_that(self):
        self.store.add_note("Luca is getting a 3D printer", "fact")
        forgotten = self.store.forget_last_saved_this_run()
        self.assertEqual(forgotten["text"], "Luca is getting a 3D printer")
        self.assertEqual(self.store.all(), [])

    def test_check_in_changes_need_the_exact_revision(self):
        guess = self.store.add_note("Luca is getting a 3D printer", "fact")
        revision = entry_revision(guess)
        self.assertFalse(self.store.confirm_checked(guess["id"], revision + "x"))
        self.assertFalse(self.store.confirm_checked("m_missing", revision))
        self.assertTrue(self.store.confirm_checked(guess["id"], revision))
        self.assertEqual(self.store.all()[0]["source"], "stated")
        # Once confirmed, the old revision can no longer retract it.
        self.assertFalse(self.store.retract_checked(guess["id"], revision))
        other = self.store.add_note("Luca is building an arm", "fact")
        self.assertTrue(self.store.retract_checked(other["id"], entry_revision(other)))
        self.assertEqual([e["text"] for e in self.store.all()], ["Luca is getting a 3D printer"])

    def test_check_in_candidates_are_guesses_only_newest_first(self):
        self.store.add("I prefer PETG")
        first = self.store.add_note("Luca is getting a 3D printer", "fact")
        second = self.store.add_note("Luca is building an arm", "fact")
        self.assertEqual([e["id"] for e in self.store.checkin_candidates(5)],
                         [second["id"], first["id"]])
        self.assertEqual(len(self.store.checkin_candidates(1)), 1)

    def test_concurrent_writers_leave_a_consistent_file(self):
        def writer(prefix):
            for index in range(25):
                self.store.add_note(f"Luca {prefix} fact {index}", "fact")
                self.store.forget_last_saved_this_run()

        threads = [threading.Thread(target=writer, args=(name,)) for name in ("a", "b", "c")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        on_disk = json.loads(self.store.path.read_text())
        self.assertEqual(on_disk, self.store.all())
        self.assertEqual(on_disk, [])


class ApplierTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = MemoryStore(Path(tmp.name) / "memory.json")
        self.apply = store_applier(self.store)

    def record(self, **kwargs):
        values = {"utterance": "u", "reply": "r", "epoch": self.store.epoch}
        values.update(kwargs)
        return TurnRecord(**values)

    def test_add_from_the_current_epoch_is_saved(self):
        outcome = self.apply(self.record(), Action("add", "Luca is getting a printer", "fact"),
                             lambda: False)
        self.assertEqual(outcome, "saved")
        self.assertEqual(self.store.all()[0]["source"], "inferred")

    def test_stale_or_stopped_results_are_dropped(self):
        record = self.record()
        self.store.epoch += 1
        action = Action("add", "Luca is getting a printer", "fact")
        self.assertEqual(self.apply(record, action, lambda: False), "dropped (memory changed)")
        self.assertEqual(self.apply(self.record(), action, lambda: True), "dropped (shutting down)")
        self.assertEqual(self.store.all(), [])

    def test_confirm_applies_only_to_the_pending_entry_and_advances_the_epoch(self):
        guess = self.store.add_note("Luca is getting a printer", "fact")
        pending = PendingCheckin(guess["id"], entry_revision(guess), guess["text"], "Still?")
        self.assertEqual(self.apply(self.record(), Action("confirm"), lambda: False),
                         "dropped (entry changed)")
        epoch = self.store.epoch
        self.assertEqual(self.apply(self.record(checkin=pending), Action("confirm"), lambda: False),
                         "done")
        self.assertEqual(self.store.epoch, epoch + 1)
        self.assertEqual(self.store.all()[0]["source"], "stated")


class BrainFlowTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.store = MemoryStore(self.dir / "memory.json")
        self.clock = Clock()
        self.logs = []

    def make(self, *replies, note_client=None, voice=True):
        tars = brain.TARS(client=ReplyClient(*replies), memory=self.store, voice=voice)
        if note_client is not None:
            tars.notes = NoteTaker(note_client, store_applier(self.store), log=self.logs.append)
            tars.checkins = CheckinState(self.dir / "memory-state.json", today=self.clock,
                                         warn=self.logs.append)
            self.addCleanup(tars.close)
        return tars

    def say(self, tars, text):
        return "".join(tars.respond_stream(text))

    def system(self, tars, turn=-1):
        return tars.client.calls[turn]["system"]

    def used_dates(self):
        return json.loads((self.dir / "memory-state.json").read_text())["used_dates"]

    # ---- turn gating ----

    def test_completed_turn_submits_one_record_with_its_own_memory_block(self):
        self.store.add("I prefer PETG")
        note_client = NoteClient({"action": "add", "kind": "fact",
                                  "text": "Luca is getting a 3D printer around November 13"})
        tars = self.make("Nice. Then I get a body.", note_client=note_client)
        self.say(tars, "I'm getting a 3D printer around November 13.")
        self.assertTrue(tars.submit_note())
        self.assertFalse(tars.submit_note())  # one turn, one job
        self.assertTrue(tars.notes.wait_idle(2))
        sent = note_client.requests[0]["messages"][0]["content"]
        self.assertIn("[CONFIRMED] I prefer PETG", sent)
        self.assertIn("TARS: Nice. Then I get a body.", sent)
        texts = {e["text"]: e["source"] for e in self.store.all()}
        self.assertEqual(texts["Luca is getting a 3D printer around November 13"], "inferred")
        self.assertEqual(self.logs, ["[memory note] add: saved"])

    def test_interrupted_or_failed_turns_submit_nothing(self):
        tars = self.make(FakeStream(["First. ", "Second. ", "Third."]),
                         FakeStream(["Partial. "], error=RuntimeError("network")),
                         note_client=NoteClient())
        response = tars.respond_stream("Tell me three things.")
        next(response)
        response.close()
        self.assertFalse(tars.submit_note())
        with self.assertRaises(RuntimeError):
            self.say(tars, "Try again.")
        self.assertFalse(tars.submit_note())

    def test_a_new_turn_discards_an_unsubmitted_record(self):
        tars = self.make("One.", FakeStream(["Two. "], error=RuntimeError("x")),
                         note_client=NoteClient())
        self.say(tars, "First question.")  # playback then failed; never submitted
        with self.assertRaises(RuntimeError):
            self.say(tars, "Second question.")
        self.assertFalse(tars.submit_note())

    def test_memory_command_turns_never_submit_and_advance_the_epoch(self):
        tars = self.make("Saved.", "Nothing deleted.", "Here you go.", note_client=NoteClient())
        for text in ("remember that I prefer PETG", "forget them my test word is pineapple",
                     "what do you remember"):
            epoch = self.store.epoch
            self.say(tars, text)
            self.assertFalse(tars.submit_note(), text)
            self.assertEqual(self.store.epoch, epoch + 1, text)
        self.assertEqual([e["text"] for e in self.store.all()], ["I prefer PETG"])

    def test_pattern_inference_runs_only_when_notes_are_off(self):
        off = self.make("Noted.")
        self.say(off, "I like PETG.")
        self.assertEqual(len(self.store.all()), 1)
        on = self.make("Noted.", note_client=NoteClient())
        self.say(on, "I like ASA.")
        self.assertEqual(len(self.store.all()), 1)  # only the model may add now

    def test_honesty_rule_is_always_in_the_prompt(self):
        tars = self.make("Hi.")
        self.say(tars, "Delete that memory.")
        self.assertIn("Only the TARS application changes your memory", self.system(tars))

    # ---- stale results ----

    def test_forget_that_waits_for_a_queued_note_then_deletes_it(self):
        gate = threading.Event()
        note_client = NoteClient({"action": "add", "kind": "fact",
                                  "text": "Luca is getting a 3D printer"}, gate=gate)
        tars = self.make("Nice.", "Done.", note_client=note_client)
        self.say(tars, "I'm getting a 3D printer.")
        tars.submit_note()
        self.assertTrue(note_client.started.wait(2))
        threading.Timer(0.1, gate.set).start()  # finishes inside the 2 s wait
        self.say(tars, "Forget that.")
        self.assertEqual(self.store.all(), [])
        self.assertIn('"Luca is getting a 3D printer"', self.system(tars))

    def test_note_still_running_after_the_wait_is_dropped_even_if_nothing_was_deleted(self):
        gate = threading.Event()
        note_client = NoteClient({"action": "add", "kind": "fact",
                                  "text": "Luca is getting a 3D printer"}, gate=gate)
        tars = self.make("Nice.", "Nothing to forget.", note_client=note_client)
        self.say(tars, "I'm getting a 3D printer.")
        tars.submit_note()
        self.assertTrue(note_client.started.wait(2))
        with patch.object(brain, "MEMORY_COMMAND_WAIT_SECONDS", 0.05):
            self.say(tars, "Forget that.")  # finds nothing saved this run
        self.assertIn("No memory was deleted", self.system(tars))
        gate.set()
        self.assertTrue(tars.notes.wait_idle(2))
        self.assertEqual(self.store.all(), [])
        self.assertEqual(self.logs, ["[memory note] add: dropped (memory changed)"])

    # ---- check-ins ----

    def seed_guess(self):
        return self.store.add_note("Luca is getting a 3D printer around November 13", "fact")

    def test_marked_check_in_is_spoken_once_recorded_and_confirmable(self):
        guess = self.seed_guess()
        note_client = NoteClient({"action": "none"}, {"action": "confirm"})
        tars = self.make(FakeStream(["Sure. [check-in c1] You mentioned a printer in ",
                                     "November. Still the plan? "]),
                         "Good to hear.", note_client=note_client)
        spoken = self.say(tars, "I'm planning the robot body this weekend.")
        self.assertIn("[check-in c1] Luca is getting a 3D printer around November 13",
                      self.system(tars))
        self.assertNotIn("check-in", spoken)
        self.assertEqual(spoken.strip(), "Sure. You mentioned a printer in November. Still the plan?")
        self.assertNotIn("check-in", tars.conversation[-1]["content"])
        self.assertEqual(self.used_dates(), ["2026-10-05"])
        tars.submit_note()  # the asking turn's own job: nothing to note
        self.assertTrue(tars.notes.wait_idle(2))

        self.say(tars, "Yes, still on track.")
        self.assertIn(brain.CHECKIN_ANSWER_NOTE, self.system(tars))
        self.assertNotIn("CHECK-IN (optional)", self.system(tars))  # used for today
        tars.submit_note()
        self.assertTrue(tars.notes.wait_idle(2))
        sent = note_client.requests[-1]["messages"][0]["content"]
        self.assertIn("TARS asked: You mentioned a printer in November.", sent)
        self.assertEqual(self.store.all()[0]["id"], guess["id"])
        self.assertEqual(self.store.all()[0]["source"], "stated")

    def test_reply_without_a_marker_releases_the_allowance(self):
        self.seed_guess()
        tars = self.make("Sounds fun.", "Sure.", note_client=NoteClient())
        self.say(tars, "I'm planning the robot body this weekend.")
        self.assertIn("CHECK-IN (optional)", self.system(tars))
        self.assertEqual(self.used_dates(), [])
        self.say(tars, "Any tips?")
        self.assertIn("CHECK-IN (optional)", self.system(tars))

    def test_failed_turn_with_an_offer_uses_the_allowance(self):
        self.seed_guess()
        tars = self.make(FakeStream(["Hmm. "], error=RuntimeError("x")), note_client=NoteClient())
        with self.assertRaises(RuntimeError):
            self.say(tars, "I'm planning the robot body.")
        self.assertEqual(self.used_dates(), ["2026-10-05"])

    def test_pending_answer_expires_after_one_turn_and_at_a_new_wake(self):
        self.seed_guess()
        tars = self.make("[check-in c1] Still getting the printer? ", "Okay.", "Okay.",
                         "[check-in c1] Still getting it? ", "Sure.",
                         note_client=NoteClient())
        self.say(tars, "Robot plans.")
        self.say(tars, "Unrelated question.")
        self.say(tars, "Yes.")  # two turns later: no longer an answer
        self.assertNotIn(brain.CHECKIN_ANSWER_NOTE, self.system(tars))
        self.clock.day = date(2026, 10, 6)
        self.say(tars, "Robot plans again.")
        tars.begin_session()  # TARS slept and woke before the answer
        self.say(tars, "Yes.")
        self.assertNotIn(brain.CHECKIN_ANSWER_NOTE, self.system(tars))

    def test_no_check_in_on_memory_commands_or_when_notes_are_off(self):
        self.seed_guess()
        tars = self.make("Here.", note_client=NoteClient())
        self.say(tars, "what do you remember")
        self.assertNotIn("CHECK-IN (optional)", self.system(tars))
        off = self.make("Hi.")
        self.say(off, "Robot plans.")
        self.assertNotIn("CHECK-IN (optional)", self.system(off))

    def test_unoffered_marker_is_never_spoken(self):
        tars = self.make("Fine. [check-in c1] Still getting a printer? Anyway.",
                         note_client=NoteClient())
        spoken = self.say(tars, "Robot plans.")
        self.assertEqual(spoken, "Fine. Anyway.")


if __name__ == "__main__":
    unittest.main()
