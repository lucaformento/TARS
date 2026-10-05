"""Memory v2 standalone parts: schema, secret guard, check-in state, marker
filter, and the note-taker worker. Fake model and clock only; no paid calls."""

from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory_notes as notes
from memory_notes import (Action, CheckinState, MarkerFilter, NoteTaker, PendingCheckin,
                          TurnRecord, build_request, contains_secret, validate_action)


class ActionSchemaTests(unittest.TestCase):
    def test_valid_actions(self):
        self.assertEqual(validate_action({"action": "none"}), (Action("none"), None))
        self.assertEqual(
            validate_action({"action": "add", "text": " Luca is building a robot. ", "kind": "fact"}),
            (Action("add", "Luca is building a robot", "fact"), None))
        self.assertEqual(validate_action({"action": "confirm"}, checkin_pending=True),
                         (Action("confirm"), None))
        self.assertEqual(validate_action({"action": "retract"}, checkin_pending=True),
                         (Action("retract"), None))

    def test_invalid_output_is_none_with_a_reason(self):
        cases = [
            "add",
            ["action", "add"],
            {},
            {"action": "delete"},
            {"action": 3},
            {"action": "add", "text": "Luca likes PETG"},
            {"action": "add", "text": "Luca likes PETG", "kind": "fact", "id": "m_1"},
            {"action": "add", "text": 7, "kind": "fact"},
            {"action": "add", "text": "Luca likes PETG", "kind": "bit"},
            {"action": "add", "text": "Luca likes PETG", "kind": "preferences"},
            {"action": "add", "text": "   ", "kind": "fact"},
            {"action": "add", "text": "x" * 241, "kind": "fact"},
            {"action": "confirm", "id": "m_1"},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                action, reason = validate_action(payload, checkin_pending=True)
                self.assertEqual(action, Action("none"))
                self.assertIsNotNone(reason)

    def test_confirm_and_retract_need_a_pending_check_in(self):
        for kind in ("confirm", "retract"):
            action, reason = validate_action({"action": kind}, checkin_pending=False)
            self.assertEqual(action, Action("none"))
            self.assertEqual(reason, "no pending check-in")

    def test_extra_fields_on_none_are_harmless(self):
        self.assertEqual(validate_action({"action": "none", "text": ""}), (Action("none"), None))

    def test_response_parsing_requires_one_named_tool_call(self):
        def call(payload, name=notes.TOOL_NAME):
            return SimpleNamespace(type="tool_use", name=name, input=payload)
        ok = SimpleNamespace(stop_reason="tool_use", content=[call({"action": "none"})])
        self.assertEqual(notes.action_from_response(ok), (Action("none"), None))
        for response in (
            SimpleNamespace(stop_reason="max_tokens", content=[call({"action": "none"})]),
            SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="hi")]),
            SimpleNamespace(stop_reason="tool_use", content=[call({"action": "none"}),
                                                             call({"action": "none"})]),
            SimpleNamespace(stop_reason="tool_use", content=[call({"action": "none"}, "other")]),
        ):
            with self.subTest(response=response):
                action, reason = notes.action_from_response(response)
                self.assertEqual(action, Action("none"))
                self.assertIsNotNone(reason)


class SecretGuardTests(unittest.TestCase):
    BLOCKED = [
        "My PIN is 4821.",
        "my pin is four eight two one",
        "The door code is 1234.",
        "the combination is 12 34 56",
        "the code is A7X9K2",
        "My card is 4111 1111 1111 1111.",
        "card 4111-1111-1111-1111",
        "My social is 123-45-6789.",
        "my social is 123 45 6789",
        "The wifi password is hunter2.",
        "My bank account number ends in 42.",
        "Luca's debit card PIN code is private",
        "the verification code they sent was 551 204",
        "Use the key sk-ant-api03-abcdefghijklmnop",
        "token ghp_abcdefghijklmnopqrstuvwxyz123456",
        "my seed phrase is in the drawer",
    ]
    ALLOWED = [
        "My phone number is 555 123 4567.",
        "Call me at 555-123-4567 after school.",
        "I'm getting a 3D printer on November 13, 2026.",
        "The printer arrives 11.13.2026.",
        "The server runs on port 8080.",
        "I have an RTX 4090 in my PC.",
        "I ordered 1250 grams of PETG.",
        "My zip code is 94110.",
        "I fixed error code 404 in my Python code.",
        "My course is CS101.",
        "The course code is CS101.",
        "I wrote code in 2026.",
        "Connect the servo signal wire to GPIO pin 18.",
        "The header has 40 pins.",
        "I'm building TARS a body.",
    ]

    def test_blocked_examples(self):
        for text in self.BLOCKED:
            with self.subTest(text=text):
                self.assertTrue(contains_secret(text))

    def test_allowed_examples(self):
        for text in self.ALLOWED:
            with self.subTest(text=text):
                self.assertFalse(contains_secret(text))

    def test_either_the_utterance_or_the_note_can_trip_the_guard(self):
        self.assertTrue(contains_secret("my pin is 4821", "Luca likes the number 4821"))
        self.assertTrue(contains_secret("I like numbers", "Luca's PIN number is private"))
        self.assertFalse(contains_secret("I like PETG", "Luca likes PETG"))

    def test_social_security_pattern_survives_the_full_pipeline(self):
        # Normalization joins digit groups; the 3-2-4 check must run first.
        self.assertNotRegex(notes._normalized("123-45-6789"), r"-")
        self.assertTrue(contains_secret("number 123-45-6789"))
        self.assertTrue(contains_secret("number 123 45 6789"))


class FakeClock:
    def __init__(self, day):
        self.day = day

    def __call__(self):
        return self.day


class CheckinStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory-state.json"
        self.clock = FakeClock(date(2026, 10, 5))
        self.warnings = []

    def state(self):
        return CheckinState(self.path, today=self.clock, warn=self.warnings.append)

    def stored(self):
        return json.loads(self.path.read_text())

    def test_reserve_then_asked_uses_today_and_survives_restart(self):
        state = self.state()
        self.assertTrue(state.available())
        self.assertTrue(state.reserve([{"label": "c1", "entry_id": "m_1", "revision": "r"}]))
        self.assertEqual(self.stored()["reservation"]["date"], "2026-10-05")
        self.assertFalse(state.available())
        state.settle(completed=True, asked=True)
        self.assertEqual(self.stored(), {"version": 1, "used_dates": ["2026-10-05"],
                                         "reservation": None})
        self.assertFalse(self.state().available())
        self.clock.day += timedelta(days=1)
        self.assertTrue(self.state().available())

    def test_completed_turn_without_a_question_releases_the_reservation(self):
        state = self.state()
        state.reserve([{"label": "c1"}])
        state.settle(completed=True, asked=False)
        self.assertEqual(self.stored()["used_dates"], [])
        self.assertTrue(state.available())

    def test_failed_turn_consumes_the_allowance(self):
        state = self.state()
        state.reserve([{"label": "c1"}])
        state.settle(completed=False, asked=False)
        self.assertEqual(self.stored()["used_dates"], ["2026-10-05"])
        self.assertFalse(state.available())

    def test_question_across_midnight_uses_both_dates(self):
        state = self.state()
        state.reserve([{"label": "c1"}])
        self.clock.day = date(2026, 10, 6)
        state.settle(completed=True, asked=True)
        self.assertEqual(self.stored()["used_dates"], ["2026-10-05", "2026-10-06"])
        self.assertFalse(state.available())
        self.assertFalse(state.reserve([{"label": "c1"}]))

    def test_crash_across_midnight_uses_reservation_and_startup_dates(self):
        self.state().reserve([{"label": "c1"}])
        self.clock.day = date(2026, 10, 6)  # crash, then restart after midnight
        state = self.state()
        self.assertEqual(self.stored()["used_dates"], ["2026-10-05", "2026-10-06"])
        self.assertIsNone(self.stored()["reservation"])
        self.assertFalse(state.available())

    def test_small_clock_rollback_blocks_today(self):
        self.path.write_text(json.dumps({"version": 1, "used_dates": ["2026-10-06"],
                                         "reservation": None}))
        self.assertFalse(self.state().available())
        self.clock.day = date(2026, 10, 4)
        self.assertFalse(self.state().available())  # used date two days ahead
        self.clock.day = date(2026, 10, 3)
        self.assertTrue(self.state().available())   # three days ahead: bad clock

    def test_far_future_bad_clock_date_is_kept_but_does_not_block_today(self):
        self.path.write_text(json.dumps({"version": 1, "used_dates": ["2027-03-01"],
                                         "reservation": None}))
        state = self.state()
        self.assertTrue(state.available())
        state.reserve([{"label": "c1"}])
        state.settle(completed=True, asked=True)
        self.assertEqual(self.stored()["used_dates"], ["2026-10-05", "2027-03-01"])
        self.clock.day = date(2027, 3, 1)
        self.assertFalse(self.state().available())

    def test_old_used_dates_are_never_pruned(self):
        self.path.write_text(json.dumps({"version": 1, "used_dates": ["2026-01-02"],
                                         "reservation": None}))
        state = self.state()
        state.reserve([{"label": "c1"}])
        state.settle(completed=True, asked=True)
        self.assertIn("2026-01-02", self.stored()["used_dates"])
        self.clock.day = date(2026, 1, 2)  # a rollback of nine months
        self.assertFalse(self.state().available())

    def test_corrupt_state_is_quarantined_and_today_is_used(self):
        for bad in ("{not json", json.dumps({"used_dates": "2026-10-05"}),
                    json.dumps({"version": 1, "used_dates": ["yesterday"]}), json.dumps([1])):
            with self.subTest(bad=bad):
                for old in Path(self.tmp.name).glob("memory-state.corrupt.*.json"):
                    old.unlink()
                self.path.write_text(bad)
                state = self.state()
                self.assertFalse(state.available())
                self.assertFalse(state.disabled)
                self.assertEqual(self.stored()["used_dates"], ["2026-10-05"])
                preserved = list(Path(self.tmp.name).glob("memory-state.corrupt.*.json"))
                self.assertEqual([p.read_text() for p in preserved], [bad])
                self.clock.day = date(2026, 10, 6)
                self.assertTrue(self.state().available())
                self.clock.day = date(2026, 10, 5)

    def test_unreadable_state_disables_check_ins(self):
        self.path.mkdir()  # reading a directory raises an OSError that is not corruption
        state = self.state()
        self.assertTrue(state.disabled)
        self.assertFalse(state.available())
        self.assertFalse(state.reserve([{"label": "c1"}]))
        self.assertTrue(self.path.is_dir())
        self.assertTrue(self.warnings)

    def test_unwritable_state_offers_nothing(self):
        state = self.state()
        with patch.object(notes.os, "replace", side_effect=PermissionError("read-only")):
            self.assertFalse(state.reserve([{"label": "c1"}]))
        self.assertTrue(state.disabled)
        self.assertIsNone(state.reservation)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])  # temp file removed


class MarkerFilterTests(unittest.TestCase):
    def run_filter(self, deltas, offered=("c1", "c2")):
        flt = MarkerFilter(offered)
        out = []
        for delta in deltas:
            out.extend(flt.feed(delta))
        out.extend(flt.finish())
        return flt, "".join(out)

    def test_reply_without_markers_is_unchanged(self):
        text = "Fifty-six. Mercury is first, Neptune is last. Version 2.5 works."
        flt, spoken = self.run_filter([text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(spoken, text)
        self.assertEqual(flt.reply, text)
        self.assertIsNone(flt.recorded_label)

    def test_first_valid_marker_is_spoken_without_the_marker(self):
        flt, spoken = self.run_filter(
            ["Sure thing. [check-in c2] You mentioned a printer", " in November. Still the plan? ",
             "Anyway."])
        self.assertEqual(spoken, "Sure thing. You mentioned a printer in November. "
                                 "Still the plan? Anyway.")
        self.assertEqual(flt.recorded_label, "c2")
        self.assertEqual(flt.recorded_question, "You mentioned a printer in November.")

    def test_marker_split_across_deltas(self):
        flt, spoken = self.run_filter(["[che", "ck-i", "n c", "1] Still building ", "the arm?"])
        self.assertEqual(spoken, "Still building the arm?")
        self.assertEqual(flt.recorded_label, "c1")

    def test_suppressed_sentences(self):
        cases = {
            "repeated in a later sentence": ["[check-in c1] Printer still coming? ",
                                             "[check-in c2] Arm still planned? Good."],
            "repeated in the same sentence": ["[check-in c1] Printer [check-in c2] still coming? "
                                              "Good."],
            "label not offered": ["[check-in c4] Printer still coming? Good."],
            "malformed marker": ["[check-in 1] Printer still coming? Good."],
            "marker not at the start": ["So, [check-in c1] printer still coming? Good."],
            "incomplete marker at the end": ["Good. [check-in c"],
            "incomplete marker prefix at the end": ["Good. [che"],
        }
        expected = {
            "repeated in a later sentence": ("Printer still coming? Good.", "c1"),
            "repeated in the same sentence": ("Good.", None),
            "label not offered": ("Good.", None),
            "malformed marker": ("Good.", None),
            "marker not at the start": ("Good.", None),
            "incomplete marker at the end": ("Good.", None),
            "incomplete marker prefix at the end": ("Good.", None),
        }
        for name, deltas in cases.items():
            with self.subTest(name):
                flt, spoken = self.run_filter(deltas)
                self.assertEqual((spoken.strip(), flt.recorded_label), expected[name])
                self.assertNotIn("check", flt.reply.lower())

    def test_marker_without_any_offer_is_suppressed(self):
        flt, spoken = self.run_filter(["Hi. [check-in c1] Printer still coming? Bye."], offered=())
        self.assertEqual(spoken, "Hi. Bye.")
        self.assertIsNone(flt.recorded_label)

    def test_ordinary_brackets_pass(self):
        flt, spoken = self.run_filter(["Noted [twice] for safety. Done."])
        self.assertEqual(spoken, "Noted [twice] for safety. Done.")

    def test_output_resplits_identically_downstream(self):
        from sentences import split_sentences
        flt, spoken = self.run_filter(["One. [check-in c1] Two? ", "Three! Four."])
        self.assertEqual(split_sentences(spoken, final=True)[0], ["One.", "Two?", "Three!", "Four."])


class FakeMessages:
    def __init__(self, responses=None, gate=None):
        self.responses = list(responses or [])
        self.requests = []
        self.gate = gate

    def create(self, **request):
        self.requests.append(request)
        if self.gate is not None:
            self.gate.wait(2)
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def tool_response(payload):
    return SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="tool_use", name=notes.TOOL_NAME, input=payload)])


class NoteTakerTests(unittest.TestCase):
    def make(self, responses, gate=None, block="- [CONFIRMED] Luca likes PETG"):
        self.client = SimpleNamespace(messages=FakeMessages(responses, gate))
        self.logs = []
        self.applied = []
        self.lock = threading.Lock()

        def apply(record, action, stopped):
            with self.lock:
                if stopped():
                    return "dropped"
                self.applied.append((record, action))
                return "saved"

        def memory_block():
            with self.lock:
                return block

        taker = NoteTaker(self.client, memory_block, apply, log=self.logs.append)
        self.addCleanup(taker.stop)
        return taker

    def record(self, **kwargs):
        values = {"utterance": "I'm getting a 3D printer around November 13.",
                  "reply": "Good. Then I get a body.", "epoch": 3}
        values.update(kwargs)
        return TurnRecord(**values)

    def test_add_is_applied_with_the_immutable_record(self):
        taker = self.make([tool_response({"action": "add", "kind": "fact",
                                          "text": "Luca is getting a 3D printer around November 13"})])
        record = self.record()
        self.assertTrue(taker.submit(record))
        self.assertTrue(taker.wait_idle(2))
        self.assertEqual(self.applied, [(record, Action(
            "add", "Luca is getting a 3D printer around November 13", "fact"))])
        self.assertEqual(self.logs, ["[memory note] add: saved"])

    def test_request_is_bounded_and_forced(self):
        request = build_request(self.record(utterance="u" * 5000, reply="r" * 5000), "- [CONFIRMED] x")
        self.assertEqual(request["model"], notes.NOTE_MODEL)
        self.assertEqual(request["max_tokens"], notes.NOTE_MAX_TOKENS)
        self.assertEqual(request["tool_choice"], {"type": "tool", "name": notes.TOOL_NAME})
        content = request["messages"][0]["content"]
        self.assertLess(len(content), 2 * notes.MAX_UTTERANCE_CHARS + 200)
        self.assertIn("- [CONFIRMED] x", content)
        self.assertNotIn("<check_in>", content)
        pending = PendingCheckin("m_1", "rev", "Luca is getting a printer", "Still the plan?")
        content = build_request(self.record(checkin=pending), None)["messages"][0]["content"]
        self.assertIn("TARS asked: Still the plan?", content)
        self.assertIn("(empty)", content)

    def test_failures_change_nothing_and_never_log_text(self):
        secret_note = {"action": "add", "kind": "fact", "text": "Luca's PIN is 4821"}
        taker = self.make([RuntimeError("network down: 4821"), tool_response({"action": "delete"}),
                           tool_response(secret_note)])
        for _ in range(3):
            taker.submit(self.record(utterance="my pin is 4821"))
            self.assertTrue(taker.wait_idle(2))
        self.assertEqual(self.applied, [])
        self.assertEqual(self.logs, ["[memory note] request failed: RuntimeError",
                                     "[memory note] ignored model output: unsupported action",
                                     "[memory note] rejected: secret guard"])
        self.assertFalse(any("4821" in line for line in self.logs))

    def test_confirm_requires_the_pending_check_in(self):
        taker = self.make([tool_response({"action": "confirm"}), tool_response({"action": "confirm"})])
        taker.submit(self.record(utterance="yes"))
        self.assertTrue(taker.wait_idle(2))
        self.assertEqual(self.applied, [])
        pending = PendingCheckin("m_1", "rev", "Luca is getting a printer", "Still the plan?")
        taker.submit(self.record(utterance="yes", checkin=pending))
        self.assertTrue(taker.wait_idle(2))
        self.assertEqual([a for _, a in self.applied], [Action("confirm")])

    def test_memory_command_turns_are_never_submitted(self):
        taker = self.make([])
        self.assertFalse(taker.submit(self.record(memory_command=True)))
        self.assertEqual(self.client.messages.requests, [])

    def test_queue_holds_one_job_and_drops_the_next(self):
        gate = threading.Event()
        taker = self.make([tool_response({"action": "none"})] * 2, gate=gate)
        self.assertTrue(taker.submit(self.record()))
        for _ in range(100):  # wait for the worker to take the first job
            if self.client.messages.requests:
                break
            threading.Event().wait(0.01)
        self.assertTrue(taker.submit(self.record()))   # queued behind the running job
        self.assertFalse(taker.submit(self.record()))  # queue full
        self.assertIn("[memory note] skipped: busy", self.logs)
        self.assertFalse(taker.wait_idle(0.05))
        gate.set()
        self.assertTrue(taker.wait_idle(2))

    def test_model_call_runs_without_the_store_lock(self):
        lock_states = []

        class LockCheckingMessages(FakeMessages):
            def create(inner, **request):
                lock_states.append(self.lock.locked())
                return super().create(**request)

        taker = self.make([])
        self.client.messages = LockCheckingMessages([tool_response({"action": "none"})])
        taker._client = self.client
        taker.submit(self.record())
        self.assertTrue(taker.wait_idle(2))
        self.assertEqual(lock_states, [False])

    def test_late_result_after_stop_is_discarded(self):
        gate = threading.Event()
        taker = self.make([tool_response({"action": "add", "kind": "fact",
                                          "text": "Luca is building a robot"})], gate=gate)
        taker.submit(self.record())
        for _ in range(100):
            if self.client.messages.requests:
                break
            threading.Event().wait(0.01)
        taker.stop(timeout=0.05)
        self.assertFalse(taker.submit(self.record()))
        gate.set()
        threading.Event().wait(0.2)
        self.assertEqual(self.applied, [])

    def test_module_never_touches_audio(self):
        source = Path(notes.__file__).read_text()
        self.assertNotIn("sounddevice", source)
        self.assertNotIn("cloud_speech", source)


if __name__ == "__main__":
    unittest.main()
