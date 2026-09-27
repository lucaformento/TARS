"""Memory tests: parsing and inference are pure; storage uses a temp file."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import (MemoryStore, handle_memory_turn, infer_memories,
                    parse_memory_command)
from personality import BASELINE, build_personality


class TestParsing(unittest.TestCase):
    def test_remember_with_object(self):
        for phrase in ("remember that I prefer PETG Basic",
                       "Remember I prefer PETG Basic.",
                       "can you remember that I prefer PETG Basic",
                       "please remember the fact that I prefer PETG Basic"):
            self.assertEqual(parse_memory_command(phrase),
                             ("remember", "I prefer PETG Basic"), phrase)

    def test_bare_remember_targets_last_reply(self):
        for phrase in ("remember that", "Remember this.", "ok remember it"):
            self.assertEqual(parse_memory_command(phrase), ("remember_last", None), phrase)

    def test_forget(self):
        self.assertEqual(parse_memory_command("forget about the PETG thing"),
                         ("forget", "PETG thing"))
        self.assertEqual(parse_memory_command("forget that"), ("forget", None))

    def test_recall(self):
        for phrase in ("what do you remember", "what do you remember about me",
                       "what do you know about me", "list your memories"):
            self.assertEqual(parse_memory_command(phrase), ("recall", None), phrase)

    def test_ordinary_speech_is_inert(self):
        for phrase in ("how are you doing today", "I love you so much",
                       "do you remember the movie", "", "   ",
                       "what should we work on next"):
            self.assertIsNone(parse_memory_command(phrase), phrase)

    def test_transcribed_wake_prefix_preserves_memory_commands(self):
        cases = {
            "Hey, Tars, remember that I like my eggs scrambled":
                ("remember", "I like my eggs scrambled"),
            "hey tars what do you remember about me?": ("recall", None),
            "Hey Tarts, forget that I like my eggs scrambled":
                ("forget", "I like my eggs scrambled"),
            "TARS, remember that": ("remember_last", None),
        }
        for phrase, expected in cases.items():
            self.assertEqual(parse_memory_command(phrase), expected, phrase)


class TestInference(unittest.TestCase):
    def test_preferences(self):
        found = infer_memories("I really like the black and gold look.")
        self.assertEqual([(f["kind"], f["text"]) for f in found],
                         [("preference", "likes the black and gold look")])

    def test_dislike_and_fact(self):
        found = infer_memories("I hate the gaps. I use a Raspberry Pi 5.")
        self.assertEqual({f["text"] for f in found},
                         {"dislikes the gaps", "uses a Raspberry Pi 5"})

    def test_questions_are_not_inferred(self):
        self.assertEqual(infer_memories("Do I like this voice?"), [])
        self.assertEqual(infer_memories("What do I use for printing?"), [])

    def test_explicit_command_is_not_double_counted(self):
        self.assertEqual(infer_memories("remember that I like PETG"), [])

    def test_one_inference_per_sentence(self):
        found = infer_memories("I like PETG and I use a Pi.")
        self.assertEqual(len(found), 1)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.path = Path(self.dir.name) / "memory.json"
        self.store = MemoryStore(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_add_and_persist_across_restart(self):
        self.store.add("prefers PETG Basic", source="stated")
        reopened = MemoryStore(self.path)
        self.assertEqual(len(reopened.all()), 1)
        self.assertEqual(reopened.all()[0]["text"], "prefers PETG Basic")

    def test_inferred_is_upgraded_by_explicit_statement(self):
        self.store.add("likes black and gold", source="inferred")
        self.store.add("likes black and gold", source="stated")
        self.assertEqual(len(self.store.all()), 1)
        self.assertEqual(self.store.all()[0]["source"], "stated")

    def test_stated_is_not_downgraded(self):
        self.store.add("likes black and gold", source="stated")
        self.store.add("likes black and gold", source="inferred")
        self.assertEqual(self.store.all()[0]["source"], "stated")

    def test_forget_by_fuzzy_match(self):
        self.store.add("prefers PETG Basic over HF", source="stated")
        removed = self.store.forget("the PETG thing")
        self.assertIsNotNone(removed)
        self.assertEqual(self.store.all(), [])

    def test_forget_missing_returns_none(self):
        self.assertIsNone(self.store.forget("something he never said"))

    def test_prompt_block_labels_and_ordering(self):
        self.store.add("guessed thing", source="inferred")
        self.store.add("confirmed thing", source="stated")
        self.store.add("a joke line", kind="bit", source="stated")
        block = self.store.for_prompt()
        self.assertIn("[CONFIRMED] confirmed thing", block)
        self.assertIn("[UNCONFIRMED] guessed thing", block)
        self.assertIn("[RUNNING JOKE] a joke line", block)
        self.assertLess(block.index("CONFIRMED] confirmed"), block.index("UNCONFIRMED"))
        self.assertLess(block.index("UNCONFIRMED"), block.index("RUNNING JOKE"))

    def test_empty_store_injects_nothing(self):
        self.assertIsNone(self.store.for_prompt())

    def test_prompt_block_is_bounded(self):
        for i in range(60):
            self.store.add(f"fact number {i} " + "x" * 60, source="stated")
        block = self.store.for_prompt()
        self.assertLessEqual(len(block), 1400)

    def test_corrupt_file_is_quarantined_not_lost(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        store = MemoryStore(self.path)
        self.assertEqual(store.all(), [])
        self.assertTrue(self.path.with_suffix(".corrupt.json").exists())

    def test_file_is_human_readable(self):
        self.store.add("prefers PETG Basic", source="stated")
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(set(loaded[0]), {"id", "text", "kind", "source", "created", "updated"})


class TestTurnHandling(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.store = MemoryStore(Path(self.dir.name) / "memory.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_explicit_remember_saves_as_stated(self):
        note = handle_memory_turn(self.store, "remember that I prefer PETG Basic")
        self.assertIn("remember", note.lower())
        self.assertEqual(self.store.all()[0]["source"], "stated")

    def test_bare_remember_saves_tars_last_line_as_bit(self):
        note = handle_memory_turn(self.store, "remember that",
                                  last_reply="I'm a robot, Luca. No commute.")
        self.assertIn("running joke", note.lower())
        self.assertEqual(self.store.all()[0]["kind"], "bit")

    def test_bare_remember_without_a_reply_is_handled(self):
        note = handle_memory_turn(self.store, "remember that", last_reply=None)
        self.assertIn("isn't one yet", note)
        self.assertEqual(self.store.all(), [])

    def test_inference_is_silent(self):
        note = handle_memory_turn(self.store, "I really like the black and gold look.")
        self.assertIsNone(note)
        self.assertEqual(self.store.all()[0]["source"], "inferred")

    def test_forget_unknown_is_honest(self):
        note = handle_memory_turn(self.store, "forget about my birthday")
        self.assertIn("do not have", note)

    def test_recall_lists_and_flags_guesses(self):
        self.store.add("prefers PETG Basic", source="stated")
        self.store.add("likes black and gold", source="inferred")
        note = handle_memory_turn(self.store, "what do you remember about me")
        self.assertIn("PETG Basic", note)
        self.assertIn("not confirmed", note)

    def test_ordinary_conversation_changes_nothing(self):
        self.assertIsNone(handle_memory_turn(self.store, "how are you doing today"))
        self.assertEqual(self.store.all(), [])


class TestPromptIntegration(unittest.TestCase):
    def test_memories_absent_keeps_prompt_unchanged(self):
        self.assertEqual(build_personality(BASELINE.copy()),
                         build_personality(BASELINE.copy(), memories=None))

    def test_memories_present_adds_labelled_block(self):
        prompt = build_personality(BASELINE.copy(), memories="- [CONFIRMED] prefers PETG")
        self.assertIn("WHAT YOU REMEMBER ABOUT LUCA", prompt)
        self.assertIn("[CONFIRMED] prefers PETG", prompt)
        self.assertIn("UNCONFIRMED", prompt)

    def test_memory_and_control_event_coexist(self):
        prompt = build_personality(BASELINE.copy(), note="Luca reset the dials.",
                                   memories="- [CONFIRMED] prefers PETG")
        self.assertIn("WHAT YOU REMEMBER", prompt)
        self.assertIn("CONTROL EVENT", prompt)
        self.assertLess(prompt.index("WHAT YOU REMEMBER"), prompt.index("CONTROL EVENT"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
