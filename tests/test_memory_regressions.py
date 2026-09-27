"""Regression cases from the memory candidate review; all stores are temporary."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory import (AmbiguousMemory, MAX_PROMPT_CHARS, MemoryStore,
                    handle_memory_turn, infer_memories)


class MemoryRegressionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "memory.json"
        self.store = MemoryStore(self.path)

    def test_quiz_paraphrases_merge_in_both_orders_and_keep_explicit_wording(self):
        phrases = ["remember that I prefer PETG Basic", "I really like PETG Basic"]
        for order in (phrases, list(reversed(phrases))):
            with self.subTest(order=order):
                store = MemoryStore(self.path.with_name(f"order-{order[0][:1]}.json"))
                for phrase in order:
                    handle_memory_turn(store, phrase)
                reopened = MemoryStore(store.path)
                self.assertEqual(len(reopened.all()), 1)
                entry = reopened.all()[0]
                self.assertEqual(entry["text"], "I prefer PETG Basic")
                self.assertEqual(entry["source"], "stated")
                self.assertEqual(entry["kind"], "preference")

    def test_qualifiers_objects_and_opposite_preferences_stay_distinct(self):
        for phrase in ("I prefer PETG Basic", "I prefer PETG HF", "I dislike PETG Basic",
                       "I prefer PETG Basic over HF", "I prefer HF over PETG Basic"):
            self.store.add(phrase)
        self.assertEqual(len(self.store.all()), 5)

    def test_bits_cannot_upgrade_or_merge_with_facts_in_either_order(self):
        for reverse in (False, True):
            store = MemoryStore(self.path.with_name(f"bit-{reverse}.json"))
            entries = [("fact", "inferred"), ("bit", "stated")]
            for kind, source in reversed(entries) if reverse else entries:
                store.add("I am made of gold", kind=kind, source=source)
            self.assertEqual({(e["kind"], e["source"]) for e in store.all()},
                             {("fact", "inferred"), ("bit", "stated")})
            self.assertIn("[RUNNING JOKE]", store.for_prompt())
            self.assertNotIn("[CONFIRMED]", store.for_prompt())

    def test_forget_cannot_delete_another_object_using_a_shared_verb(self):
        handle_memory_turn(self.store, "remember that I prefer PLA")
        note = handle_memory_turn(self.store, "forget that I prefer PETG")
        self.assertIn("do not have", note)
        self.assertEqual(self.store.all()[0]["text"], "I prefer PLA")

    def test_ambiguous_reference_changes_neither_memory_nor_disk(self):
        self.store.add("I prefer PETG Basic")
        self.store.add("I dislike PETG HF")
        before = self.path.read_bytes()
        note = handle_memory_turn(self.store, "forget about the PETG thing")
        self.assertIn("Nothing was deleted", note)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(self.store.all()), 2)
        handle_memory_turn(self.store, "forget that I prefer PETG Basic")
        self.assertEqual(self.store.all()[0]["text"], "I dislike PETG HF")

    def test_forget_requires_whole_words_and_all_reference_words(self):
        self.store.add("uses plastic sheets")
        self.store.add("uses a PETG spool")
        self.assertIsNone(self.store.forget("PLA"))
        self.assertIsNone(self.store.forget("PETG printer"))
        self.assertEqual(self.store.forget("PETG spool")["text"], "uses a PETG spool")

    def test_fact_and_bit_with_identical_text_are_ambiguous_for_forget(self):
        self.store.add("the gold robot", kind="fact")
        self.store.add("the gold robot", kind="bit")
        with self.assertRaises(AmbiguousMemory):
            self.store.forget("the gold robot")
        self.assertEqual(len(self.store.all()), 2)

    def test_metadata_missing_or_wrong_type_is_repaired_without_promoting_source(self):
        self.path.write_text(json.dumps([{"text": "likes coffee", "kind": "preference",
                                        "source": "inferred", "updated": [], "id": 4}]))
        store = MemoryStore(self.path)
        entry = store.all()[0]
        self.assertEqual(entry["source"], "inferred")
        self.assertTrue(entry["created"].startswith("1970-01-01"))
        self.assertIsInstance(entry["id"], str)
        self.assertIn("[UNCONFIRMED] likes coffee", store.for_prompt())
        self.assertIn("not confirmed", store.spoken_summary())
        store.forget_last()
        self.assertEqual(MemoryStore(self.path).all(), [])

    def test_old_duplicate_preferences_merge_on_load_without_touching_disk(self):
        raw = [{"text": "likes PETG Basic", "kind": "preference", "source": "inferred"},
               {"text": "I prefer PETG Basic", "kind": "fact", "source": "stated"}]
        self.path.write_text(json.dumps(raw))
        before = self.path.read_bytes()
        store = MemoryStore(self.path)
        self.assertEqual(len(store.all()), 1)
        self.assertEqual(store.all()[0]["text"], "I prefer PETG Basic")
        self.assertEqual(store.all()[0]["source"], "stated")
        self.assertEqual(self.path.read_bytes(), before)

    def test_bad_schema_and_invalid_encoding_are_preserved_without_overwriting_backups(self):
        previous = self.path.with_suffix(".corrupt.json")
        previous.write_text("older corrupt file")
        for contents in (b'{"not": "an array"}', b'\xff\xfe', b'[{"text": "missing source"}]'):
            self.path.write_bytes(contents)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                store = MemoryStore(self.path)
            self.assertEqual(store.all(), [])
            self.assertFalse(self.path.exists())
            self.assertIsNone(store.load_error)
        self.assertEqual(previous.read_text(), "older corrupt file")
        self.assertEqual(len(list(self.path.parent.glob("memory.corrupt*.json"))), 4)

    def test_unreadable_file_is_not_renamed_or_overwritten(self):
        self.path.write_text("[]")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                store = MemoryStore(self.path)
        with self.assertRaises(OSError):
            store.add("likes coffee")
        self.assertEqual(self.path.read_text(), "[]")
        self.assertEqual(store.all(), [])
        self.assertFalse(self.path.with_suffix(".corrupt.json").exists())

    def test_failed_quarantine_disables_writes(self):
        self.path.write_text("invalid")
        with patch.object(Path, "replace", side_effect=PermissionError("denied")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                store = MemoryStore(self.path)
        with self.assertRaises(OSError):
            store.add("likes coffee")
        self.assertEqual(self.path.read_text(), "invalid")

    def test_write_failure_leaves_add_upgrade_and_forget_unchanged(self):
        self.store.add("likes coffee", kind="preference", source="inferred")
        before = self.path.read_bytes()
        original = self.store.all()
        with patch("memory.os.replace", side_effect=OSError("disk full")):
            for action in (lambda: self.store.add("new fact"),
                           lambda: self.store.add("I prefer coffee"),
                           lambda: self.store.forget("coffee"), self.store.forget_last):
                with self.assertRaises(OSError):
                    action()
                self.assertEqual(self.store.all(), original)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(list(self.path.parent.glob("memory.json.*.tmp")), [])

    def test_retrieval_includes_newlines_in_character_budget(self):
        for index in range(100):
            self.store.add(f"preference {index} " + "x" * 50)
        self.assertLessEqual(len(self.store.for_prompt()), MAX_PROMPT_CHARS)

    def test_inference_skips_questions_quotes_and_reported_other_people(self):
        for phrase in ("I like PETG?", '"I like PETG," said my friend.',
                       "My friend said I like PETG", "If I like PETG, what next?"):
            self.assertEqual(infer_memories(phrase), [], phrase)

    def test_transcribed_wake_prefix_does_not_block_inference(self):
        self.assertEqual(
            infer_memories("Hey, Tars, I really like scrambled eggs."),
            [{"text": "likes scrambled eggs", "kind": "preference"}],
        )

    def test_natural_first_person_lead_in_is_inferred_but_reports_stay_excluded(self):
        self.assertEqual(
            infer_memories("Just want to let you know that I like my mac and cheese really cheesy"),
            [{"text": "likes my mac and cheese really cheesy", "kind": "preference"}],
        )
        self.assertEqual(
            infer_memories("I just wanted to tell you that I prefer PETG Basic"),
            [{"text": "likes PETG Basic", "kind": "preference"}],
        )
        self.assertEqual(infer_memories("My friend wanted to tell you that I like PETG"), [])

    def test_oversized_fact_is_not_silently_truncated_and_confirmed(self):
        note = handle_memory_turn(self.store, "remember that " + "x" * 241)
        self.assertIn("No memory was saved", note)
        self.assertEqual(self.store.all(), [])


if __name__ == "__main__":
    unittest.main()
