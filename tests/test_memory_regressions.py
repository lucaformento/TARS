"""Regression cases from the memory candidate review; all stores are temporary."""

import json
import importlib.util
from datetime import datetime as real_datetime
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch
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
        self.assertIn("Nothing was deleted", note)
        self.assertIn("did not match a stored memory", note)
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

    def test_remember_questions_and_reminiscences_do_not_write(self):
        self.store.add("existing fact", source="stated")
        before_entries = self.store.all()
        before_disk = self.path.read_bytes()
        for phrase in ("remember when we fixed the servo?",
                       "Hey TARS, remember the time we fixed the servo",
                       "remember that I prefer PETG?"):
            with self.subTest(phrase=phrase):
                note = handle_memory_turn(self.store, phrase)
                self.assertIn("Nothing was saved", note)
                self.assertIn("Do not claim you remembered it", note)
                self.assertEqual(self.store.all(), before_entries)
                self.assertEqual(self.path.read_bytes(), before_disk)

    def test_real_remember_statements_still_save_as_confirmed(self):
        for index, phrase in enumerate(("remember that I prefer PETG",
                                        "Hey TARS, remember I work nights")):
            with self.subTest(phrase=phrase):
                store = MemoryStore(self.path.with_name(f"real-save-{index}.json"))
                note = handle_memory_turn(store, phrase)
                self.assertIn("remember", note.lower())
                self.assertEqual(len(store.all()), 1)
                self.assertEqual(store.all()[0]["source"], "stated")

    def test_forget_it_variants_do_not_delete_or_write(self):
        self.store.add("I prefer PETG", source="stated")
        before_entries = self.store.all()
        before_disk = self.path.read_bytes()
        for phrase in ("forget it", "okay, forget it", "never mind, forget it"):
            with self.subTest(phrase=phrase):
                note = handle_memory_turn(self.store, phrase)
                self.assertIn("Nothing was saved or deleted", note)
                self.assertIn("Do not claim", note)
                self.assertEqual(self.store.all(), before_entries)
                self.assertEqual(self.path.read_bytes(), before_disk)

    def test_previous_run_entry_is_never_deleted_by_bare_forget(self):
        self.store.add("I work nights", source="stated")
        reopened = MemoryStore(self.path)
        before = self.path.read_bytes()
        handle_memory_turn(reopened, "forget that")
        self.assertEqual(reopened.all()[0]["text"], "I work nights")
        self.assertEqual(self.path.read_bytes(), before)

    def test_current_run_entry_is_deleted_and_quoted(self):
        for index, phrase in enumerate(("forget that", "forget this")):
            with self.subTest(phrase=phrase):
                store = MemoryStore(self.path.with_name(f"current-run-{index}.json"))
                store.add("I prefer PLA", source="stated")
                store.add("I prefer PETG", source="stated")
                note = handle_memory_turn(store, phrase)
                self.assertEqual([entry["text"] for entry in store.all()],
                                 ["I prefer PLA"])
                self.assertIn('"I prefer PETG"', note)

    def test_no_current_run_save_names_latest_entry_date_and_command(self):
        self.path.write_text(json.dumps([
            {"id": "m_old", "text": "I prefer PLA", "kind": "preference",
             "source": "stated", "created": "2026-09-20T12:00:00+00:00",
             "updated": "2026-09-20T12:00:00+00:00"},
            {"id": "m_latest", "text": "I work nights", "kind": "fact",
             "source": "stated", "created": "2026-09-29T12:00:00+00:00",
             "updated": "2026-09-29T12:00:00+00:00"},
        ]))
        store = MemoryStore(self.path)
        before = self.path.read_bytes()
        note = handle_memory_turn(store, "forget that")
        self.assertIn("I work nights", note)
        self.assertIn("September 29, 2026", note)
        self.assertIn('"forget that I work nights"', note)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(store.all()), 2)

    def test_latest_memory_date_uses_pi_local_time(self):
        self.path.write_text(json.dumps([
            {"id": "m_latest", "text": "I work nights", "kind": "fact",
             "source": "stated", "created": "2026-09-29T01:00:00+00:00",
             "updated": "2026-09-29T01:00:00+00:00"},
        ]))
        store = MemoryStore(self.path)
        parsed_time = Mock()
        parsed_time.astimezone.return_value = real_datetime(2026, 9, 28, 21, 0)
        with patch("memory.datetime") as mocked_datetime:
            mocked_datetime.fromisoformat.return_value = parsed_time
            note = handle_memory_turn(store, "forget that")
        parsed_time.astimezone.assert_called_once_with()
        self.assertIn("September 28, 2026", note)
        self.assertEqual(len(store.all()), 1)

    def test_upgrade_this_run_counts_as_saved_this_run(self):
        self.store.add("likes PETG", kind="preference", source="inferred")
        reopened = MemoryStore(self.path)
        handle_memory_turn(reopened, "remember that I prefer PETG")
        self.assertEqual(reopened.all()[0]["source"], "stated")
        note = handle_memory_turn(reopened, "forget this")
        self.assertEqual(reopened.all(), [])
        self.assertIn('"I prefer PETG"', note)

    def test_live_turn_four_fillers_delete_only_latest_this_run_and_quote_it(self):
        self.store.add("I work nights", source="stated")
        current_run = MemoryStore(self.path)
        current_run.add("my test word is pineapple", source="stated")
        note = handle_memory_turn(current_run, "Yeah, just forget that")
        self.assertEqual([entry["text"] for entry in current_run.all()], ["I work nights"])
        self.assertIn('"my test word is pineapple"', note)
        self.assertIn("was deleted", note)
        self.assertEqual(MemoryStore(self.path).all(), current_run.all())

    def test_live_turn_four_fillers_never_delete_a_previous_run_entry(self):
        self.store.add("I work nights", source="stated")
        reopened = MemoryStore(self.path)
        before = self.path.read_bytes()
        note = handle_memory_turn(reopened, "Yeah, just forget that")
        self.assertIn("Nothing was deleted", note)
        self.assertIn('"forget that I work nights"', note)
        self.assertEqual(len(reopened.all()), 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_live_turn_five_forget_it_is_honest_even_after_a_real_deletion(self):
        self.store.add("my test word is pineapple", source="stated")
        handle_memory_turn(self.store, "Yeah, just forget that")
        before = self.path.read_bytes()
        note = handle_memory_turn(self.store, "Yeah, just forget it")
        self.assertIn("Nothing was saved or deleted this turn", note)
        self.assertIn('Do not claim "got it", "forgotten", "already done"', note)
        self.assertEqual(self.store.all(), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_forget_them_mishears_never_delete_even_an_exact_matching_entry(self):
        self.store.add("my test word is pineapple", source="stated")
        # The old broad parser could even remove this literal target. Neither
        # silently correcting "them" nor treating it as part of a target is safe.
        self.store.add("them my test word is pineapple", source="stated")
        before_entries = self.store.all()
        before = self.path.read_bytes()
        for phrase in ("forget them my test word is pineapple",
                       "Yeah, just forget them my test word is pineapple",
                       "forget them", "forget them all"):
            with self.subTest(phrase=phrase):
                note = handle_memory_turn(self.store, phrase)
                self.assertIn("Nothing was saved or deleted this turn", note)
                self.assertIn('"forget them" is not a supported deletion command', note)
                self.assertIn('"forget that <exact memory text>"', note)
                self.assertEqual(self.store.all(), before_entries)
                self.assertEqual(self.path.read_bytes(), before)

    def test_parsed_unmatched_target_has_honest_retry_note_and_does_not_write(self):
        self.store.add("my test word is pineapple", source="stated")
        before = self.path.read_bytes()
        note = handle_memory_turn(self.store, "forget that my test word is papaya")
        self.assertIn("Nothing was deleted", note)
        self.assertIn("did not match a stored memory", note)
        self.assertIn("Do not claim", note)
        self.assertIn("previously removed", note)
        self.assertIn('"forget that <exact memory text>"', note)
        self.assertEqual(self.path.read_bytes(), before)

    def test_guarded_requests_skip_all_inference_in_the_same_turn(self):
        self.store.add("I work nights", source="stated")
        before = self.path.read_bytes()
        for phrase in ("I like PETG. Could you remember that?",
                       "Yeah, just forget it. I prefer PLA",
                       "I prefer PLA. Forget them my test word is pineapple"):
            with self.subTest(phrase=phrase):
                with patch("memory.infer_memories", side_effect=AssertionError("must not infer")):
                    note = handle_memory_turn(self.store, phrase)
                self.assertIn("Nothing was saved or deleted this turn", note)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(len(self.store.all()), 1)

    def test_guarded_requests_without_a_store_file_do_not_create_one(self):
        for phrase in ("Yeah, just forget it", "do you remember the movie",
                       "forget them my test word is pineapple"):
            self.assertIn("Nothing was saved or deleted", handle_memory_turn(self.store, phrase))
            self.assertFalse(self.path.exists())

    def test_unparsed_remember_request_gives_statement_retry_without_saving(self):
        note = handle_memory_turn(self.store, "I'd like you to remember I prefer PETG")
        self.assertIn("Nothing was saved or deleted", note)
        self.assertIn('"remember that <fact>" as a statement', note)
        self.assertFalse(self.path.exists())
        note = handle_memory_turn(self.store, "Yeah, just remember that I prefer PETG")
        self.assertIn("Luca asked you to remember", note)
        self.assertEqual(self.store.all()[0]["text"], "I prefer PETG")
        self.assertEqual(self.store.all()[0]["source"], "stated")

    def test_filler_questions_and_reminiscences_stay_unsaved_with_retry_note(self):
        for phrase in ("Yeah, just remember that I prefer PETG?",
                       "Actually, remember when we fixed the servo",
                       "Um, remember the time we fixed the servo"):
            with self.subTest(phrase=phrase):
                note = handle_memory_turn(self.store, phrase)
                self.assertIn("Nothing was saved", note)
                self.assertIn("Do not claim you remembered it", note)
                self.assertIn('"remember that <fact>"', note)
                self.assertFalse(self.path.exists())

    def test_fillers_do_not_delete_for_quoted_reported_or_conditional_requests(self):
        self.store.add("my test word is pineapple", source="stated")
        before = self.path.read_bytes()
        for phrase in ('"Yeah, just forget that"', 'Yeah, my friend said "forget that"',
                       "Actually, if I say forget that, will you delete it",
                       "Yes, I might forget that", "I said yeah just forget that"):
            with self.subTest(phrase=phrase):
                self.assertIn("Nothing was saved or deleted", handle_memory_turn(self.store, phrase))
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(len(self.store.all()), 1)

    def test_memory_word_boundaries_preserve_ordinary_inference(self):
        self.assertIsNone(handle_memory_turn(self.store, "I like forgetful robots"))
        self.assertEqual(self.store.all()[0]["text"], "likes forgetful robots")
        self.assertEqual(self.store.all()[0]["source"], "inferred")

    def test_filler_specific_forget_and_inferred_upgrade_still_work(self):
        self.store.add("likes PETG", kind="preference", source="inferred")
        reopened = MemoryStore(self.path)
        handle_memory_turn(reopened, "Yes, actually remember that I prefer PETG")
        note = handle_memory_turn(reopened, "So, just forget this")
        self.assertIn('"I prefer PETG"', note)
        self.assertEqual(reopened.all(), [])
        handle_memory_turn(reopened, "Um, remember that I work nights")
        note = handle_memory_turn(reopened, "Actually, forget that I work nights")
        self.assertIn("It is deleted", note)
        self.assertEqual(reopened.all(), [])

    def test_honesty_notes_reach_real_brain_request_without_paid_calls(self):
        # Load a private module instance with fake SDK/dotenv so this test
        # neither reads .env nor interferes with tests/test_brain.py's module.
        anthropic = ModuleType("anthropic")
        anthropic.Anthropic = MagicMock()
        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = MagicMock()
        spec = importlib.util.spec_from_file_location(
            "honesty_test_brain", Path(__file__).resolve().parents[1] / "brain.py")
        brain = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"anthropic": anthropic, "dotenv": dotenv}):
            spec.loader.exec_module(brain)
        self.store.add("my test word is pineapple", source="stated")
        before = self.path.read_bytes()
        for phrase in ("Yeah, just forget it", "forget them my test word is pineapple",
                       "forget that my test word is papaya"):
            with self.subTest(phrase=phrase):
                stream = MagicMock()
                stream.__enter__.return_value.text_stream = iter(["Please rephrase."])
                messages = SimpleNamespace(stream=MagicMock(return_value=stream))
                tars = brain.TARS(client=SimpleNamespace(messages=messages), memory=self.store)
                tars.respond(phrase)
                prompt = messages.stream.call_args.kwargs["system"]
                self.assertIn("CONTROL EVENT", prompt)
                self.assertIn("deleted", prompt)
                self.assertIn("Do not claim", prompt)
                self.assertIn('"forget that <exact memory text>"', prompt)
                self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
