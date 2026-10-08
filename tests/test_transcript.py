"""Speech-recognition text handling tests; no audio or model."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcript import (STT_HOTWORDS, clean_transcript, is_bare_address, is_sleep_command,
                        mentions_tars)


class TranscriptTests(unittest.TestCase):
    def test_name_is_always_written_in_capitals(self):
        for said in ("Hey, Tars, what's up?", "What are your settings, Tarz?",
                     "tars switch to buddy mode", "TARS."):
            with self.subTest(said=said):
                self.assertIn("TARS", clean_transcript(said))
                self.assertNotRegex(clean_transcript(said), r"\b(?:Tars|Tarz|tars)\b")

    def test_similar_words_are_left_alone(self):
        for said in ("Look at the stars tonight.", "The Tarzan movie was fun.",
                     "My sign is Taurus.", "I spilled tar on the driveway."):
            with self.subTest(said=said):
                self.assertEqual(clean_transcript(said), said)

    def test_body_mode_becomes_buddy_mode_but_your_body_stays(self):
        self.assertEqual(clean_transcript("Switch to body mode."), "Switch to buddy mode.")
        self.assertEqual(clean_transcript("Body Mode, please."), "buddy Mode, please.")
        self.assertEqual(clean_transcript("I'm printing your body next month."),
                         "I'm printing your body next month.")
        self.assertEqual(clean_transcript("The body is almost done, mode or not."),
                         "The body is almost done, mode or not.")

    def test_name_mentions_anywhere_in_the_sentence(self):
        self.assertTrue(mentions_tars("What do you think, TARS?"))
        self.assertTrue(mentions_tars("tarz, help me out"))
        self.assertFalse(mentions_tars("We should look at the stars."))
        self.assertFalse(mentions_tars(""))

    def test_bare_address_is_only_the_name(self):
        for said in ("Hey TARS.", "TARS?", "Okay, Tars.", "hey hey tars"):
            with self.subTest(said=said):
                self.assertTrue(is_bare_address(said))
        for said in ("Hey TARS, what's up?", "TARS, go to sleep.", "Thanks."):
            with self.subTest(said=said):
                self.assertFalse(is_bare_address(said))

    def test_go_to_sleep_must_be_the_whole_request(self):
        for said in ("Go to sleep.", "TARS, go to sleep.", "Okay TARS, go back to sleep now.",
                     "You can go to sleep, Tars.", "Go to sleep for now.",
                     "TARS go up to sleep."):
            with self.subTest(said=said):
                self.assertTrue(is_sleep_command(said))
        for said in ("I need to go to sleep.", "When do you go to sleep?",
                     "Don't go to sleep.", "Go to sleep and dream of servos."):
            with self.subTest(said=said):
                self.assertFalse(is_sleep_command(said))

    def test_hotwords_name_tars_and_both_modes(self):
        for word in ("TARS", "buddy mode", "know-it-all"):
            self.assertIn(word, STT_HOTWORDS)


if __name__ == "__main__":
    unittest.main()
