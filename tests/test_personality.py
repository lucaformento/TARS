"""Personality command parsing tests; no API calls."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personality import BASELINE, apply_command, build_personality


class PersonalityTests(unittest.TestCase):
    def test_ordinary_settings_sentence_is_not_a_command(self):
        settings = BASELINE.copy()
        note = apply_command(
            "I just want to make sure you have the right settings and are doing okay.",
            settings,
        )
        self.assertIsNone(note)
        self.assertEqual(settings, BASELINE)

    def test_unrelated_reset_sentence_is_not_a_command(self):
        settings = BASELINE.copy()
        self.assertIsNone(apply_command("I need to reset the router later", settings))
        self.assertEqual(settings, BASELINE)

    def test_explicit_settings_question_generates_control_event(self):
        settings = BASELINE.copy()
        note = apply_command("What are your settings?", settings)
        self.assertIn("explicitly asked", note)
        self.assertIn("Humor 75", note)

    def test_explicit_reset_and_preset_commands(self):
        settings = {key: 1 for key in BASELINE}
        self.assertIsNotNone(apply_command("reset your personality", settings))
        self.assertEqual(settings, BASELINE)
        self.assertIsNotNone(apply_command("switch to buddy mode", settings))
        self.assertEqual(settings["humor"], 90)

    def test_dial_commands_support_digits_and_spoken_numbers(self):
        settings = BASELINE.copy()
        note = apply_command(
            "Set humor to 82 and adjust sarcasm to twenty five.", settings
        )
        self.assertIn("humor to 82", note)
        self.assertIn("sarcasm to 25", note)
        self.assertEqual(settings["humor"], 82)
        self.assertEqual(settings["sarcasm"], 25)

    def test_unrelated_number_does_not_change_a_dial(self):
        settings = BASELINE.copy()
        self.assertIsNone(apply_command("Humor me: I found 2 bugs", settings))
        self.assertEqual(settings, BASELINE)

    def test_voice_prompt_is_grounded_and_does_not_require_name_repetition(self):
        prompt = build_personality(BASELINE, voice=True)
        self.assertIn("Do not invent sensor readings", prompt)
        self.assertIn("not as a requirement in every reply", prompt)
        self.assertIn("15-35 words", prompt)
        self.assertNotIn("make the differences DRAMATIC", prompt)


if __name__ == "__main__":
    unittest.main()
