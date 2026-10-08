"""Personality command parsing tests; no API calls."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personality import (BASELINE, NEAR_MISS_NOTE, PRESETS, apply_command, build_personality,
                         describe_mode)


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


def run(text, settings=None):
    settings = dict(BASELINE if settings is None else settings)
    return apply_command(text, settings), settings


class ForgivingCommandTests(unittest.TestCase):
    """Wordings from Luca's October 8 demo and how people naturally say them."""

    def test_natural_absolute_wordings_set_the_dial(self):
        for said in ("Change your humour to 90.", "Hey TARS, change your humor to 90.",
                     "Change your humor setting to 90.", "Make your humor 90.",
                     "Can you change your humor to a 90?", "Humor 90.", "Humor 90%.",
                     "Turn your humor up to ninety.", "Set humor to ninety percent."):
            with self.subTest(said=said):
                note, settings = run(said)
                self.assertEqual(settings["humor"], 90)
                self.assertIn("humor to 90", note)

    def test_intelligence_is_an_alias_for_intellect(self):
        _, settings = run("Set your intelligence to 80")
        self.assertEqual(settings["intellect"], 80)

    def test_relative_wordings_move_one_dial_by_ten(self):
        cases = {
            "Be funnier.": ("humor", 85), "Turn the humor up.": ("humor", 85),
            "More humor please": ("humor", 85), "Make yourself funnier": ("humor", 85),
            "Can you be a little less sarcastic?": ("sarcasm", 50),
            "Tone down the sarcasm": ("sarcasm", 50), "Less sarcasm.": ("sarcasm", 50),
            "Turn your sarcasm down a bit": ("sarcasm", 50),
            "TARS, be more honest.": ("honesty", 100), "Be smarter": ("intellect", 60),
            "Be dumber": ("intellect", 40), "Be funner.": ("humor", 85),
        }
        for said, (dial, value) in cases.items():
            with self.subTest(said=said):
                note, settings = run(said)
                self.assertEqual(settings[dial], value)
                self.assertEqual({k: v for k, v in settings.items() if k != dial},
                                 {k: v for k, v in BASELINE.items() if k != dial})
                self.assertIn(f"to {value}", note)

    def test_relative_change_stops_at_the_limits_and_says_so(self):
        note, settings = run("Be more honest.", {**BASELINE, "honesty": 100})
        self.assertEqual(settings["honesty"], 100)
        self.assertIn("already at its maximum", note)
        _, settings = run("Less humor", {**BASELINE, "humor": 5})
        self.assertEqual(settings["humor"], 0)

    def test_mode_wordings_switch_presets(self):
        for said in ("Buddy mode.", "Hey TARS, switch to buddy mode.",
                     "Switch to buddy mode please.", "Turn on buddy mode.",
                     "Activate buddy mode.", "Go into buddy mode, TARS.",
                     "Put yourself in buddy mode", "Turn buddy mode on",
                     "TARS switched to buddy mode."):
            with self.subTest(said=said):
                note, settings = run(said)
                self.assertEqual(settings, PRESETS["buddy mode"])
                self.assertIn("buddy mode", note)
        for said in ("Switch to know-it-all mode.", "Know it all mode.", "know-it-all"):
            with self.subTest(said=said):
                _, settings = run(said)
                self.assertEqual(settings, PRESETS["know-it-all"])

    def test_turning_a_mode_off_or_going_back_to_normal_resets(self):
        for said in ("Turn off buddy mode.", "Exit buddy mode", "Back to normal.",
                     "Reset.", "Reset your settings please"):
            with self.subTest(said=said):
                _, settings = run(said, PRESETS["buddy mode"])
                self.assertEqual(settings, BASELINE)

    def test_mode_and_dial_questions_state_the_current_settings(self):
        for said in ("What mode are you in?", "Hey TARS, what are your settings?",
                     "What's your humor set to?"):
            with self.subTest(said=said):
                note, _ = run(said, PRESETS["buddy mode"])
                self.assertIn("mode buddy mode", note)
                self.assertIn("Humor 90", note)

    def test_ordinary_conversation_stays_inert(self):
        for said in ("Be honest, do you like it?", "Be honest.", "Thanks, buddy.", "Buddy.",
                     "I'm printing your body next month.", "You should get more sleep.",
                     "What is the sarcasm in that movie about?",
                     "I love your personality, don't change.", "Let's change the subject.",
                     "Can you turn up the music?", "Earlier you were funnier."):
            with self.subTest(said=said):
                note, settings = run(said)
                self.assertIsNone(note)
                self.assertEqual(settings, BASELINE)

    def test_unrecognized_settings_request_is_honest_and_changes_nothing(self):
        for said in ("Can you change your humor?", "Turn the humor way way up",
                     "Could you maybe go into buddy mode", "tome down the sarcasm.",
                     "TARS, change your humor tonight."):
            with self.subTest(said=said):
                note, settings = run(said)
                self.assertEqual(note, NEAR_MISS_NOTE)
                self.assertEqual(settings, BASELINE)
        self.assertIn("nothing changed", NEAR_MISS_NOTE)
        self.assertIn("Never mention an app", NEAR_MISS_NOTE)


class ModeTests(unittest.TestCase):
    def test_mode_is_named_from_the_dials(self):
        self.assertEqual(describe_mode(BASELINE), "baseline")
        self.assertEqual(describe_mode(PRESETS["buddy mode"]), "buddy mode")
        self.assertEqual(describe_mode(PRESETS["know-it-all"]), "know-it-all")
        self.assertEqual(describe_mode({**BASELINE, "humor": 90}), "custom")

    def test_prompt_names_the_mode_lists_the_modes_and_denies_an_app(self):
        prompt = build_personality(PRESETS["buddy mode"])
        self.assertIn("CURRENT MODE: buddy mode", prompt)
        self.assertIn("- know-it-all: Humor 20, Sarcasm 30, Honesty 100, Intellect 95.", prompt)
        self.assertIn("There is\nno app, menu, or settings screen.", prompt)
        self.assertIn("Never claim a dial or mode changed", prompt)

    def test_voice_prompt_explains_name_aware_listening(self):
        prompt = build_personality(BASELINE, voice=True)
        self.assertIn("only sentences that include your name", prompt)
        self.assertIn('"Go to sleep"', prompt)


if __name__ == "__main__":
    unittest.main()
