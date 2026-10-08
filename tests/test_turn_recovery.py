"""Voice loop behavior: failed turns return TARS to standby, and the name-aware
conversation flow. No microphone, audio, or paid calls."""

from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

numpy_stub = MagicMock()
numpy_stub.float32 = "float32"
with patch.dict(sys.modules, {"numpy": numpy_stub, "sounddevice": MagicMock()}):
    sys.modules.pop("tars_voice", None)
    voice_frontend = importlib.import_module("tars_voice")


class APIStatusError(Exception):
    """Stands in for an Anthropic overload or rate-limit error."""


class PortAudioError(Exception):
    """Stands in for a sounddevice playback failure."""


class Frame:
    def flatten(self):
        return self


class FakeBuffer:
    """Microphone buffer: one wake frame, then Ctrl+C at the next wake read."""

    fail_start = False

    def __init__(self):
        self.events = []
        self.reads = 0
        FakeBuffer.latest = self

    def callback(self, *_args):
        pass

    def attach(self, _stream):
        pass

    def read(self, _frames=voice_frontend.FRAME):
        self.reads += 1
        if self.reads > 1:
            # Reaching the wake loop again proves the turn failure was
            # contained. Stop the otherwise endless loop here.
            raise KeyboardInterrupt
        return Frame(), 0

    def stop(self, ignore_errors=False):
        self.events.append("stop")

    def start(self):
        self.events.append("start")
        if self.fail_start:
            raise OSError("device gone")


def silence():
    """Nobody started speaking before the current listening window ran out."""
    return voice_frontend.CaptureResult(
        clip=None, last_loud_read_at=None, endpoint_reason="speech_timeout",
        rms_levels=[], input_overflows=0,
    )


def noise(reason="too_short"):
    """A cough, laugh, or click: loud enough to start a capture, not speech."""
    return voice_frontend.CaptureResult(
        clip=None, last_loud_read_at=0.0, endpoint_reason=reason,
        rms_levels=[], input_overflows=0,
    )


def capture(overflows=0):
    return voice_frontend.CaptureResult(
        clip=[0.0] * 1600, last_loud_read_at=0.0, endpoint_reason="silence",
        rms_levels=[], input_overflows=overflows,
    )


def heard(*texts):
    """Whisper stand-in returning these transcripts in order."""
    replies = iter(texts)
    return lambda *_a, **_k: (iter([types.SimpleNamespace(text=next(replies))]), None)


class LoopHarness:
    def run_loop(self, captures, speak_stream=None, speak=None, transcribe=None,
                 fail_start=False, **conversation_options):
        detector = MagicMock()
        detector.predict.return_value = {"hey_tars": 0.9}
        stt = MagicMock()
        stt.transcribe.side_effect = transcribe or (
            lambda *_a, **_k: (iter([types.SimpleNamespace(text="hello")]), None)
        )
        modules = {
            "openwakeword": types.SimpleNamespace(Model=MagicMock(return_value=detector)),
            "faster_whisper": types.SimpleNamespace(WhisperModel=MagicMock(return_value=stt)),
            "brain": types.SimpleNamespace(TARS=MagicMock()),
        }
        self.stt = stt
        self.whisper = modules["faster_whisper"].WhisperModel
        self.tars = modules["brain"].TARS.return_value
        device = MagicMock()
        device.query_devices.return_value = [{"name": "USB PnP", "max_input_channels": 1}]
        self.speak_stream = MagicMock(side_effect=speak_stream)
        self.speak = MagicMock(side_effect=speak)
        output = io.StringIO()
        with patch.dict(sys.modules, modules), \
                patch.object(voice_frontend, "sd", device), \
                patch.object(voice_frontend, "BufferedInput", FakeBuffer), \
                patch.object(FakeBuffer, "fail_start", fail_start), \
                patch.object(voice_frontend, "warm_stt"), \
                patch.object(voice_frontend, "calibrate", return_value=150.0), \
                patch.object(voice_frontend, "record_utterance", side_effect=captures), \
                patch.object(voice_frontend, "speak_stream", self.speak_stream), \
                patch.object(voice_frontend, "speak", self.speak), \
                redirect_stdout(output):
            try:
                voice_frontend.run_conversation(MagicMock(), **conversation_options)
            finally:
                self.output = output.getvalue()
                self.buffer = FakeBuffer.latest

    def spoken_to_brain(self):
        return [call.args[2] for call in self.speak_stream.call_args_list]

    def lines_spoken(self):
        return [call.args[1] for call in self.speak.call_args_list]


class TurnRecoveryTests(LoopHarness, unittest.TestCase):
    def assert_slept_with_microphone_restarted(self):
        self.assertEqual(self.buffer.events, ["stop", "start"])
        self.assertEqual(self.buffer.reads, 2, "did not return to wake-word listening")
        self.assertIn("[sleep]", self.output)

    def test_anthropic_error_returns_to_sleep_instead_of_exiting(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()],
                          speak_stream=APIStatusError("Error code: 529 - overloaded"))
        self.assert_slept_with_microphone_restarted()
        self.assertIn("[turn failed] APIStatusError: Error code: 529 - overloaded",
                      self.output)

    def test_playback_device_error_returns_to_sleep(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()], speak_stream=PortAudioError("Stream is stopped"))
        self.assert_slept_with_microphone_restarted()
        self.assertIn("[turn failed] PortAudioError", self.output)

    def test_transcription_error_returns_to_sleep_without_a_request(self):
        def broken_whisper(*_args, **_kwargs):
            raise RuntimeError("model failure")

        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()], transcribe=broken_whisper)
        self.assert_slept_with_microphone_restarted()
        self.speak_stream.assert_not_called()

    def test_default_speech_recognition_is_tiny_english_greedy(self):
        with self.assertRaises(KeyboardInterrupt):
            # A failed reply ends the fake session after transcription.
            self.run_loop([capture()], speak_stream=RuntimeError("end of test"))
        self.assertEqual(self.whisper.call_args.args, ("tiny.en",))
        self.assertEqual(self.stt.transcribe.call_args.kwargs,
                         {"language": "en", "beam_size": 1,
                          "hotwords": voice_frontend.STT_HOTWORDS})
        self.assertIn("STT: tiny.en, beam 1", self.output)

    def test_speech_recognition_override_reaches_whisper(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()], speak_stream=RuntimeError("end of test"),
                          stt_model="base", stt_beam=5)
        self.assertEqual(self.whisper.call_args.args, ("base",))
        self.assertEqual(self.stt.transcribe.call_args.kwargs["beam_size"], 5)

    def test_note_is_submitted_only_after_a_reply_plays_in_full(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), silence()])
        self.tars.enable_memory_notes.assert_called_once_with()
        self.tars.begin_session.assert_called_once_with()
        self.tars.submit_note.assert_called_once_with()
        self.tars.close.assert_called_once_with()

    def test_failed_reply_never_submits_a_note(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()], speak_stream=PortAudioError("Stream is stopped"))
        self.tars.submit_note.assert_not_called()
        self.tars.close.assert_called_once_with()

    def test_cloud_speech_error_keeps_existing_message(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()],
                          speak_stream=voice_frontend.CloudSpeechError("HTTP 429"))
        self.assert_slept_with_microphone_restarted()
        self.assertIn("[speech unavailable] HTTP 429", self.output)

    def test_retry_prompt_speech_failure_returns_to_sleep(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(overflows=1)],
                          speak=voice_frontend.CloudSpeechError("HTTP 429"))
        self.assert_slept_with_microphone_restarted()
        self.assertIn("[speech unavailable] HTTP 429", self.output)
        self.speak_stream.assert_not_called()

    def test_microphone_restart_failure_is_still_fatal(self):
        with self.assertRaises(voice_frontend.MicrophoneStreamError):
            self.run_loop([capture()], speak_stream=APIStatusError("overloaded"),
                          fail_start=True)
        self.assertEqual(self.buffer.reads, 1)

    def test_successful_turn_still_waits_for_follow_up(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), silence()])
        self.speak_stream.assert_called_once()
        # One pause for the turn, one for the standby line.
        self.assertEqual(self.buffer.events, ["stop", "start", "stop", "start"])
        self.assertNotIn("[turn failed]", self.output)

    def test_failure_report_is_one_bounded_line(self):
        output = io.StringIO()
        with redirect_stdout(output):
            voice_frontend.report_turn_failure(ValueError("x" * 500 + "\nsecond line"))
        self.assertEqual(output.getvalue(), f"[turn failed] ValueError: {'x' * 200}\n")


class ConversationFlowTests(LoopHarness, unittest.TestCase):
    def test_conversation_pause_listens_for_name_before_standby(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), silence()])
        self.assertIn("[waiting for 'TARS']", self.output)
        self.assertLess(self.output.index("[waiting for 'TARS']"), self.output.index("[sleep]"))
        self.assertEqual(self.lines_spoken(), [voice_frontend.SLEEP_LINE])

    def test_noise_and_blank_transcripts_never_end_the_conversation(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), noise(), noise("no_speech"), capture(), capture(),
                           silence(), silence()],
                          transcribe=heard("hello", "", "still there?"))
        self.assertEqual(self.spoken_to_brain(), ["hello", "still there?"])
        self.assertEqual(self.output.count("[sleep]"), 1)

    def test_after_the_pause_only_sentences_naming_tars_are_answered(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), capture(), capture(), silence(), silence()],
                          transcribe=heard("hello", "We should get pizza later.",
                                           "What do you think, Tarz?"))
        self.assertEqual(self.spoken_to_brain(), ["hello", "What do you think, TARS?"])
        # Side chatter is neither printed nor sent anywhere.
        self.assertNotIn("pizza", self.output)
        self.tars.submit_note.assert_called()
        self.assertEqual(self.tars.submit_note.call_count, 2)

    def test_answering_by_name_reopens_the_conversation(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), capture(), capture(), silence(), silence()],
                          transcribe=heard("hello", "TARS, what time is it?", "and tomorrow?"))
        self.assertEqual(self.spoken_to_brain(),
                         ["hello", "TARS, what time is it?", "and tomorrow?"])

    def test_bare_name_listens_without_a_reply(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), capture(), capture(), silence(), silence()],
                          transcribe=heard("hello", "Hey TARS.", "what's the plan?"))
        self.assertEqual(self.spoken_to_brain(), ["hello", "what's the plan?"])

    def test_go_to_sleep_says_the_standby_line_without_a_brain_request(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture()], transcribe=heard("Okay TARS, go to sleep."))
        self.speak_stream.assert_not_called()
        self.assertEqual(self.lines_spoken(), [voice_frontend.SLEEP_LINE])
        self.assertEqual(self.buffer.events, ["stop", "start"])
        self.assertIn("[sleep]", self.output)

    def test_go_to_sleep_without_the_name_is_ignored_after_the_pause(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), capture(), silence()],
                          transcribe=heard("hello", "I need to go to sleep."))
        self.assertEqual(self.spoken_to_brain(), ["hello"])
        self.assertEqual(self.lines_spoken(), [voice_frontend.SLEEP_LINE])

    def test_idle_limit_returns_to_standby_even_after_answers(self):
        with patch.object(voice_frontend, "IDLE_LIMIT", 0.0), self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence()])
        self.speak_stream.assert_called_once()
        self.assertEqual(self.lines_spoken(), [voice_frontend.SLEEP_LINE])

    def test_damaged_audio_is_ignored_quietly_while_waiting_for_name(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), capture(overflows=1), silence()])
        self.assertEqual(self.lines_spoken(), [voice_frontend.SLEEP_LINE])
        self.assertEqual(self.stt.transcribe.call_count, 1)

    def test_side_chatter_is_never_saved_to_diagnostics(self):
        with patch.object(voice_frontend, "save_capture_diagnostics") as save, \
                self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), capture(), capture(), silence(), silence()],
                          transcribe=heard("hello", "pizza later?", "TARS, you there?"),
                          diagnostic_dir=Path("/nonexistent"))
        self.assertEqual([call.args[4] for call in save.call_args_list],
                         ["hello", "TARS, you there?"])

    def test_standby_line_failure_still_reaches_standby(self):
        def speak(_voice, text):
            if text == voice_frontend.SLEEP_LINE:
                raise voice_frontend.CloudSpeechError("HTTP 429")

        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), silence()], speak=speak)
        self.assertIn("[speech unavailable] HTTP 429", self.output)
        self.assertIn("[sleep]", self.output)

    def test_saved_personality_is_loaded_at_startup(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence(), silence()])
        self.tars.enable_saved_settings.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
