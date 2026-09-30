"""A failed conversation turn returns TARS to sleep; no microphone or paid calls."""

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


def capture(overflows=0):
    return voice_frontend.CaptureResult(
        clip=[0.0] * 1600, last_loud_read_at=0.0, endpoint_reason="silence",
        rms_levels=[], input_overflows=overflows,
    )


class TurnRecoveryTests(unittest.TestCase):
    def run_loop(self, captures, speak_stream=None, speak=None, transcribe=None,
                 fail_start=False):
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
                voice_frontend.run_conversation(MagicMock())
            finally:
                self.output = output.getvalue()
                self.buffer = FakeBuffer.latest

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
        silence = voice_frontend.CaptureResult(None, None, "speech_timeout", [], 0)
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([capture(), silence])
        self.speak_stream.assert_called_once()
        self.assertEqual(self.buffer.events, ["stop", "start"])
        self.assertNotIn("[turn failed]", self.output)

    def test_failure_report_is_one_bounded_line(self):
        output = io.StringIO()
        with redirect_stdout(output):
            voice_frontend.report_turn_failure(ValueError("x" * 500 + "\nsecond line"))
        self.assertEqual(output.getvalue(), f"[turn failed] ValueError: {'x' * 200}\n")


if __name__ == "__main__":
    unittest.main()
