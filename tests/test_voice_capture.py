"""Deterministic capture-loop tests; no microphone, model, or paid API calls."""

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

numpy_stub = MagicMock()
numpy_stub.float32 = "float32"
with patch.dict(sys.modules, {"numpy": numpy_stub, "sounddevice": MagicMock()}):
    sys.modules.pop("tars_voice", None)
    voice_frontend = importlib.import_module("tars_voice")


class FakeStream:
    def __init__(self, overflows):
        self.overflows = iter(overflows)
        self.read_count = 0

    def read(self, frames):
        self.read_count += 1
        return object(), next(self.overflows)


class FakeSamples:
    def __init__(self, length):
        self.length = length

    def flatten(self):
        return self

    def astype(self, _dtype):
        return self

    def __truediv__(self, _divisor):
        return self

    def __len__(self):
        return self.length


class CopyFrame:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return self


class Status:
    def __init__(self, input_overflow=False):
        self.input_overflow = input_overflow


class CaptureTests(unittest.TestCase):
    def test_sixteen_second_utterance_no_longer_hits_old_cap(self):
        rms_levels = [800.0] * 200 + [0.0] * 16
        stream = FakeStream([False] * len(rms_levels))
        reader = voice_frontend.TrackedInput(stream, warn=MagicMock())
        samples = FakeSamples(voice_frontend.SR * 17)

        with patch.object(voice_frontend, "level", side_effect=rms_levels), \
                patch.object(voice_frontend.np, "concatenate", return_value=samples):
            result = voice_frontend.record_utterance(reader, threshold=200.0, wait=2.0)

        self.assertEqual(result.endpoint_reason, "silence")
        self.assertGreater(len(result.clip), voice_frontend.SR * 15)

    def test_natural_pause_shorter_than_hangover_does_not_end_turn(self):
        # 0.4 s speech, 0.8 s natural pause, more speech, then endpoint silence.
        rms_levels = [900.0] * 5 + [0.0] * 10 + [800.0] * 5 + [0.0] * 16
        stream = FakeStream([False] * len(rms_levels))
        reader = voice_frontend.TrackedInput(stream, warn=MagicMock())
        samples = FakeSamples(voice_frontend.SR * 3)

        with patch.object(voice_frontend, "level", side_effect=rms_levels), \
                patch.object(voice_frontend.np, "concatenate", return_value=samples):
            result = voice_frontend.record_utterance(reader, threshold=200.0, wait=2.0)

        self.assertIsNotNone(result.clip)
        self.assertEqual(result.endpoint_reason, "silence")
        self.assertIn(800.0, result.rms_levels)
        self.assertGreater(stream.read_count, 20)
        self.assertEqual(result.input_overflows, 0)

    def test_single_loud_spike_is_not_promoted_by_padding(self):
        # One 80 ms spike followed by endpoint silence must remain too short,
        # even though the retained pre-roll and tail make the clip much longer.
        rms_levels = [0.0] * 4 + [900.0] + [0.0] * 16
        stream = FakeStream([False] * len(rms_levels))
        reader = voice_frontend.TrackedInput(stream, warn=MagicMock())

        with patch.object(voice_frontend, "level", side_effect=rms_levels):
            result = voice_frontend.record_utterance(reader, threshold=200.0, wait=2.0)

        self.assertIsNone(result.clip)
        self.assertEqual(result.endpoint_reason, "too_short")

    def test_every_input_overflow_is_counted_and_warned(self):
        warnings = []
        reader = voice_frontend.TrackedInput(
            FakeStream([False, True, True, False]), warn=warnings.append
        )

        for _ in range(4):
            reader.read()

        self.assertEqual(reader.stats.reads, 4)
        self.assertEqual(reader.stats.overflows, 2)
        self.assertEqual(len(warnings), 2)
        self.assertIn("#1", warnings[0])
        self.assertIn("#2", warnings[1])

    def test_callback_buffer_drains_audio_while_consumer_is_delayed(self):
        buffered = voice_frontend.BufferedInput(max_seconds=0.16)
        first, second = CopyFrame("first"), CopyFrame("second")
        buffered.callback(first, voice_frontend.FRAME, None, Status())
        buffered.callback(second, voice_frontend.FRAME, None, Status())

        self.assertIs(buffered.read()[0], first)
        self.assertIs(buffered.read()[0], second)

    def test_callback_queue_loss_is_reported_to_tracked_reader(self):
        buffered = voice_frontend.BufferedInput(max_seconds=0.16)
        buffered.callback(CopyFrame("old"), voice_frontend.FRAME, None, Status())
        buffered.callback(CopyFrame("middle"), voice_frontend.FRAME, None, Status())
        buffered.callback(CopyFrame("new"), voice_frontend.FRAME, None, Status())
        warnings = []
        reader = voice_frontend.TrackedInput(buffered, warn=warnings.append)

        reader.read()

        self.assertEqual(reader.stats.overflows, 1)
        self.assertEqual(len(warnings), 1)

    def test_callback_hardware_overflow_is_reported(self):
        buffered = voice_frontend.BufferedInput()
        buffered.callback(
            CopyFrame("damaged"), voice_frontend.FRAME, None, Status(input_overflow=True)
        )
        _, overflows = buffered.read()
        self.assertEqual(overflows, 1)

    def test_buffer_restart_discards_stale_audio(self):
        buffered = voice_frontend.BufferedInput()
        stream = MagicMock()
        buffered.attach(stream)
        buffered.callback(
            CopyFrame("stale"), voice_frontend.FRAME, None,
            Status(input_overflow=True),
        )

        buffered.stop(ignore_errors=False)
        buffered.start()

        stream.stop.assert_called_once_with(ignore_errors=False)
        stream.start.assert_called_once_with()
        self.assertTrue(buffered._frames.empty())
        self.assertEqual(buffered._pending_overflows, 0)

    def test_whisper_warmup_fully_consumes_generator(self):
        consumed = []

        def segments():
            consumed.append("started")
            yield object()
            consumed.append("finished")

        model = MagicMock()
        model.transcribe.return_value = (segments(), object())

        voice_frontend.warm_stt(model)

        self.assertEqual(consumed, ["started", "finished"])
        model.transcribe.assert_called_once()

    def test_input_restarts_even_when_processing_fails(self):
        stream = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with voice_frontend.paused_input(stream):
                raise RuntimeError("failed")

        stream.stop.assert_called_once_with(ignore_errors=False)
        stream.start.assert_called_once_with()

    def test_restart_failure_after_processing_error_stops_conversation(self):
        stream = MagicMock()
        stream.start.side_effect = RuntimeError("restart failed")

        with self.assertRaisesRegex(
                voice_frontend.MicrophoneStreamError, "dead listening loop"):
            with voice_frontend.paused_input(stream):
                raise ValueError("processing failed")

    def test_clean_restart_failure_stops_with_actionable_error(self):
        stream = MagicMock()
        stream.start.side_effect = RuntimeError("restart failed")

        with self.assertRaisesRegex(RuntimeError, "did not restart"):
            with voice_frontend.paused_input(stream):
                pass

    def test_private_capture_directory_is_rejected_inside_git_project(self):
        project_child = Path(voice_frontend.__file__).resolve().parent / "private-captures"
        with self.assertRaisesRegex(ValueError, "outside the TARS Git folder"):
            voice_frontend.prepare_diagnostic_directory(project_child)

    def test_private_capture_directory_is_allowed_outside_git_project(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "captures"
            path = voice_frontend.prepare_diagnostic_directory(base)
            self.assertTrue(path.is_dir())
            self.assertEqual(path.parent, base.resolve())
            self.assertTrue(path.name.startswith("session-"))

    def test_private_capture_path_does_not_mutate_existing_base_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            before = base.stat().st_mode
            path = voice_frontend.prepare_diagnostic_directory(base)
            self.assertEqual(base.stat().st_mode, before)
            self.assertNotEqual(path, base)


if __name__ == "__main__":
    unittest.main()
