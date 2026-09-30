"""Concurrency tests for bounded ElevenLabs sentence lookahead; no paid calls."""

import importlib
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

numpy_stub = MagicMock()
numpy_stub.float32 = "float32"
with patch.dict(sys.modules, {"numpy": numpy_stub, "sounddevice": MagicMock()}):
    sys.modules.pop("tars_voice", None)
    voice_frontend = importlib.import_module("tars_voice")


class StreamResponse:
    def __init__(self, deltas):
        self.deltas = iter(deltas)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.deltas)

    def close(self):
        self.closed = True


class FakeTars:
    def __init__(self, deltas):
        self.response = StreamResponse(deltas)

    def respond_stream(self, _user_input):
        return self.response


class FakeCloudVoice(voice_frontend.ElevenLabsVoice):
    def __init__(self):
        self.prepared = []
        self.played = []
        self.prepare_events = [threading.Event() for _ in range(3)]
        self.first_play_started = threading.Event()
        self.release_first_play = threading.Event()
        self.last_underflows = 0
        self.last_stats = {}
        self.cancelled = False

    def begin_response(self):
        pass

    def ensure_output(self):
        pass

    def prepare(self, text, cancel_event=None):
        index = len(self.prepared)
        self.prepared.append(text)
        if index < len(self.prepare_events):
            self.prepare_events[index].set()
        return text

    def play(self, prepared):
        if not self.played:
            self.first_play_started.set()
            if not self.release_first_play.wait(2):
                raise TimeoutError("test did not release first playback")
        requested = time.perf_counter()
        self.played.append(prepared)
        self.last_stats = {"text": prepared}
        return 0.01, 0.01, requested

    def cancel_pending(self):
        self.cancelled = True


class PipelineTests(unittest.TestCase):
    def test_cloud_pipeline_overlaps_one_sentence_and_never_runs_two_ahead(self):
        tars = FakeTars(["One. Two. Three."])
        voice = FakeCloudVoice()
        result = {}
        errors = []

        def run():
            try:
                result["metrics"] = voice_frontend.speak_stream(tars, voice, "hello")
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(voice.first_play_started.wait(1), "first sentence never played")
        self.assertTrue(voice.prepare_events[1].wait(1),
                        "second sentence was not prepared during first playback")
        self.assertFalse(voice.prepare_events[2].wait(0.1),
                         "third sentence began before the one-sentence slot opened")

        voice.release_first_play.set()
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(voice.prepared, ["One.", "Two.", "Three."])
        self.assertEqual(voice.played, ["One.", "Two.", "Three."])
        self.assertEqual(result["metrics"]["chunks"], 3)
        self.assertEqual(len(result["metrics"]["cloud_sentences"]), 3)
        self.assertTrue(tars.response.closed)
        self.assertTrue(voice.cancelled)

    def test_cloud_pipeline_propagates_worker_failure(self):
        class FailingVoice(FakeCloudVoice):
            def prepare(self, text, cancel_event=None):
                raise voice_frontend.CloudSpeechError("safe test failure")

        tars = FakeTars(["One sentence."])
        voice = FailingVoice()

        with self.assertRaisesRegex(voice_frontend.CloudSpeechError, "safe test failure"):
            voice_frontend.speak_stream(tars, voice, "hello")

        self.assertTrue(tars.response.closed)
        self.assertTrue(voice.cancelled)

    def test_keyboard_interrupt_cancels_inflight_lookahead_and_joins_worker(self):
        class CancellationVoice(FakeCloudVoice):
            def __init__(self):
                super().__init__()
                self.second_prepare_started = threading.Event()

            def prepare(self, text, cancel_event=None):
                self.prepared.append(text)
                if len(self.prepared) == 1:
                    return text
                self.second_prepare_started.set()
                if not cancel_event.wait(2):
                    raise TimeoutError("lookahead worker was not cancelled")
                raise voice_frontend.SpeechPreparationCancelled()

            def play(self, prepared):
                if not self.second_prepare_started.wait(1):
                    raise TimeoutError("second sentence did not begin preparing")
                raise KeyboardInterrupt()

        tars = FakeTars(["One. Two."])
        voice = CancellationVoice()

        with self.assertRaises(KeyboardInterrupt):
            voice_frontend.speak_stream(tars, voice, "hello")

        self.assertTrue(voice.second_prepare_started.is_set())
        self.assertTrue(voice.cancelled)
        self.assertTrue(tars.response.closed)
        self.assertFalse(any(
            thread.name == "tars-elevenlabs-lookahead" and thread.is_alive()
            for thread in threading.enumerate()
        ))


if __name__ == "__main__":
    unittest.main()
