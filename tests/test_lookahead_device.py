"""The ElevenLabs lookahead worker never touches the audio device (H2).

Uses the real ElevenLabsVoice and the real lookahead loop with a fake output
device and fake network; no paid calls, audio, or microphone.
"""

from email.message import Message
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cloud_speech as cloud  # noqa: E402

numpy_stub = MagicMock()
numpy_stub.float32 = "float32"
with patch.dict(sys.modules, {"numpy": numpy_stub, "sounddevice": MagicMock()}):
    sys.modules.pop("tars_voice", None)
    voice_frontend = importlib.import_module("tars_voice")

PCM = {"One.": b"\x01\x00", "Two.": b"\x02\x00", "Three.": b"\x03\x00"}


class Response:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.headers = Message()
        self.headers["Content-Type"] = "audio/pcm"

    def read(self, _size):
        chunk = next(self.chunks, b"")
        return chunk() if callable(chunk) else chunk

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class RecordingDevice:
    """Fake RawOutputStream that records which thread performs each action."""

    def __init__(self, log, block_write=None):
        self.log = log
        self.block_write = block_write or {}
        self.writing = False
        self.completed_writes = []
        self.record("open")

    def record(self, action):
        self.log.append((action, threading.get_ident()))

    def start(self):
        self.record("start")

    def write(self, data):
        self.record("write")
        self.writing = True
        gate = self.block_write.get(data)
        if gate is not None and not gate.wait(2):
            raise TimeoutError("the next sentence never failed during this write")
        self.writing = False
        self.completed_writes.append(data)
        return False

    def stop(self, ignore_errors=True):
        self.record("stop")

    def abort(self):
        self.record("abort-during-write" if self.writing else "abort")

    def close(self):
        self.record("close")


class StreamResponse:
    def __init__(self, deltas):
        self.deltas = iter(deltas)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.deltas)

    def close(self):
        pass


class LookaheadDeviceTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "ELEVENLABS_API_KEY": "test-key-never-log",
            "ELEVENLABS_VOICE_ID": "voice123",
            "ELEVENLABS_MODEL_ID": cloud.DEFAULT_MODEL,
            "TARS_NAME_ALIAS": "",
        })
        env.start()
        self.addCleanup(env.stop)
        self.log = []
        self.requests = []
        self.next_failed = threading.Event()
        self.main_thread = threading.get_ident()
        self.device = None

        def open_device(**_kwargs):
            self.device = RecordingDevice(
                self.log, block_write={PCM["Two."]: self.next_failed})
            return self.device

        audio = MagicMock()
        audio.RawOutputStream.side_effect = open_device
        self.audio = audio
        modules = patch.dict(sys.modules, {"sounddevice": audio})
        modules.start()
        self.addCleanup(modules.stop)

        def open_request(request, timeout):
            text = json.loads(request.data)["text"]
            self.requests.append(text)
            self.log.append((f"request {text}", threading.get_ident()))
            if text == "Three.":
                # Simulates the Part 3 HTTPS block: the lookahead download of
                # sentence 3 fails while sentence 2 is still being written.
                self.next_failed.set()
                raise URLError("connection reset")
            return Response([PCM[text]])

        opener = MagicMock()
        opener.open.side_effect = open_request
        network = patch.object(cloud, "build_opener", return_value=opener)
        network.start()
        self.addCleanup(network.stop)

    def speak_reply(self, deltas):
        tars = MagicMock()
        tars.respond_stream.return_value = StreamResponse(deltas)
        voice = cloud.ElevenLabsVoice()
        self.addCleanup(voice.close)
        self.tars = tars
        return voice_frontend.speak_stream(tars, voice, "hello")

    def test_failed_next_sentence_lets_current_finish_then_reports(self):
        with self.assertRaisesRegex(cloud.CloudSpeechError, "connection failed"):
            self.speak_reply(["One. Two. Three."])

        # Sentence 2 finished playing even though sentence 3 failed mid-write.
        self.assertTrue(self.next_failed.is_set())
        self.assertEqual(self.device.completed_writes, [PCM["One."], PCM["Two."]])
        actions = [action for action, _thread in self.log]
        self.assertNotIn("abort-during-write", actions)
        # Sentence 3 failed during sentence 2's write; this thread then
        # finished that write, stopped the device, and only then raised.
        self.assertEqual(actions[-3:], ["write", "request Three.", "stop"])

        device_actions = [(action, thread) for action, thread in self.log
                          if not action.startswith("request")]
        worker_actions = [action for action, thread in device_actions
                          if thread != self.main_thread]
        self.assertEqual(worker_actions, [],
                         "the lookahead worker touched the audio device")
        self.assertEqual(self.requests, ["One.", "Two.", "Three."])

    def test_device_opens_on_main_thread_before_any_paid_request(self):
        self.next_failed.set()  # no sentence blocks in this test
        self.speak_reply(["One. Two."])
        actions = [action for action, _thread in self.log]
        self.assertLess(actions.index("open"), actions.index("request One."))
        opened_by = next(thread for action, thread in self.log if action == "open")
        self.assertEqual(opened_by, self.main_thread)

    def test_missing_speaker_stops_before_claude_or_elevenlabs_is_charged(self):
        self.audio.RawOutputStream.side_effect = RuntimeError("no output device")
        with self.assertRaisesRegex(RuntimeError, "no output device"):
            self.speak_reply(["One."])
        self.tars.respond_stream.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_cancelled_download_is_discarded_and_keeps_speech_context(self):
        # L4: a worker cancelled mid-download (its response closed, so the
        # read ends cleanly) must not return audio or change the context
        # sent with the next reply.
        cancel = threading.Event()

        def closed_by_cancellation():
            cancel.set()
            return b""

        def open_request(request, timeout):
            return Response([b"\x01\x00", closed_by_cancellation])

        voice = cloud.ElevenLabsVoice()
        with patch.object(cloud, "build_opener") as build:
            build.return_value.open.side_effect = open_request
            with self.assertRaises(cloud.SpeechPreparationCancelled):
                voice.prepare("Stale sentence.", cancel_event=cancel)
        self.assertEqual(voice._previous_text, "")
        self.assertIsNone(self.device, "prepare() opened the audio device")


if __name__ == "__main__":
    unittest.main()
