"""Transport/playback tests use fake devices and responses, never paid calls."""

from email.message import Message
import importlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cloud_speech as cloud


class Response:
    def __init__(self, chunks, mime="audio/pcm"):
        self.chunks = iter(chunks)
        self.closed = False
        self.headers = Message()
        self.headers["Content-Type"] = mime

    def read(self, size):
        chunk = next(self.chunks, b"")
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class GatedResponse(Response):
    """Pause a network read so tests can inspect playback while bytes are late."""

    def __init__(self, chunks, blocked, release, events):
        super().__init__(chunks)
        self.blocked = blocked
        self.release = release
        self.events = events
        self.reads = 0

    def read(self, size):
        self.reads += 1
        self.events.append(f"read-{self.reads}")
        if self.reads == 2:
            self.blocked.set()
            if not self.release.wait(2):
                raise TimeoutError("test did not release delayed network read")
        return super().read(size)

    def __exit__(self, *args):
        self.events.append("response-closed")
        return super().__exit__(*args)


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "ELEVENLABS_API_KEY": "test-key-never-log",
            "ELEVENLABS_VOICE_ID": "voice123",
            "ELEVENLABS_MODEL_ID": cloud.DEFAULT_MODEL,
            "TARS_NAME_ALIAS": "",
            "ELEVENLABS_SPEED": "",
            "TARS_VOLUME": "",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.device = MagicMock()
        self.device.write.return_value = False
        self.audio = MagicMock()
        self.audio.RawOutputStream.return_value = self.device
        self.modules = patch.dict(sys.modules, {"sounddevice": self.audio})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.opener = MagicMock()
        self.network = patch.object(cloud, "build_opener", return_value=self.opener)
        self.network.start()
        self.addCleanup(self.network.stop)

    def respond(self, chunks, mime="audio/pcm"):
        response = Response(chunks, mime)
        self.opener.open.return_value = response
        return response

    def payload(self):
        return json.loads(self.opener.open.call_args.args[0].data)

    def test_stream_keeps_split_samples_and_reuses_device(self):
        voice = cloud.ElevenLabsVoice()
        first = self.respond([b"\x01", b"\x00\x02", b"\x00"])
        prepare, playback, requested = voice.speak("Hello, Luca.")
        self.assertGreaterEqual(prepare, 0)
        self.assertGreaterEqual(playback, 0)
        self.assertIsNotNone(requested)
        self.assertEqual(b"".join(c.args[0] for c in self.device.write.call_args_list),
                         b"\x01\x00\x02\x00")
        self.assertTrue(first.closed)
        self.assertEqual(self.payload()["text"], "Hello, Luca.")
        self.assertEqual(self.payload()["voice_settings"]["speed"], cloud.DEFAULT_SPEED)
        self.assertNotIn("previous_text", self.payload())
        request = self.opener.open.call_args.args[0]
        self.assertTrue(request.full_url.endswith("?output_format=pcm_24000"))
        self.assertEqual(request.get_header("Xi-api-key"), "test-key-never-log")
        self.assertEqual(self.opener.open.call_args.kwargs["timeout"], cloud.SOCKET_TIMEOUT)
        self.respond([b"\x00\x00"])
        voice.speak("We are ready.")
        self.assertEqual(self.payload()["previous_text"], "Hello, Luca.")
        self.assertEqual(self.audio.RawOutputStream.call_count, 1)
        self.assertEqual(self.device.stop.call_count, 2)
        self.device.stop.assert_called_with(ignore_errors=False)
        voice.begin_response()
        self.respond([b"\x00\x00"])
        voice.speak("Next answer.")
        self.assertNotIn("previous_text", self.payload())
        voice.close()
        self.device.close.assert_called_once()

    def test_prepare_downloads_without_playing_until_play_is_called(self):
        voice = cloud.ElevenLabsVoice()
        self.respond([b"\x01\x00", b"\x02\x00"])

        prepared = voice.prepare("First sentence.")

        self.device.start.assert_not_called()
        self.device.write.assert_not_called()
        self.device.stop.assert_not_called()
        self.assertEqual(prepared.pcm, b"\x01\x00\x02\x00")

        voice.play(prepared)

        self.device.start.assert_called_once()
        self.device.write.assert_called_once_with(b"\x01\x00\x02\x00")
        self.device.stop.assert_called_once_with(ignore_errors=False)

    def test_delayed_network_is_fully_buffered_before_real_time_playback(self):
        events = []
        blocked, release = threading.Event(), threading.Event()
        block = b"\x00\x00" * 2048
        response = GatedResponse([block, block, block], blocked, release, events)
        self.opener.open.return_value = response
        self.device.start.side_effect = lambda: events.append("device-start")
        self.device.write.side_effect = lambda data: events.append(("device-write", len(data))) or False

        voice = cloud.ElevenLabsVoice()
        result = []
        worker = threading.Thread(target=lambda: result.append(voice.speak("A delayed sentence.")))
        worker.start()
        self.assertTrue(blocked.wait(1), "network read never reached the delay gate")
        self.device.start.assert_not_called()
        self.device.write.assert_not_called()

        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive(), "speech worker did not finish")
        self.assertEqual(len(result), 1)
        self.assertTrue(response.closed)
        self.assertLess(events.index("response-closed"), events.index("device-start"))
        self.assertEqual(self.device.write.call_count, 1)
        self.assertEqual(self.device.write.call_args.args[0], block * 3)
        self.assertEqual(voice.last_stats["audio_bytes"], len(block) * 3)
        self.assertEqual(voice.last_stats["audio_s"], len(block) * 3 / 48000)
        self.assertEqual(voice.last_stats["pcm_chunks"], 3)

    def test_playback_consumption_cannot_block_or_resume_network_download(self):
        events = []
        playback_blocked, release = threading.Event(), threading.Event()
        block = b"\x00\x00" * 512
        response = self.respond([block, block])

        def consume_in_real_time(data):
            events.append("playback-started")
            self.assertTrue(response.closed)
            playback_blocked.set()
            if not release.wait(2):
                raise TimeoutError("test did not release simulated playback")
            events.append("playback-finished")
            return False

        self.device.write.side_effect = consume_in_real_time
        voice = cloud.ElevenLabsVoice()
        failures = []

        def run():
            try:
                voice.speak("Playback is deliberately slow.")
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(playback_blocked.wait(1), "playback never reached its delay gate")
        self.assertTrue(response.closed)
        self.assertEqual(voice.last_stats["pcm_chunks"], 2)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive(), "speech worker did not finish")
        self.assertEqual(failures, [])
        self.assertEqual(events, ["playback-started", "playback-finished"])

    def test_underflow_is_always_reported_and_recorded(self):
        self.respond([b"\x00\x00" * 16])
        self.device.write.return_value = True
        voice = cloud.ElevenLabsVoice()
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            voice.speak("Report the audio problem.")
        self.assertIn("Output underflow", stderr.getvalue())
        self.assertEqual(voice.last_underflows, 1)
        self.assertEqual(voice.last_stats["underflows"], 1)
        self.assertGreaterEqual(voice.last_stats["download_s"], 0)
        self.assertGreaterEqual(voice.last_stats["playback_s"], 0)

    def test_name_alias_is_opt_in_and_whole_word_only(self):
        voice = cloud.ElevenLabsVoice()
        self.assertEqual(voice.prepare_text("Luca, Lucas and LUCA's"), "Luca, Lucas and LUCA's")
        voice = cloud.ElevenLabsVoice(name_alias="Loo kah")
        self.assertEqual(voice.prepare_text("Luca, Lucas and LUCA's"), "Loo kah, Lucas and Loo kah's")
        with self.assertRaises(cloud.CloudSpeechError):
            voice.prepare_text("Hello [[IPA]]")

    def test_reject_empty_or_incomplete_stream_and_close(self):
        for chunks in ([], [b"\x01"]):
            response = self.respond(chunks)
            voice = cloud.ElevenLabsVoice()
            with self.assertRaisesRegex(cloud.CloudSpeechError, "empty or incomplete"):
                voice.speak("Luca.")
            self.assertTrue(response.closed)
        self.device.abort.assert_called()

    def test_reject_json_error_as_audio_without_speaker_write(self):
        response = self.respond([b'{"detail":"private response"}'], "application/json")
        with self.assertRaisesRegex(cloud.CloudSpeechError, "unexpected audio format"):
            cloud.ElevenLabsVoice().speak("Luca.")
        self.device.write.assert_not_called()
        self.assertTrue(response.closed)

    def test_http_error_is_sanitized_and_not_retried(self):
        self.opener.open.side_effect = HTTPError(
            "https://api.elevenlabs.io/private", 401, "private response with key",
            {}, io.BytesIO(b"test-key-never-log"))
        with self.assertRaises(cloud.CloudSpeechError) as error:
            cloud.ElevenLabsVoice().speak("private conversation")
        self.assertIn("HTTP 401", str(error.exception))
        self.assertNotIn("test-key-never-log", str(error.exception))
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(self.opener.open.call_count, 1)

    def test_size_and_deadline_bounds(self):
        self.respond([b"\x00\x00" * 3])
        with patch.object(cloud, "MAX_AUDIO_BYTES", 4):
            with self.assertRaisesRegex(cloud.CloudSpeechError, "size limit"):
                cloud.ElevenLabsVoice().speak("Hello.")
        self.respond([b"\x00\x00"])
        with patch.object(cloud.time, "monotonic", side_effect=[0, 91]):
            with self.assertRaisesRegex(cloud.CloudSpeechError, "time limit"):
                cloud.ElevenLabsVoice().speak("Hello.")

    def test_interrupt_aborts_audio_and_closes_http(self):
        response = self.respond([b"\x00\x00"])
        self.device.write.side_effect = KeyboardInterrupt
        voice = cloud.ElevenLabsVoice()
        with self.assertRaises(KeyboardInterrupt):
            voice.speak("Hello.")
        self.assertTrue(response.closed)
        self.device.abort.assert_called()
        voice.close()
        self.device.close.assert_called_once()

    def test_abort_failure_does_not_hide_original_playback_error(self):
        self.respond([b"\x00\x00"])
        self.device.write.side_effect = ValueError("original playback failure")
        self.device.abort.side_effect = RuntimeError("cleanup failure")

        with self.assertRaisesRegex(ValueError, "original playback failure"):
            cloud.ElevenLabsVoice().speak("Hello.")

    def test_device_failure_happens_before_charged_request(self):
        self.audio.RawOutputStream.side_effect = RuntimeError("no device")
        with self.assertRaisesRegex(RuntimeError, "no device"):
            cloud.ElevenLabsVoice().speak("Hello.")
        self.opener.open.assert_not_called()

    def test_missing_or_invalid_configuration_no_requests(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            with self.assertRaisesRegex(cloud.CloudSpeechError, "ELEVENLABS_API_KEY"):
                cloud.ElevenLabsVoice()
        with self.assertRaises(cloud.CloudSpeechError):
            cloud.ElevenLabsVoice(voice_id="../bad-path")
        with self.assertRaises(cloud.CloudSpeechError):
            cloud.ElevenLabsVoice(name_alias="[[IPA]]")
        self.opener.open.assert_not_called()

    def test_list_voices_follows_page_tokens_and_filters_output(self):
        self.opener.open.side_effect = [
            Response([json.dumps({"voices": [{"voice_id": "a", "name": "A", "private": "hidden"}],
                                  "has_more": True, "next_page_token": "next"}).encode()], "application/json"),
            Response([json.dumps({"voices": [{"voice_id": "b", "name": "B"}],
                                  "has_more": False}).encode()], "application/json"),
        ]
        rows = cloud.list_voices("deep")
        self.assertEqual([r["voice_id"] for r in rows], ["a", "b"])
        self.assertNotIn("private", rows[0])
        request = self.opener.open.call_args.args[0]
        self.assertIn("next_page_token=next", request.full_url)
        self.assertIn("search=deep", request.full_url)
        self.assertEqual(request.method, "GET")

    @staticmethod
    def pcm(*samples):
        return b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples)

    def test_default_delivery_is_slower_and_quieter(self):
        voice = cloud.ElevenLabsVoice()
        self.respond([self.pcm(1000, -1000, 32767, -32768, 0)])
        prepared = voice.prepare("Hello.")
        self.assertEqual(self.payload()["voice_settings"]["speed"], 0.92)
        self.assertEqual(prepared.pcm, self.pcm(850, -850, 27852, -27853, 0))
        self.assertEqual(prepared.stats["audio_bytes"], 10)

    def test_full_volume_leaves_audio_bit_exact(self):
        original = self.pcm(1000, -1000, 32767, -32768, 7)
        with patch.dict(os.environ, {"TARS_VOLUME": "1.0", "ELEVENLABS_SPEED": "1.0"}):
            voice = cloud.ElevenLabsVoice()
        self.respond([original])
        prepared = voice.prepare("Hello.")
        self.assertEqual(prepared.pcm, original)
        self.assertEqual(self.payload()["voice_settings"]["speed"], 1.0)

    def test_environment_overrides_delivery_settings(self):
        with patch.dict(os.environ, {"TARS_VOLUME": "0.5", "ELEVENLABS_SPEED": "0.8"}):
            voice = cloud.ElevenLabsVoice()
        self.respond([self.pcm(1000)])
        self.assertEqual(voice.prepare("Hello.").pcm, self.pcm(500))
        self.assertEqual(self.payload()["voice_settings"]["speed"], 0.8)

    def test_invalid_delivery_settings_fail_before_any_request(self):
        for name, value in (("ELEVENLABS_SPEED", "fast"), ("ELEVENLABS_SPEED", "1.5"),
                            ("ELEVENLABS_SPEED", "nan"), ("TARS_VOLUME", "0"),
                            ("TARS_VOLUME", "1.2"), ("TARS_VOLUME", "-0.5")):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(cloud.CloudSpeechError, name):
                    cloud.ElevenLabsVoice()
        self.opener.open.assert_not_called()

    def test_cloud_name_test_avoids_piper_wake_stt_and_brain(self):
        blocked = {"piper": None, "openwakeword": None, "faster_whisper": None,
                   "brain": None, "numpy": MagicMock()}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("tars_voice", None)
            frontend = importlib.import_module("tars_voice")
            with patch.object(sys, "argv", ["tars_voice.py", "--tts", "elevenlabs", "--test-name"]), \
                    patch.object(frontend, "load_environment"), \
                    patch.object(cloud.ElevenLabsVoice, "speak", return_value=(0.0, 0.0, 1.0)) as speak, \
                    patch("sys.stdout", new_callable=io.StringIO):
                frontend.main()
                speak.assert_called_once_with(cloud.NAME_TEST)
            sys.modules.pop("tars_voice", None)


if __name__ == "__main__":
    unittest.main()
