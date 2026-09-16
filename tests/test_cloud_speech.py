"""Transport/playback tests use fake devices and responses, never paid calls."""

from email.message import Message
import importlib
import io
import json
import os
from pathlib import Path
import sys
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


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "ELEVENLABS_API_KEY": "test-key-never-log",
            "ELEVENLABS_VOICE_ID": "voice123",
            "ELEVENLABS_MODEL_ID": cloud.DEFAULT_MODEL,
            "TARS_NAME_ALIAS": "",
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
        voice.begin_response()
        self.respond([b"\x00\x00"])
        voice.speak("Next answer.")
        self.assertNotIn("previous_text", self.payload())
        voice.close()
        self.device.close.assert_called_once()

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
