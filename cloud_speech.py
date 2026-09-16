"""Opt-in ElevenLabs speech with streamed mono 24 kHz signed 16-bit PCM.

No API key or voice is selected implicitly. Importing this module does not
contact the service, open audio devices, or load speech-recognition models.
"""

import argparse
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_ROOT = "https://api.elevenlabs.io"
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
# Hold a short PCM lead before starting the speaker. This absorbs normal
# internet/generation jitter without waiting for the complete response.
PLAYBACK_PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.5)
DEFAULT_MODEL = "eleven_multilingual_v2"
MODELS = (DEFAULT_MODEL, "eleven_flash_v2_5")
SOCKET_TIMEOUT = 15.0
REQUEST_DEADLINE = 90.0
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * 120
NAME_TEST = "Luca. I'm listening, Luca."
VOICE_TEST = (
    "Luca, I'm ready when you are. "
    "We can check the wiring, work through a problem, or just talk. "
    "My humor setting is seventy-five percent. You're welcome."
)


class CloudSpeechError(RuntimeError):
    """A safe, actionable error; never contains an API response or a key."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the xi-api-key header to a redirected host.
        fp.close()
        raise CloudSpeechError("ElevenLabs returned an unexpected redirect.")


def load_environment():
    from dotenv import load_dotenv

    # Read this project's .env only; exported variables take precedence.
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


def _api_key():
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise CloudSpeechError(
            "Set ELEVENLABS_API_KEY in ~/TARS/.env or the environment first. "
            "See docs/cloud-voice.md."
        )
    if not key.isascii() or any(ord(c) < 33 or ord(c) == 127 for c in key):
        raise CloudSpeechError("ELEVENLABS_API_KEY has invalid characters.")
    return key


@contextmanager
def _request(path, key, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        API_ROOT + path,
        data=body,
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with build_opener(_NoRedirects()).open(request, timeout=SOCKET_TIMEOUT) as response:
            yield response
    except HTTPError as exc:
        messages = {
            401: "Check the ElevenLabs API key and its permissions.",
            403: "Check key permissions, account credits, and voice access.",
            404: "The selected voice is unavailable. List voices again.",
            422: "The chosen voice/model settings were rejected.",
            429: "ElevenLabs quota or rate limit reached. Check the account dashboard.",
        }
        status = exc.code
        exc.close()
        raise CloudSpeechError(
            f"ElevenLabs HTTP {status}. "
            + messages.get(status, "The service request failed. Try again later.")
        ) from None
    except (URLError, TimeoutError, OSError, http.client.HTTPException):
        raise CloudSpeechError(
            "ElevenLabs connection failed or timed out. Check the Pi's internet connection."
        ) from None


def list_voices(search=""):
    """Read account-accessible voices; no speech is generated or purchased."""
    key, voices, token = _api_key(), [], None
    seen_tokens = set()
    started = time.monotonic()
    for _ in range(20):
        if time.monotonic() - started > REQUEST_DEADLINE:
            raise CloudSpeechError("Voice listing exceeded its time limit; narrow --search.")
        params = {"page_size": 100, "include_total_count": "false"}
        if search:
            params["search"] = search
        if token:
            params["next_page_token"] = token
        with _request("/v2/voices?" + urlencode(params), key) as response:
            data = response.read(2_000_001)
        try:
            if len(data) > 2_000_000:
                raise ValueError
            page = json.loads(data)
            rows = page["voices"]
            if not isinstance(rows, list) or not isinstance(page["has_more"], bool):
                raise ValueError
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("voice_id"), str):
                    raise ValueError
                voices.append({"voice_id": row["voice_id"], "name": row.get("name", ""),
                               "category": row.get("category", ""),
                               "labels": row.get("labels", {})})
            if not page["has_more"]:
                return voices
            token = page["next_page_token"]
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise ValueError
            seen_tokens.add(token)
        except (ValueError, KeyError, TypeError):
            raise CloudSpeechError("ElevenLabs returned an invalid voice list.") from None
    raise CloudSpeechError("Too many voice pages; use --search to narrow the list.")


class ElevenLabsVoice:
    def __init__(self, voice_id=None, model_id=None, name_alias=None):
        self._key = _api_key()
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        self.model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID", DEFAULT_MODEL).strip()
        self.name_alias = name_alias if name_alias is not None else os.environ.get("TARS_NAME_ALIAS", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.voice_id):
            raise CloudSpeechError("Choose a voice and set ELEVENLABS_VOICE_ID. See docs/cloud-voice.md.")
        if self.model_id not in MODELS:
            raise CloudSpeechError("Choose eleven_multilingual_v2 or eleven_flash_v2_5 for ELEVENLABS_MODEL_ID.")
        if (len(self.name_alias) > 60 or any(ord(c) < 32 for c in self.name_alias)
                or any(c in self.name_alias for c in "[]<>")):
            raise CloudSpeechError("TARS_NAME_ALIAS must be a short plain-text pronunciation, such as Loo kah.")
        self._output = None
        self._previous_text = ""
        self.last_underflows = 0

    def begin_response(self):
        self._previous_text = ""

    def prepare_text(self, text):
        # Piper's [[IPA]] syntax is never appropriate for this provider.
        if "[[" in text or "]]" in text:
            raise CloudSpeechError("Cloud speech received Piper phoneme markup; send plain text.")
        if self.name_alias:
            return re.sub(r"(?i)\bluca\b", lambda _: self.name_alias, text)
        return text

    def _pcm_chunks(self, text):
        payload = {"text": text, "model_id": self.model_id,
                   "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                                      "style": 0.0, "use_speaker_boost": True,
                                      "speed": 1.0}}
        if self._previous_text:
            payload["previous_text"] = self._previous_text
        started, received, pending = time.monotonic(), 0, b""
        path = f"/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_24000"
        with _request(path, self._key, payload) as response:
            mime = response.headers.get_content_type()
            if mime not in {"audio/pcm", "audio/x-pcm", "application/octet-stream"}:
                raise CloudSpeechError("ElevenLabs returned an unexpected audio format; playback stopped.")
            while True:
                if time.monotonic() - started > REQUEST_DEADLINE:
                    raise CloudSpeechError("ElevenLabs speech exceeded its time limit.")
                chunk = response.read(4096)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_AUDIO_BYTES:
                    raise CloudSpeechError("ElevenLabs speech exceeded the audio size limit.")
                pending += chunk
                aligned = len(pending) - len(pending) % 2
                if aligned:
                    yield pending[:aligned]
                    pending = pending[aligned:]
            if not received or pending:
                raise CloudSpeechError("ElevenLabs returned empty or incomplete PCM audio.")

    def speak(self, text):
        """Return wait-before-playback, playback phase, and first write timestamp.

        Playback phase includes overlapping network/generation work and drain;
        it is not a measurement of pure device time or acoustic onset.
        """
        from contextlib import closing
        import sounddevice as sd

        text = self.prepare_text(text.strip())
        if not text:
            return 0.0, 0.0, None
        if len(text) > 2500:
            raise CloudSpeechError("Speech chunk exceeds 2,500 characters; shorten the reply.")
        if sys.byteorder != "little":
            raise CloudSpeechError("This PCM playback path requires a little-endian machine.")
        started = time.perf_counter()
        requested = None
        self.last_underflows = 0
        try:
            # Open the device before spending credits on generation. Reuse its
            # PortAudio handle across sentences, stopping/draining between them.
            if self._output is None:
                self._output = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1,
                                                  dtype="int16", latency="high")
            with closing(self._pcm_chunks(text)) as chunks:
                # Starting on the first small network chunk can starve the Pi's
                # speaker whenever the following chunk is delayed. Accumulate a
                # half-second lead, while retaining streaming for longer replies.
                preroll = bytearray()
                for pcm in chunks:
                    preroll.extend(pcm)
                    if len(preroll) >= PLAYBACK_PREROLL_BYTES:
                        break
                if preroll:
                    self._output.start()
                    requested = time.perf_counter()
                    self.last_underflows += int(bool(self._output.write(bytes(preroll))))
                for pcm in chunks:
                    self.last_underflows += int(bool(self._output.write(pcm)))
            self._output.stop()  # drain the final samples before listening again
            ended = time.perf_counter()
            self._previous_text = text[-500:]
            return requested - started, ended - requested, requested
        except BaseException:
            if self._output is not None:
                self._output.abort()
            raise

    def close(self):
        if self._output is not None:
            try:
                self._output.abort()
            finally:
                self._output.close()
                self._output = None


def main():
    parser = argparse.ArgumentParser(description="List or audition ElevenLabs voices for TARS.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List accessible voices; no speech generation.")
    listing.add_argument("--search", default="")
    audition = commands.add_parser("audition", help="Generate one short paid speech sample.")
    audition.add_argument("--voice-id")
    audition.add_argument("--model", choices=MODELS)
    audition.add_argument("--name-alias", help='Optional speech-only spelling, e.g. "Loo kah".')
    audition.add_argument("--test-name", action="store_true")
    configure = commands.add_parser("configure", help="Enter a key privately and verify the selected voice.")
    configure.add_argument("--voice-id", required=True)
    configure.add_argument("--save", action="store_true", help="Save the verified key and voice in this project's .env.")
    args = parser.parse_args()
    load_environment()
    if args.command == "configure":
        import getpass
        from dotenv import set_key

        key = getpass.getpass("ElevenLabs API key (hidden; Enter keeps an existing key): ").strip()
        if key:
            os.environ["ELEVENLABS_API_KEY"] = key
        voice = ElevenLabsVoice(voice_id=args.voice_id)
        available = list_voices()
        if not any(row["voice_id"] == voice.voice_id for row in available):
            raise CloudSpeechError("That voice is not in this account's available voices. Add it to My Voices first.")
        if args.save:
            env_file = Path(__file__).resolve().with_name(".env")
            set_key(str(env_file), "ELEVENLABS_API_KEY", _api_key())
            set_key(str(env_file), "ELEVENLABS_VOICE_ID", voice.voice_id)
            if os.name == "posix":
                env_file.chmod(0o600)
            print("Verified and saved ElevenLabs settings in .env. Existing unrelated entries were preserved.")
        else:
            print("Key and voice verified. Nothing saved; use --save to retain them in .env.")
        return
    if args.command == "list":
        # JSON escaping prevents remote names/labels from emitting terminal controls.
        print(json.dumps(list_voices(args.search), indent=2, ensure_ascii=True))
        return
    from contextlib import closing
    with closing(ElevenLabsVoice(args.voice_id, args.model, args.name_alias)) as voice:
        print("Generating one ElevenLabs sample; account credits apply.")
        voice.speak(NAME_TEST if args.test_name else VOICE_TEST)
        print("Audition finished. Compare Luca's first vowel, final vowel, and sentence flow.")
        if voice.last_underflows:
            print(f"Output underflows during audition: {voice.last_underflows}")


if __name__ == "__main__":
    try:
        main()
    except CloudSpeechError as exc:
        print(f"Speech unavailable: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nVoice audition stopped.")
