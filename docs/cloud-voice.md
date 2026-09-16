# ElevenLabs voice audition and optional runtime

Luca approved using paid online speech if it sounds better. The repository now
has an **opt-in** ElevenLabs engine. Piper remains the default while the new
voice is auditioned. ElevenLabs account access, a successful cloud sample, and
playback on the Pi are separate checks; local automated tests do not establish
how the voice sounds or how quickly the Pi responds.

## What to choose

Choose a voice in your ElevenLabs account's voice library by listening to its
preview, then add it to **My Voices** and copy its voice ID. There is no hardcoded
voice ID in this integration. Start with a clear, conversational male voice and
judge it on normal sentences as well as “Luca.”

The default model is `eleven_multilingual_v2`, selected for quality and consistent
delivery. `eleven_flash_v2_5` is also supported for a later speed comparison.
Provider latency estimates exclude application/network delay and are not TARS
measurements. See the [official model guide](https://elevenlabs.io/docs/overview/models).

## Set up on the Pi

Stop the running TARS process with Ctrl+C, then run these in the SSH terminal:

```bash
cd ~/TARS
git pull --ff-only origin main
./venv/bin/python cloud_speech.py configure --voice-id YOUR_VOICE_ID --save
```

Replace `YOUR_VOICE_ID` with the ID copied from your account. The helper asks for
the API key with input hidden; do not put it in the command or send it in chat.
It verifies account voice access using a read-only request, then saves only
`ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` in the existing, Git-ignored `.env`.
Other entries, including the Anthropic key, are preserved. On the Pi, `.env`
permissions are set to owner read/write. Omit `--save` to verify without writing.
An existing exported key takes precedence when loading `.env`; the helper lets
you press Enter to retain it.

Use a restricted API key with Text to Speech permission and voice-read access.
Choose a credit cap in the account dashboard. A voice being listed does not
guarantee that the account plan permits generation with it; the audition checks
that next. The script does not purchase a plan or add credits.

List available voices later without generating speech:

```bash
./venv/bin/python cloud_speech.py list
./venv/bin/python cloud_speech.py list --search deep
```

This needs no new ElevenLabs SDK. HTTP uses Python's standard library; playback
uses the existing `sounddevice`/PortAudio installation and setup uses the existing
`python-dotenv` dependency. Cloud auditions do not import Piper, openWakeWord,
faster-whisper, or the Anthropic client. The normal conversation still uses the
existing microphone, wake model, STT, and brain.

## Hear the name and a normal reply

Each audition generates fresh speech and uses account credits:

```bash
./venv/bin/python tars_voice.py --tts elevenlabs --test-name
./venv/bin/python cloud_speech.py audition
```

The first says “Luca. I'm listening, Luca.” The second uses a short conversational
reply. Start with the ordinary spelling **Luca**. If it is still wrong, compare
one speech-only alias:

```bash
./venv/bin/python tars_voice.py --tts elevenlabs --test-name --name-alias "Loo kah"
```

Listen for a clear “oo,” first-syllable emphasis, and the final “ah.” An alias can
also introduce awkward spacing, so do not call it a fix until Luca approves what
he hears. `--name-alias` only affects spoken text; transcripts and conversation
history retain the original spelling. An approved alias can be retained with
`TARS_NAME_ALIAS='Loo kah'` in `.env`. Set it to an empty value to restore the
ordinary spelling. No Piper `[[phoneme]]` markup is sent to ElevenLabs.

The earlier stress-marker edit in Piper did **not** establish the cause of the
mispronunciation or a successful repair. eSpeak can place the stress mark before
the vowel in its phoneme representation. Keep that past explanation as an
unverified diagnosis; the audible result is the acceptance check.

After selecting a voice and pronunciation:

```bash
./venv/bin/python tars_voice.py --tts elevenlabs
```

For a model comparison without changing saved settings:

```bash
./venv/bin/python cloud_speech.py audition --model eleven_flash_v2_5
```

To return to the existing local voice:

```bash
./venv/bin/python tars_voice.py --tts piper
```

## Runtime behavior and measurements

The API returns mono, signed 16-bit little-endian PCM at 24 kHz. TARS writes it to
a sounddevice output stream as bytes arrive, using the default PortAudio output
device. No temporary MP3 decoding or `aplay` process is needed for this engine.
The device handle is reused across sentences; each sentence drains before the
next one starts. A supported PortAudio output device is required and may differ
from the ALSA `default` route used by Piper.

Response text still arrives from the brain in sentence chunks. Network reads and
playback within a sentence overlap, but the next sentence is processed only when
the current sentence finishes. This is not the previously tested HUH callback
player, does not add echo cancellation, and does not add interruption while TARS
speaks. Internet stalls may produce audible gaps. Microphone behavior and the
existing follow-up timing are unchanged.

Add `--diagnostics` to see measurements. For cloud speech, **wait before playback**
includes connection, synthesis, and initial buffering until the first PCM write.
**Stream/playback phase** includes continued generation/download, audio writes,
and final drain. Neither is a pure provider-latency measurement. Output
underflows are counted. Historical Piper timings do not measure this new path.

Requests use a 15-second socket timeout, a 90-second elapsed check between reads,
and a 120-second PCM byte limit; a blocked read can last until the socket timeout.
Replies over 2,500 characters per chunk are rejected. HTTP failures are summarized
without exposing provider response bodies or keys. No automatic paid retries
occur. A failed reply returns the conversation to sleep with a visible error.
Ctrl+C closes the HTTP response and audio stream.

## Data sent online

When `--tts elevenlabs` is selected, TARS sends each generated reply chunk and up
to 500 characters of the preceding chunk from that same reply to ElevenLabs for
speech continuity. It does not send microphone audio, raw recordings, or the
whole conversation history to this speech service. Replies can contain personal
information from the conversation. ElevenLabs handles this text and generated
audio under the account's service/data settings; this integration does not claim
zero retention. The Anthropic brain's existing text requests remain separate.

No ElevenLabs requests occur in Piper mode. Credentials are sent only to the
fixed `https://api.elevenlabs.io` host; redirects are rejected.

## Checks completed

The mock-network/audio tests cover split PCM samples, empty/truncated or wrong
format responses, API error redaction, limits, cancellation cleanup, device
reuse, voice pagination, optional aliases, and name auditions that skip the
other speech models. Run without API credentials or hardware:

```bash
python -m unittest discover -s tests -v
```

Real ElevenLabs generation, pronunciation, output routing, and Pi latency still
need an audible hardware audition before marking the new voice accepted.

Protocol references, checked September 15, 2026:

- [Stream speech](https://elevenlabs.io/docs/api-reference/text-to-speech/stream)
- [List voices](https://elevenlabs.io/docs/api-reference/voices/search)
- [Pronunciation dictionaries and aliases](https://elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/pronunciation-dictionaries)
