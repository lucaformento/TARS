<p align="center">
  <img src="docs/assets/tars-banner.svg" alt="TARS — Built to help he who wants to explore and conquer." width="100%">
</p>

# TARS

A voice-driven conversational robot inspired by TARS from *Interstellar*, running
on a Raspberry Pi 5. Wake-word detection and speech recognition run on the
device; response generation streams from the Anthropic API; speech is
synthesized sentence-by-sentence through ElevenLabs (or local Piper) with
one-sentence lookahead. The robot has application-owned personality state and a
persistent, user-correctable memory. The articulated body is the next phase.

**Status:** working voice prototype, validated live on the Pi. 113 offline tests.

## System overview

```mermaid
flowchart LR
    mic["USB mic · 16 kHz<br/>PortAudio callback<br/>bounded frame queue"] --> wake["openWakeWord<br/>custom 'Hey TARS' + VAD"]
    wake --> cap["Endpointing<br/>RMS · pre-roll · tail pad"]
    cap --> stt["faster-whisper base<br/>CPU INT8 · local"]
    stt --> brain["Brain<br/>dials · memory · history"]
    brain --> llm["Anthropic API<br/>streamed text"]
    llm --> split["Sentence splitter"]
    split --> worker["Lookahead worker<br/>downloads sentence N+1"]
    worker --> play["Main thread<br/>owns output device<br/>plays sentence N"]
    play -. "restart capture,<br/>follow-up window" .-> cap
```

One turn:

1. **Wake.** The input stream runs a PortAudio callback that enqueues 80 ms
   frames into a bounded queue (3 s). The main thread runs openWakeWord on each
   frame (threshold 0.5, VAD-gated at 0.5, 2 s post-sleep cooldown).
2. **Capture.** An RMS endpointer, calibrated at startup to `max(3 × ambient
   median, 150)`, keeps 0.5 s of pre-roll, ends after 1.2 s of silence, pads a
   0.5 s tail, rejects bursts with under 0.4 s of loud frames, and caps an
   utterance at 30 s. Queue drops and PortAudio overflows are counted per
   capture; a damaged capture is never transcribed — TARS asks for a repeat.
3. **Transcribe.** Capture is **stopped** for STT, generation, and playback.
   This prevents USB input overruns and speaker echo, at the cost of
   interruption (see [limitations](#limitations)).
4. **Generate.** The brain rebuilds the system prompt on every request from
   current dial values, a bounded memory block, and an optional one-turn
   control event, then streams text deltas.
5. **Speak.** Deltas are split into sentences. A worker thread downloads
   sentence N+1 as complete PCM while the main thread plays sentence N; it never
   prepares two ahead. After the reply, capture restarts with a clean buffer and
   an 8 s follow-up window (no wake word needed).

## Design decisions

**State lives in the application, not the model.** Personality dials, presets,
memory, and conversation history are Python state. The model receives them
fresh each request; commands are parsed locally and reach the model as a
one-turn control event, while history keeps the user's literal words.

**Full-sentence buffering before playback.** Each ElevenLabs sentence is fully
downloaded before the device write, so network jitter can never underflow the
output. Live data showed streaming within a sentence would gain little: across
nine replies, the full sentence arrived on average 0.06 s after its first PCM
byte, so time-to-first-byte dominates.

**Single owner for the audio device.** Only the main thread opens, writes,
stops, or aborts the output stream; the lookahead worker only downloads. An
earlier version let the worker abort the device after a failed download. Under
a deliberate 20 s HTTPS outage this blocked the main thread inside PortAudio
`poll()` indefinitely, unkillable by Ctrl+C (confirmed with a gdb backtrace).
Moving all device control to one thread fixed it; the same outage test now
recovers cleanly.

**Fail the turn, not the process.** Any error inside a turn — Anthropic
timeout or overload, ElevenLabs failure, PortAudio playback error, Whisper
failure — ends that turn, prints a one-line reason, and returns to wake-word
listening with capture restarted. The one fatal condition is a microphone
stream that cannot restart, because continuing would leave TARS deaf.

**History stays well-formed under failure.** A request that fails before any
text removes its orphaned user turn; an interrupted reply is kept as a
complete user/assistant pair. History is capped at 24 messages and pruned in
pairs.

## Memory

A single human-readable JSON file (`memory.json`, git-ignored), designed so a
guess can never silently become a fact.

| Source | How it is created | Prompt label |
| :--- | :--- | :--- |
| Explicit | "remember that I prefer PETG", "remember I work nights" | `CONFIRMED` |
| Inferred | Narrow first-person patterns ("I like…", "I'm working on…", "my X is…"); questions are never mined | `UNCONFIRMED` |
| Running joke | "remember that" right after a TARS line | `RUNNING JOKE` |

- **Retrieval is bounded:** confirmed before inferred, newest first, at most 20
  entries and 1,200 characters; entries are capped at 240 characters and the
  store at 500 (oldest inferred entries are evicted first, explicit ones never).
- **Corrections:** "forget that I like PETG" deletes one unambiguous match;
  ambiguous requests delete nothing and ask for specifics. Re-stating an
  inferred entry upgrades it to `CONFIRMED`.
- **Durability:** writes go to a temp file, `fsync`, then atomic `os.replace`.
  A failed write leaves memory and disk unchanged, and TARS is told not to claim
  success. An unparsable file is quarantined, never overwritten; an unreadable
  one disables writes.
- **Prompt hygiene:** memory text is framed as data, not instructions, and
  whitespace is collapsed so an entry cannot forge additional lines.
- Whisper often transcribes the wake phrase; one leading "Hey TARS" is stripped
  before command parsing.

## Personality

Four dials (0–100) are injected as style guidance, never recited unless asked:
humor (75), sarcasm (60), honesty (90), intellect (50). Commands are parsed
locally from text or speech, including spoken numbers: `set humor to ninety`,
`buddy mode`, `know-it-all`, `what are your settings`, `reset`. Presets,
resets, and setting queries must be the whole utterance, and dial changes need
an explicit "set/change/turn … to N" form, so ordinary speech that merely
mentions "settings" or "humor" does not trigger a control event.

## Measured performance

Live session on the Pi (September 30, 2026; ElevenLabs, `claude-sonnet-4-6`):

| Metric | Result |
| :--- | :--- |
| Turns / wakes | 7 turns across 2 wakes; both wakes on the first attempt |
| Input overflows / output underflows | 0 / 0 |
| Gap between spoken sentences | 11 of 13 at 0.00 s; max 0.05 s |
| Long utterance | 21.7 s clip transcribed in full, ended on silence |
| End of speech → first playback request | 4.9–8.5 s |
| Network-outage recovery (20 s HTTPS block mid-reply) | reported, slept, woke and answered normally, clean exit |

First-response latency breakdown: ~1.2 s endpoint silence, 1.6–4.5 s Whisper
(fit: **1.40 s fixed + 0.14 s per second of audio** — the fixed part is the
encoder's padded 30 s window), ~1.3 s to Claude's first complete sentence, and
1.2–1.5 s ElevenLabs time-to-first-byte. The reduction plan, ranked by expected
savings and risk, is in [docs/latency-proposal.md](docs/latency-proposal.md).

Earlier, keeping the Piper voice resident instead of spawning it per reply cut
synthesis from ~2.1 s to 0.05 s for a short line
([performance notes](docs/performance.md)).

## Privacy boundaries

- **Microphone audio never leaves the Pi.** Wake detection and STT are local.
- The transcript, dials, and the bounded memory block go to Anthropic.
- Only generated reply text (plus up to 500 characters of same-reply context)
  goes to ElevenLabs. The API key is never forwarded on redirects, and errors
  are sanitized before display.
- Opt-in capture diagnostics (`--capture-diagnostics`) refuse any path inside
  the repository and write WAV/JSON files with `0700`/`0600` permissions.

## Hardware and stack

| Layer | Implementation |
| :--- | :--- |
| Compute | Raspberry Pi 5, 8 GB, active cooling; Debian 13 (aarch64); Python 3.13 |
| Audio | USB microphone (matched by device name) and speaker; PortAudio via `sounddevice`, 200 ms latency both directions |
| Wake word | openWakeWord 0.4.0, custom "Hey TARS" ONNX model; 9/9 detections in clean Pi trials |
| STT | faster-whisper 1.2.1, `base`, CPU INT8, English |
| LLM | Anthropic SDK 0.111.0; default `claude-sonnet-4-6`; 20 s timeout, 1 retry; 160 output tokens for voice |
| TTS | ElevenLabs `eleven_multilingual_v2`, 24 kHz s16le PCM; Piper 1.8.0 as the local fallback |

## Repository layout

| Path | Responsibility |
| :--- | :--- |
| [`tars_voice.py`](tars_voice.py) | Voice front end: capture, endpointing, wake/follow-up loop, sentence lookahead, recovery, diagnostics |
| [`cloud_speech.py`](cloud_speech.py) | ElevenLabs client: bounded PCM download, playback, cancellation; voice list/audition/configure CLI |
| [`brain.py`](brain.py) | Request construction, streaming, history consistency, model selection |
| [`memory.py`](memory.py) | Memory store, command parsing, inference, bounded retrieval |
| [`personality.py`](personality.py) | Pure prompt building and dial/preset command parsing |
| [`tars.py`](tars.py) | Terminal front end for the same brain |
| [`tests/`](tests) | 113 offline tests with fake devices, network, and models |
| [`docs/`](docs) | Decision record, measurements, and design notes |

## Running it

Developed on Python 3.13; requires an Anthropic API key. The terminal
interface needs no audio hardware:

```bash
git clone https://github.com/lucaformento/TARS.git && cd TARS
python3 -m venv venv && source venv/bin/activate
pip install anthropic==0.111.0 python-dotenv
echo "ANTHROPIC_API_KEY=..." > .env
python tars.py                      # optional: --brain-model claude-sonnet-5
```

The voice front end additionally needs `numpy`, `sounddevice` (with PortAudio),
`faster-whisper`, `openwakeword==0.4.0`, `piper-tts`, and locally provisioned
models (not distributed; see [voice setup](docs/voice-setup.md)). For
ElevenLabs, configure a key and voice with `python cloud_speech.py configure
--voice-id VOICE_ID --save` ([cloud voice](docs/cloud-voice.md)).

```bash
python tars_voice.py --tts elevenlabs                # converse
python tars_voice.py --tts elevenlabs --diagnostics  # per-turn timing, overflow, underflow report
python cloud_speech.py audition --voice-id VOICE_ID  # one paid sample
```

| Variable | Purpose |
| :--- | :--- |
| `ANTHROPIC_API_KEY` | Required |
| `ANTHROPIC_MODEL` | Optional; one of `claude-sonnet-4-6`, `claude-sonnet-5`, `claude-haiku-4-5-20251001` |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Required for `--tts elevenlabs` |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` (default) or `eleven_flash_v2_5` |

Tests (no network, audio device, or API key needed):

```bash
python -m unittest discover -s tests
```

## Limitations

- **First-response latency (4.9–8.5 s)** is the main open problem; see the
  latency plan above.
- **No barge-in.** Capture is paused while TARS thinks and speaks. Interruption
  and a wake acknowledgment both need continuous capture with speaker-echo
  control.
- The wake word accepts some near phrases ("hey stars"); this was a deliberate
  trade-off for recall.
- Whisper `base` can mishear short, quiet phrases.
- A bare "forget that" removes the most recently updated memory; a version
  scoped to the current session is in review.
- Conversation history is process-local; only memory persists across restarts.
- Model and device paths are configured for this Pi; a portable install script
  and lockfile do not exist yet.
- The voice is a stock ElevenLabs voice for now. The replacement is being
  created with text-prompted voice design — a new synthetic voice, not a clone
  of any real person.

## Roadmap

- [x] Wake word, follow-up loop, and streamed sentence speech
- [x] One-sentence lookahead with zero-gap transitions
- [x] Persistent memory with confirmed/inferred provenance
- [x] Turn-level failure recovery, verified under a live network outage
- [ ] First-response latency: same-clip `tiny.en` vs. beam-size benchmark, endpoint tuning, Flash TTS for sentence one
- [ ] Designed TARS-style voice and optional "machine body" output filter
- [ ] Model-judged memory: let the model decide what is worth keeping
- [ ] Continuous capture with echo control for barge-in
- [ ] Articulated, 3D-printed body with servo control

---

Built by [Luca Formento](https://github.com/lucaformento). An independent project inspired by *Interstellar*.
