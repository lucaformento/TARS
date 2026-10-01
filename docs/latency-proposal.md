# First-response latency proposal

Last updated: October 1, 2026.

## Goal and baseline

Reduce the delay from Luca's last spoken word to TARS's first audible response
without giving back the accepted capture reliability, transcription quality, or
H2 audio-device ownership fix.

The September 30 live session measured **4.9–8.5 seconds** from the last loud
microphone frame to the first playback request:

| Stage | Observed time | Share of the opportunity |
| --- | ---: | --- |
| End-of-speech confirmation | about 1.2 s | Fixed by `SILENCE_END`; predictable |
| faster-whisper `base` | 1.6–4.5 s | Largest and most variable stage |
| Claude to first complete sentence | about 1.3 s | Requires a sentence boundary before TTS starts |
| First ElevenLabs sentence download | 1.2–1.5 s | The whole first sentence is buffered before playback |

These times overlap only after the first sentence. Bounded lookahead has removed
the later inter-sentence gaps but cannot hide any of the four stages above.

Treat all savings below as **test hypotheses**, not promises. The Pi measurements
decide. Use at least ten turns covering short commands, normal questions, a
sentence with an internal pause, Luca's name, numbers, and one 15–20 second
utterance. Warm Whisper before collecting results. Change one variable at a
time and compare median, slowest turn, transcript errors, input overflows, output
underflows, and wake behavior.

The nine live replies also answer two design questions before any new code:

- A least-squares fit gives **STT ≈ 1.40 s fixed + 0.141 s per second of
  audio**. The fixed component is dominated by the encoder's padded 30-second
  window, so decoder beam size cannot remove most of it.
- First-sentence `download_s - time_to_first_pcm_s` averaged **0.062 s** and
  never exceeded **0.090 s**. Progressive playback cannot recover enough time
  to justify reopening an H2-class real-time audio risk, so that option is
  rejected. The TTS lever is time to first byte, especially Flash.

## Ranked options

### 1. Benchmark `tiny.en` and `base` with `beam_size=1` together

- **Expected saving:** `tiny.en` has the stronger hypothesis because its smaller
  encoder attacks the roughly 1.40-second fixed cost every turn pays.
  `beam_size=1` can affect only a fraction of the roughly 0.3–1.0-second
  audio-dependent portion of typical 2–7 second questions. Exact Pi savings
  remain unestablished until the same-clip benchmark.
- **Risk:** medium for beam 1 and medium to high for `tiny.en`. Greedy decoding
  can choose a worse ambiguous phrase, while the smaller model can lose quiet,
  short, or unusual words. The live session already showed `base` mishearing one
  quiet 1.4-second phrase.
- **Cost:** no API cost. Capture one approximately ten-clip private set once,
  then run all model and beam combinations offline.
- **Measurement:** use `--capture-diagnostics` with Luca to save clips outside
  the repository. Benchmark warmed `base`/beam 5, `base`/beam 1,
  `tiny.en`/beam 5, and `tiny.en`/beam 1 against those exact WAVs. Record `stt`
  duration, exact transcript, clip length, and warm-up separately. Rank by
  accuracy first, then median and slowest decode time.
- **Robot/Luca:** Luca is needed once to record and approve the private clip set.
  The repeated STT experiments do not need him or the robot afterward.

This joint benchmark replaces treating beam size as the clear first choice.
faster-whisper documents a default `beam_size=5`; its maintainer has suggested
`beam_size=1` for short CPU transcriptions, with an accuracy tradeoff. OpenAI's
cross-hardware table rates `tiny` at about 10x and `base` at about 7x relative
to `large`, while warning that real speed varies by hardware. See the
[faster-whisper transcription options](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py),
the maintainer's [short-audio guidance](https://github.com/SYSTRAN/faster-whisper/discussions/169),
and the [Whisper model table](https://github.com/openai/whisper/blob/main/README.md#available-models-and-languages).

### 2. Reduce `SILENCE_END` in frame-sized steps

- **Expected saving:** **0.24 seconds** at 0.96 s, or **0.40 seconds** at 0.80 s,
  versus the current 1.20 s. The recorder advances in 80 ms frames, so values
  should remain multiples of 0.08 s.
- **Risk:** medium. A lower value can split a thoughtful pause into two turns or
  end before a quiet final word. The accepted 21.7-second utterance must remain
  intact.
- **Cost:** no API or hardware cost.
- **Measurement:** test 1.20 s, 0.96 s, then 0.80 s. `--diagnostics` already
  reports clip duration, endpoint reason, transcript, overflows, and detected
  end to first playback. Reject any setting that increases premature endpoints,
  `too_short` captures, or missing final words.
- **Robot/Luca:** yes. Natural pauses and room noise are the acceptance test.

Start at 0.96 s. Move to 0.80 s only if ten varied turns remain clean.

### 3. Use `eleven_flash_v2_5` for sentence one only

- **Expected saving:** test hypothesis of **0.2–0.8 seconds** from the current
  1.2–1.5 second first-sentence TTS stage. Network time and buffering remain, so
  the advertised model inference latency is not the user-visible result.
- **Risk:** medium. Sentence one may sound slightly different from later
  Multilingual v2 sentences. Flash also handles number, date, and currency
  normalization less reliably, so the LLM must speak those naturally in words.
- **Cost:** lower than the current model for that sentence. ElevenLabs lists
  Flash at $0.05 per 1,000 API characters versus $0.10 for Multilingual v2, or
  0.5 versus 1 self-serve credit per character.
- **Measurement:** after Luca selects the new designed voice, first verify it
  is available with Flash. Then A/B identical first-sentence text and settings. Compare
  `Cloud sentence 1: first PCM`, `download`, and `Detected end -> first playback
  request`; keep underflows at zero. Then compare voice continuity into sentence
  two by ear.
- **Robot/Luca:** yes. The A/B uses paid ElevenLabs calls and must use the new
  designed voice Luca selects, not Roger. Luca must accept voice continuity
  and number pronunciation.

ElevenLabs describes Flash v2.5 as roughly 75 ms model inference, excluding
application and network latency, and as 50% cheaper per API character. See the
[model overview](https://elevenlabs.io/docs/overview/models) and
[API pricing](https://elevenlabs.io/pricing/api).

### 4. Require a short, useful first sentence

- **Expected saving:** roughly **0.1–0.6 seconds** across Claude sentence
  completion and first-sentence TTS. The current voice prompt allows 15–35 words;
  a 4–8 word lead sentence should reach punctuation and finish synthesis sooner.
- **Risk:** low technical risk, medium character risk. A forced opener can sound
  repetitive, shallow, or like filler. It must contain useful content rather
  than “Sure” or “Let me explain.”
- **Cost:** no added API cost and potentially fewer ElevenLabs characters.
- **Measurement:** compare `Request -> first text`, `first complete sentence`,
  sentence-one `download`, and end-to-first-playback across the same prompts.
  Review whether the full answer remains natural and accurate.
- **Robot/Luca:** yes for character acceptance; timing can first be collected
  without changing audio hardware.

Suggested prompt direction: “Lead with one useful sentence of 4–8 words, then
add detail if needed. Never use a filler acknowledgment as the first sentence.”

### 5. Add a cached local acknowledgment sound

- **Expected saving:** **zero actual answer latency**. It can reduce perceived
  silence by providing feedback immediately after endpoint detection, masking
  the 1.6–4.5 second STT stage and later network work.
- **Risk:** low engineering risk, medium annoyance risk. It must be brief,
  nonverbal, quiet enough not to feel like an interruption, and must not imply
  the request succeeded.
- **Cost:** no per-turn API cost if a local PCM asset is cached.
- **Measurement:** route the acknowledgment through the existing diagnostics so
  `first playback request` represents the acknowledgment, while retaining the
  existing Claude first-text and cloud-sentence timings for the real answer.
  The answer's measured latency should remain unchanged; acceptance is Luca's
  perceived responsiveness over repeated turns.
- **Robot/Luca:** required for volume, character, and annoyance acceptance.

This is complementary feedback, not a substitute for reducing actual latency.

## Recommended experiment order

1. With Luca present once, record approximately ten private diagnostic clips
   outside the repository, covering short, quiet, long, numeric, and named
   phrases.
2. Offline, benchmark the same clips across `base` and `tiny.en` at beam 5 and
   beam 1. Choose on transcript accuracy first and latency second.
3. Live-test `SILENCE_END=0.96`; move to 0.80 only if natural pauses survive.
4. After Luca chooses the new designed voice, verify Flash support and A/B Flash
   versus Multilingual v2 for sentence one using that voice.
5. Test the short-useful-first-sentence prompt.
6. Decide on a cached acknowledgment sound as a character choice.

Do not build progressive first-sentence playback. The live data caps its gross
benefit at 0.090 seconds while reintroducing network-fed real-time playback risk.
Set an overall latency target only after the same-clip STT benchmark and the new
voice's Flash A/B establish what the Pi can actually save.
