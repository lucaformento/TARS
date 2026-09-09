# Performance notes

Measurements from the Raspberry Pi 5 prototype during the September 8–9, 2026 development session. The working implementation was committed as [`0a69120`](https://github.com/lucaformento/TARS/commit/0a69120).

## Reusing the Piper voice

The original `speak()` path launched the Piper CLI for each reply. The replacement creates one `PiperVoice` at application startup and reuses it for WAV generation.

The comparison used Piper 1.8.0 and the same `en_US-ryan-medium` voice, texts, and default synthesis settings. Both methods were timed in the same benchmark session: one method launched a fresh subprocess; the other used the already loaded Python object. Both wrote WAV files and neither played them.

Each text/method combination received a warm-up, followed by four measured trials. Text and method order alternated across trials. Times include WAV generation and file closure; audio durations were read afterward. Resident timings exclude initial import, loading, and warm-up. “Fresh” means a new process, not an intentionally cleared OS file cache.

| Input | Fresh-process median | Resident median | Difference | Reduction |
| :--- | ---: | ---: | ---: | ---: |
| Short | 2.107 s | 0.051 s | 2.056 s | 97.6% |
| Long | 3.234 s | 1.133 s | 2.101 s | 65.0% |

The measured resident setup took **0.123 s to import Piper** and **1.828 s to load the voice**. This work moves to startup rather than disappearing.

<details>
<summary>Inputs and individual trial timings</summary>

Short input: `Test.`

Long input:

```text
Olympus Mons on Mars is about twenty two kilometers high, nearly three
times Everest. Earth has nothing on that, and honestly it is kind of
humbling to think about, especially before coffee.
```

| Trial | Short: fresh | Short: resident | Long: fresh | Long: resident |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 2.105 s | 0.054 s | 3.250 s | 1.165 s |
| 2 | 2.100 s | 0.063 s | 3.229 s | 1.117 s |
| 3 | 2.108 s | 0.048 s | 3.235 s | 1.113 s |
| 4 | 2.201 s | 0.036 s | 3.233 s | 1.149 s |

Generated durations varied: short clips ranged from 0.19 to 0.42 seconds; long clips from 9.68 to 10.09 seconds. Displayed trial values are rounded, so recomputing medians can differ slightly from the benchmark's full-precision summary.

</details>

This supports a substantial reduction in per-call WAV-generation time for the two tested inputs. It does not establish a universal speedup, isolate every startup component, or measure complete conversation latency.

## First live session with sentence streaming

| Turn | Trimmed input | Transcription | First text | First speech chunk | Estimated wait to first playback request |
| :--- | ---: | ---: | ---: | ---: | ---: |
| “Now the other side.” | 1.0 s | 7.0 s | 1.19 s | 1.73 s | 9.52 s |
| “Can you hear me?” | 0.9 s | 1.7 s | 1.68 s | 1.90 s | 4.54 s |
| Olympus Mons comparison | 3.8 s | 2.4 s | 4.54 s | 5.26 s | 9.29 s |
| Current personality settings | 1.7 s | 2.2 s | 1.05 s | 1.68 s | 4.96 s |

First-text and first-chunk times start at entry to the streamed-response routine, after transcription. The estimated wait starts when Python reads the last microphone frame classified as loud and ends immediately before launching the first `aplay` process.

**Measurement boundaries matter:**

- The estimated wait includes endpointing, transcription, waiting for text, sentence buffering, and first-chunk synthesis.
- Input buffering and background noise can shift the detected-end timestamp.
- Playback startup and the time until actual audible speech are outside that interval.
- The reported response cycle includes synthesis and playback. It is not an API-latency measurement.
- Gaps between playback calls ranged from 0.10 to 0.69 seconds. They run from one call's completion to the next request and exclude the next process/device startup.

These were four different conversational inputs and generated replies. They are observational results, not a controlled before/after test of streaming. The builder reported that conversation felt more reactive, with remaining delays and uneven delivery.

## What to measure next

1. Repeat first and subsequent transcriptions on the same recorded input. Cold initialization is a candidate explanation for the first-turn spike, not a confirmed diagnosis.
2. Compare STT configurations on identical audio, including uncommon words and commands. Record accuracy alongside time.
3. Compare a fixed reply played as a whole against sentence-by-sentence playback. Listen for pauses within the audio as well as gaps between calls.
4. Evaluate preparing audio during playback and maintaining an output stream. Removing process launches alone does not hide synthesis time or prove acoustic-onset timing.
5. Repeat live runs and report a distribution. Two sub-five-second turns are not enough to establish typical latency.
