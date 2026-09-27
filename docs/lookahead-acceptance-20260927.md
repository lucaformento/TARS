# ElevenLabs lookahead acceptance — September 27, 2026

## Scope

This acceptance check covers the installed bounded one-sentence lookahead path
on the Raspberry Pi. It verifies sentence transitions using live ElevenLabs
synthesis and the real output device, first with a fixed response and then with
a live Claude response. It does not exercise microphone capture, endpoint
detection, or Whisper.

## Installed behavior

- Sentence N+1 may download while sentence N plays.
- At most one future sentence is prepared.
- Playback remains ordered.
- Worker failures reach the caller.
- Cancellation joins the worker instead of leaving background synthesis alive.
- Piper keeps its existing path.

## Verification

The targeted installed suite passed 19 tests. The full combined Pi suite passed
101 tests after the memory installation.

Three controlled live ElevenLabs trials each played a three-sentence response:

| Trial | Transition gaps (seconds) | Output underflows | Total cycle (seconds) |
| --- | --- | --- | --- |
| 1 | 0.000, 0.000 | 0 | 7.823 |
| 2 | 0.000, 0.000 | 0 | 7.730 |
| 3 | 0.001, 0.000 | 0 | 7.703 |

Across six transitions, the median gap was 0.000 seconds and the maximum was
0.001 seconds. All output remained in order, and all trials reported zero
underflows. Individual synthesis downloads took 1.148 to 1.505 seconds and were
successfully hidden behind playback after the first sentence.

A separate live integration trial streamed a real `claude-sonnet-4-6` response
into ElevenLabs. It produced three ordered sentences, transition gaps of 0.001
and 0.000 seconds, and zero output underflows. First model text arrived in 1.210
seconds, the first complete sentence was ready in 1.511 seconds, and the full
synthesis-and-playback cycle took 12.207 seconds. The ordinary test question did
not modify the private memory store.

## Result

The bounded lookahead implementation passes its Pi acceptance target of a
median gap below 0.15 seconds, a maximum gap below 0.30 seconds, correct sentence
order, and zero output underflows. First-sentence response latency remains a
separate concern because no earlier playback exists to overlap it.
