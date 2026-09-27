# Conversation reliability investigation

Updated September 27, 2026. Current status: **the 200 ms input/output buffer
change and bounded ElevenLabs lookahead both passed their recorded Pi trials**.
The earlier failures below are historical. The accepted trials establish the
current baseline, while normal use still needs to watch for intermittent USB
audio failures.

## Evidence from the Pi

- Three isolated Roger auditions after complete-sentence buffering sounded clear
  and pronounced Luca correctly. All reported zero output-underflow flags.
- The first full conversation still lost microphone input on the opening turn.
  Whisper returned "I need your help" for "How's your day going?". The capture
  reported an overflow. This is an upstream transcript failure; Claude answered
  the text it received. The exact lost audio cannot be recovered from the log.
- A long turn stopped at exactly 15.0 seconds with `endpoint max_utterance`.
  This established the recorder's duration limit as that cutoff's cause.
- The old local parser treated any occurrence of "settings" as a request for all
  personality dials. This created a system control event even for ordinary speech.
- Kernel logs showed repeated xHCI buffer-overrun events. They locate a USB audio
  problem but do not establish whether its cause is software, configuration,
  driver, cable, or hardware. Earlier attribution to a bad microphone was too strong.

## Second repair and its failed acceptance trial

The second candidate added callback microphone buffering, rejected captures with
reported lost samples, extended the utterance limit to 30 seconds, and narrowed
personality commands. It added optional response-model selection and a more
restrained voice prompt. Installation and 43 offline tests passed on the Pi.

The following real conversation did **not** pass acceptance:

- The opening question and ordinary settings sentence transcribed correctly.
  The settings sentence no longer caused a personality-dial recital.
- Two later captures reported input overflow and were rejected. Asking Luca to
  repeat protects against a corrupt transcript but does not repair lost speech.
- The first response reported one output-underflow flag, the second zero, and
  the final response two. These are flags returned by blocking writes, not an
  exact count or location of audible glitches.
- Estimated detected-speech-end to playback request was 5.86, 6.07, and 9.62
  seconds. These measurements are not acoustic-onset measurements.
- Gaps between sentence playback requests were about 1.04–1.67 seconds. The
  current synchronous loop downloads the next sentence only after playback of
  the previous sentence, which adds a wait for each separate speech request.
- Claude first-text times were 1.17, 0.90, and 5.00 seconds. Provider latency can
  contribute to waiting; changing the response model cannot recover lost audio.
- Luca found the entire experience unusable: lost speech, long waits, and
  unreliable playback. The longer-utterance change still needs a clean Pi trial.

## Completed experiment: audio without model processing

`tools/audio_isolation.py` ran on September 17 using the Pi's existing virtual
environment with TARS stopped. It recorded counters only, discarded microphone
audio, and generated quiet test tones locally. The downloaded report
`tars-audio-isolation-pmub14wa.txt` includes device profiles and kernel events
with timestamps for matching them to individual phases.

The test compares microphone-only capture, playback alone, playback while a
microphone handle remains open but stopped, and ALSA utilities without PortAudio.
It also compares the driver's `high` latency setting with an explicit 200 ms
request. The earlier Pi audit reported only 21–35 ms for its default high values;
these reported defaults are not measurements of a live stream's actual buffer.
The diagnostic therefore records the actual stream latency too.

The upstream documentation explicitly says `high` does not guarantee stable
audio and notes that simultaneous streams on one device are implementation
dependent. These are hypotheses to test, not established root causes.
See [sounddevice stream documentation](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html).

### Results

- Microphone-only capture with `high` reported actual latency 80 ms and one
  input overflow in 20 seconds. Requesting 200 ms yielded actual latency 240 ms
  and zero overflows in the following 20 seconds.
- Isolated playback reported zero underflow flags in every phase. Actual latency
  was about 21 ms with `high` and 192 ms with the explicit 200 ms request.
- ALSA playback through both tested routes and 44.1 kHz stereo capture completed
  without reported errors. Audible tone quality was not independently confirmed.
- Kernel xHCI events occurred during the first microphone phase and near stream
  boundaries. Second-resolution timestamps do not assign every boundary event
  unambiguously to one phase. These events do not prove defective hardware.
- The USB descriptor lists 16 kHz and 24 kHz among supported rates, so its default
  44.1 kHz profile alone is not evidence that the application rates are invalid.

### Subsequent conversation result shared September 18

Luca's shared Claude conversation records applying `latency=0.2` to the Pi's
microphone and ElevenLabs output streams, then three conversation turns with
zero input overflows and zero output underflows. This result is reported from
the shared conversation; no new direct Pi session was run for this update.
The local working files now match those two settings, with 49 offline tests
passing. Historical repair bundles retain their original contents and hashes.

The shared timing review gave 5.26 seconds from detected speech end to the first
playback request, excluding acoustic speaker startup. It reported later sentence
gaps of 1.19, 1.27, 1.44, and 1.67 seconds in the synchronous implementation.
The September 27 bounded-lookahead change now downloads sentence N+1 while
sentence N plays. Three controlled ElevenLabs trials kept all six transitions
at 0.001 seconds or less with zero output underflows. A live Claude-to-ElevenLabs
trial also preserved three-sentence order, reported 0.001- and 0.000-second
transitions, and had zero output underflows. See
[lookahead acceptance](lookahead-acceptance-20260927.md) for the measurements.
Lookahead does not remove the initial response delay because no earlier audio
exists to cover the first model and speech request.

Retain the successful buffer configuration and Roger voice. A `tiny.en` trial
was suggested, but no executed switch is established; local STT remains `base`.
Changing microphone capture behavior for HUH and interruption is separate work.

## Engineering lessons

- Unit tests establish software behavior; real hardware acceptance is separate.
- Label a candidate as a trial until the complete user experience passes.
- Change one cause at a time and keep the same input for comparisons.
- Rejecting damaged input is recovery, not a reliability fix.
- Fully buffering downloaded speech addresses network starvation but cannot
  guarantee a healthy local audio device.
- Preserve measurements and distinguish observations, hypotheses, and verified
  causes. Kernel errors alone do not justify replacing hardware.
