# Project state

Last updated: September 30, 2026.

This file is the canonical record for decisions, known problems, parked work,
and the next practical actions. Update it in the same commit whenever a test or
implementation changes one of those facts. GitHub `main` records published
history; this document records the installed accepted baseline and its remaining
acceptance checks. Detailed measurements stay in the linked technical notes.

**Current acceptance status:** the Pi keeps the tested 200 ms microphone and
ElevenLabs output settings. Bounded one-sentence ElevenLabs lookahead is now
installed and passed three controlled live Pi trials: all six inter-sentence
transitions were 0.001 seconds or less and all three trials reported zero output
underflows. A subsequent live Claude-to-ElevenLabs trial produced three ordered
sentences with 0.001 and 0.000-second transitions and zero underflows. The
corrected persistent-memory candidate is also installed. Its 101-test combined
Pi suite passed, followed by a disposable restart-level save, inference, labeled
recall, and forget acceptance test. The test left the real memory store empty.
Turn-failure recovery commit `7161413` and the H2 audio-device ownership repair
`a2a89dd` are installed on Pi `main`, published, and in sync with `origin/main`;
the installed suite passes all 113 offline tests. A September 30 live session
completed seven turns across two first-try wakes with zero input overflows and
zero output underflows. All seven captures ended on silence. Eleven of 13
sentence transitions were 0.00 seconds; the other two were 0.04 and 0.05
seconds. A 21.7-second utterance was transcribed in full and ended on silence,
closing the former 15-second cutoff check. One quiet 1.4-second phrase was
misheard by Whisper `base`, which Luca accepts as a model limitation rather
than capture loss. The first network-failure test permanently hung PortAudio
after the lookahead worker aborted the main thread's output device. The retest
on `a2a89dd` reported the speech failure, returned to sleep, woke on the first
try, answered normally with zero overflows and underflows, and stopped cleanly.
See [lookahead acceptance](lookahead-acceptance-20260927.md) and the
[conversation reliability investigation](voice-investigation.md).

## Decided

### Hey TARS wake word

- The original `candidate-20260913` model remains installed at threshold 0.5,
  with VAD gating at 0.5 and a two-second post-sleep cooldown.
- The intended speaker produced nine detections in nine clean Pi trials.
- Luca accepts occasional activations from close phrases such as “hey stars,”
  “hey cars,” and “hey Jarvis” so the project can proceed.
- Synthetic retraining is closed. Reopen it only if normal use shows a material
  problem. See the [wake-word decision](wake-word.md).

### Wake acknowledgment

- TARS should say **HUH!** on a randomly selected second or third actual wake.
  Every actual wake advances the counter, including wakes with a silent
  acknowledgment slot. Conversational follow-ups do not advance it.
- The Pi test validated the two/three-wake scheduler and nonblocking cached-PCM
  playback path, with estimated output-device starts of roughly 46–62 ms and no
  output underflows.
- The tested Piper acknowledgment is not approved for integration. It sounded
  too robotic, and speaker output was sometimes captured and transcribed as
  “Oh.” Keep continuous microphone capture; do not replace it with blocking
  playback followed by a microphone flush, which can discard the user's first
  words.

### Current speech voice

- The community `TARS.onnx` Piper model remains installed as the code default,
  with its `neutral` speaker. Current Pi conversation trials select ElevenLabs
  Roger explicitly. Each voice object is loaded once at startup.
- This is the installed prototype voice, not a final publication decision. The
  reviewed model package has no model card or documented training provenance.
  Verify its source and reuse terms, or replace it with a documented voice,
  before distributing the model or presenting the voice as a portfolio asset.
- Luca rejected the Piper voice and its pronunciation of his name, and approved
  paid online speech. He accepted Roger's pronunciation; retain that voice while
  preserving the accepted audio baseline.
- An opt-in ElevenLabs runtime (`--tts elevenlabs`) and secure account setup/name
  audition tools are implemented and covered by local mock-network/audio tests.
  The default model is `eleven_multilingual_v2`. After matching Brian and Roger
  auditions, Luca selected Roger (`CwhRBWXzGAHq8TQ4Fs17`) as the candidate.
  The accepted browser settings are speed 1.0, stability 0.5, similarity 0.75,
  style 0.0, and speaker boost enabled. Plain `Luca` works in the Pi auditions.
  Complete-sentence download buffering produced three clear isolated auditions.
  The first full conversations still lost audio; the subsequent 200 ms buffer
  change produced the three clean turns reported above. Bounded lookahead now
  prepares sentence N+1 while sentence N plays. Three controlled live trials
  preserved output order, kept every inter-sentence gap at 0.001 seconds or
  less, and produced zero output underflows. It does not reduce the delay before
  the first spoken sentence. See [cloud voice setup](cloud-voice.md), the
  [lookahead acceptance](lookahead-acceptance-20260927.md), and the investigation
  above.
- Cloud speech sends generated reply text and limited same-reply context to
  ElevenLabs. It does not send microphone audio to ElevenLabs. No zero-retention
  promise is made; the account's data settings apply.

### Persistent memory candidate

- The September 18 design choices are explicit saves plus tagged inference,
  running jokes stored separately from facts, and a first version focused on
  Luca. The corrected candidate is installed on the Pi.
- Its brain/personality integration matches the current repair candidate. All
  49 installed memory tests pass locally, using temporary synthetic data.
- The corrected September 18 working copy fixes supported preference matching,
  ambiguous deletion, joke/fact collisions, and incomplete metadata, and handles
  failed writes without claiming success. Private memory and recovery files are
  excluded from Git. The accepted baseline suite passed all 101 tests; after
  turn-failure recovery, the published H2 baseline passes all 113.
  The guarded installer created a rollback backup before changing the working tree.
  A disposable restart-level acceptance test confirmed explicit facts as
  `CONFIRMED`, narrow natural inference as `UNCONFIRMED`, labeled recall, and
  persistent explicit and inferred forgetting. The test cleaned up after itself,
  leaving a valid empty private store. Topic-aware retrieval and automatic
  contradiction resolution are not implemented.
- The first live save/recall trial exposed a voice integration edge case: Whisper
  included "Hey Tars" at the start of both requests, while the local parser
  expected `remember` or `what` first. The first reply was ordinary model assent;
  no entry was stored. A follow-up correction strips one leading TARS address
  before memory parsing and inference. The exact addressed save/restart/recall
  path has regression coverage in the installed suite.

### Conversation behavior awaiting decisions

- The current repair pauses microphone capture during transcription, response
  generation, and playback. It does not support interruption. Continuous capture
  that discards playback frames would still require further work for HUH and
  interruption with speaker-echo control.
- The repair also changed the character wording from the movie robot to a
  practical companion inspired by it. Luca's acceptance of that creative change
  is not established by the audio tests. Keep this separate from audio repairs.

## Known problems

- Input loss and playback underflows occurred in the earlier candidate but were
  absent throughout the September 30 seven-turn session at 200 ms latency: all
  seven captures reported zero input overflows and every reply reported zero
  output underflows. Retain that tested baseline and watch for recurrence.
- The delay before the first spoken sentence remains. The live session measured
  4.9–8.5 seconds from end of speech to first playback. Lookahead overlaps later
  sentence preparation but cannot begin until the model emits a complete next
  sentence.

- The wake detector can activate on close phrases and produced several
  unexplained activations during the acknowledgment test.
- Four of ten Pi wake-word trials had input overflows and were excluded. The
  cause has not been isolated from the audio stack.
- The current conversation path pauses microphone capture during playback. Echo
  leakage remains relevant to the parked HUH experiment and any future design
  that keeps the microphone running while TARS speaks.
- Background noise can be classified as speech, delaying endpoint detection or
  running capture to its time limit.
- ElevenLabs playback uses bounded one-sentence lookahead. It intentionally does
  not run two synthesis requests ahead. Piper still uses its existing playback
  path.
- The prior Piper stress-marker change did not fix the name to Luca's ear, and
  the claimed stress-position diagnosis was not established. Pronunciation must
  be accepted by listening to the chosen voice, including its final vowel.
- Conversation history remains process-local, while selected facts and bits now
  persist through the private memory store across restarts.

## Parked

- Do not run another synthetic wake-word training cycle unless real use shows a
  clear need.
- Do not enable openWakeWord `patience` from the saved v2.2 scores alone. Those
  samples do not preserve consecutive real-time windows, so they cannot predict
  the recall cost of a two-frame rule.
- Do not integrate the current HUH WAV until a better performance is chosen and
  speaker-echo behavior is handled.
- Walking remains a stretch goal. Body work should begin with fit checks and a
  supported single-servo test before committing to the full power system.

## Next practical actions

1. Keep the optional `tiny.en` speech-recognition comparison separate; no switch
   from `base` is established. Exercise memory naturally during normal use and
   inspect only labels and behavior unless Luca asks to inspect stored content.
2. Build one saved engineering-checklist workflow with measurement and unit
   confirmation.
3. When a better HUH performance is available, reuse the validated scheduler
   and nonblocking PCM path, then retest echo behavior before integration.
4. Investigate unexplained wake activations only if they remain material after
   input audio is reliable.
