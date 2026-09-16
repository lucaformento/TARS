# Project state

Last updated: September 15, 2026.

This file is the canonical record for decisions, known problems, parked work,
and the next practical actions. Update it in the same commit whenever a test or
implementation changes one of those facts. GitHub `main` remains the source of
truth for code and runtime configuration. Detailed measurements stay in the
linked technical notes instead of being duplicated here.

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

- The Pi currently runs the community `TARS.onnx` Piper model and explicitly
  selects its `neutral` speaker. The voice object is loaded once at startup.
- This is the installed prototype voice, not a final publication decision. The
  reviewed model package has no model card or documented training provenance.
  Verify its source and reuse terms, or replace it with a documented voice,
  before distributing the model or presenting the voice as a portfolio asset.
- Luca reports that the general voice and the pronunciation of his name still
  sound poor, and has approved paid online speech for better quality. Selecting
  and auditioning a replacement is now the active priority.
- An opt-in ElevenLabs runtime (`--tts elevenlabs`) and secure account setup/name
  audition tools are implemented and covered by local mock-network/audio tests.
  The default model is `eleven_multilingual_v2`; the voice is selected from the
  user's account. No cloud voice has yet been accepted or verified on the Pi.
  Piper remains the default until that audition. See [cloud voice setup](cloud-voice.md).
- Matching Brian and Roger samples were generated in the ElevenLabs browser
  workspace with Multilingual v2. Luca's listening verdict is pending; browser
  playback does not validate the new Pi playback path.
- Cloud speech sends generated reply text and limited same-reply context to
  ElevenLabs. It does not send microphone audio to ElevenLabs. No zero-retention
  promise is made; the account's data settings apply.

## Known problems

- The wake detector can activate on close phrases and produced several
  unexplained activations during the acknowledgment test.
- Four of ten Pi wake-word trials had input overflows and were excluded. The
  cause has not been isolated from the audio stack.
- Playback can leak into transcription. The acknowledgment test deliberately
  left echo control out so this behavior would be measurable.
- Background noise can be classified as speech, delaying endpoint detection or
  running capture to its time limit.
- Sentence playback is synchronous, so reading and synthesizing later streamed
  sentences pauses while the current sentence plays. The tested sounddevice PCM
  path is a useful basis for future output streaming.
- The prior Piper stress-marker change did not fix the name to Luca's ear, and
  the claimed stress-position diagnosis was not established. Pronunciation must
  be accepted by listening to the chosen voice, including its final vowel.
- Conversation history lasts only for the current process.

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

1. Audition replacement speech, including “Luca” and a normal reply, then verify
   the selected engine's output routing and conversation timing on the Pi.
2. After the voice choice, start persistent, user-correctable memories and running jokes.
3. Build one saved engineering-checklist workflow with measurement and unit
   confirmation.
4. When a better HUH performance is available, reuse the validated scheduler
   and nonblocking PCM path, then retest echo behavior before integration.
5. Investigate input overflows and unexplained wake activations during a future
   controlled audio session.
