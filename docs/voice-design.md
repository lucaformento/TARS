# TARS-style voice design

Goal: a voice that reads as TARS by character — deep, deadpan, dry, machine
timing — using a newly designed synthetic voice. No voice here is cloned from
Bill Irwin or from film audio: ElevenLabs requires rights to any cloned voice,
bans cloning public figures, and several US states require written consent.
Prompts deliberately name no real person, film, or character.

## Step 1 — ElevenLabs Voice Design (browser)

In ElevenLabs: **Voices → Create a voice → Voice Design**. Paste one
description and the shared preview text, generate (three candidates per run),
listen, and save any keeper with a clear name (for example `TARS-B2`). Voice
Design only charges credits for the preview text it speaks.

Recommended settings: leave loudness at default. Start with the default
guidance; if a result drifts from the description, raise guidance a little.
ElevenLabs notes that very high guidance with a niche prompt can lower audio
quality. Detailed prompts at moderate guidance work best.

### Shared preview text

Use the same text for every variant so the candidates are comparable. It tests
the name, numbers, a dry punchline, and pacing.

> Luca, I'm ready when you are. We can check the wiring, work through a
> problem, or just talk. My humor setting is seventy-five percent. Honesty is
> ninety. That's the part you should worry about.

### Variant A — deadpan machine (baseline)

> Native American English. Male, 50s. Studio quality. Persona: dry, loyal
> robot companion. Emotion: deadpan, calm, wry. Low, slightly gravelly baritone
> with flat, even intonation and crisp consonants; unhurried pacing with small
> deliberate pauses before a punchline, as if a machine were underplaying a
> joke.

### Variant B — more mechanical

> Native American English. Male, 50s. Studio quality. Persona: rugged service
> robot. Emotion: flat, matter-of-fact, understated. Medium-low baritone with
> very steady pitch and almost no rise at the ends of sentences; precise,
> clipped diction and a faintly metallic, compressed timbre, as if heard
> through a small speaker inside a metal chassis.

### Variant C — warmer crewmate

> Native American English. Male, 50s. Studio quality. Persona: seasoned
> crewmate robot. Emotion: calm, dry, quietly warm. Relaxed, grounded mid-low
> voice with conversational rhythm; humor delivered completely straight;
> steady, slightly slow pace with a sense of loyalty under the dryness.

### Variant D — older and drier

> Native American English. Male, 60s. Studio quality. Persona: sardonic
> veteran machine. Emotion: laconic, dry, unimpressed. Slightly rough,
> weathered baritone; economical phrasing, long even vowels, and a brief pause
> before the last word of a joke.

### How to judge (by ear)

Score each saved candidate 1–5 on:

1. **Depth and weight** — low and steady, not a cartoon robot.
2. **Deadpan** — jokes land flat; no sing-song or salesperson energy.
3. **Machine feel** — controlled, even, precise.
4. **"Luca"** — LOO-kah, including the final vowel (Roger's accepted standard).
5. **Clarity** — still clear on a laptop or phone speaker (the Pi speaker is
   small).

Mix-and-match is expected: if B is too robotic and C too human, write a
variant between them. Record each run below.

## Step 2 — Pi audition (Luca present; paid and audible)

With the saved voice ID, from `~/TARS`:

```bash
./venv/bin/python cloud_speech.py audition --voice-id VOICE_ID --test-name
./venv/bin/python cloud_speech.py audition --voice-id VOICE_ID
```

Switching the conversation voice is then one `.env` change
(`ELEVENLABS_VOICE_ID`), made only after Luca accepts the voice on the Pi.
Roger (`CwhRBWXzGAHq8TQ4Fs17`) remains the fallback.

## Step 3 — delivery and "machine body" (later)

- Delivery: the runtime currently fixes speed 1.0, stability 0.5, similarity
  0.75, style 0.0, speaker boost on (`cloud_speech.py`). Higher stability and
  speed near 0.95 may flatten and slow delivery; test only after a voice is
  chosen.
- Machine body: an optional, subtle playback filter (band-limit, light
  compression, very short metallic reflection) applied to each fully
  downloaded sentence before playback. It adds essentially no delay. Prototype
  offline and A/B by ear before any runtime change.

## Audition log

| Date | Variant | Guidance | Saved as | Scores (1–5: depth, deadpan, machine, Luca, clarity) | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-01 | A or C (to confirm) | default | `TARS-A1` | — | 1 of 3 kept; saved description was variant C |
| 2026-10-01 | B | default | `TARS-B1` | — | **Luca's favourite (Oct 2).** Voice ID `1LEJRs8TVTJqtqlKrZ1G`, ElevenLabs category `generated` (Voice Design, not a clone). Next: Pi audition, then switch decision |
