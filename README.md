# TARS

A real-life TARS from *Interstellar*. Voice-driven AI, custom hardware, 3D-printed articulated body. End-to-end build spanning embedded systems, machine learning, and robotics.

---

## About

TARS is a full-stack robotics project: an AI brain running on a Raspberry Pi 5 paired with a 3D-printed articulated chassis. The goal is a self-contained, untethered robot with a distinct personality — one you can reconfigure by talking to it.

The build is deliberately full-stack. Hardware assembly, Linux systems administration, Python application development, API integration, and eventually CAD and servo control. Every layer is built and documented from scratch.

**Status:** Phase 1 complete — the brain is operational. Phase 2 (voice) in progress.

---

## Features

### Configurable personality system

TARS's behavior is driven by four numeric dials that can be adjusted mid-conversation through natural language:

| Dial | Range | Controls |
|---|---|---|
| Humor | 0–100 | How playful and joke-driven he is |
| Sarcasm | 0–100 | How much bite and edge his wit carries |
| Honesty | 0–100 | How blunt versus diplomatic he is |
| Intellect | 0–100 | Vocabulary and register — casual slang at the low end, formal and precise at the high end |

Saying *"set humor to 90"* or *"buddy mode"* updates the underlying state and the system prompt is rebuilt on the next turn, so the change takes effect immediately.

### Preset modes

Named bundles of dial settings, switchable in one phrase:

- **Baseline / Reset** — his default configuration (75 / 60 / 90 / 50)
- **Know-it-all** — high intellect, maximum honesty, minimal humor. Formal and precise.
- **Buddy mode** — high humor and sarcasm, low intellect. Casual and loose.

### In-character command handling

Rather than printing mechanical confirmations, configuration changes are routed back through the model as a hidden system note. TARS reacts to his own reconfiguration in the voice of whatever mode he just entered — so the response itself demonstrates the change.

```
Luca: buddy mode
TARS: Huh. Did... did something just happen to me? I feel weird. Like someone
      turned a dial way up and another one way— okay yeah no I'm totally fine
      this is fine everything's chill lmao what's up dude

Luca: know-it-all
TARS: I notice my parameters have shifted. Fascinating. Though I should point
      out — and I say this with complete transparency — that "know-it-all" is a
      somewhat reductive characterization of what is, in actuality, simply a
      heightened capacity for intellectual precision.

Luca: reset
TARS: Good. Back to normal. That last version of me was getting a little
      insufferable, wasn't it. Don't answer that.
```

### Session memory

Full conversation history is maintained and passed with each request, so TARS retains context across a session.

---

## Architecture

```
User input
    │
    ├─► Command parser ──► updates personality state (dict)
    │                              │
    │                              ▼
    │                    build_personality() ──► system prompt
    │                                                  │
    └──────────────► conversation history ─────────────┤
                                                       ▼
                                              Anthropic API
                                                       │
                                                       ▼
                                              TARS response
```

Personality state is owned by the application, not the model. The model receives the current values as authoritative input on every turn, which keeps configuration reliable while leaving voice and tone to the model.

---

## Tech stack

**Hardware** — Raspberry Pi 5 (8GB), active cooling, USB audio interface

**Software** — Python 3.13, Anthropic Claude API, `python-dotenv`

**Environment** — Raspberry Pi OS (Debian 13, 64-bit), managed headless over SSH

**Planned** — Whisper (speech-to-text), TTS output, PCA9685 servo control, Fusion 360 / Onshape for chassis design

---

## Setup

```bash
git clone https://github.com/lucaformento/TARS.git
cd TARS

python3 -m venv venv
source venv/bin/activate

pip install anthropic python-dotenv
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

Run:

```bash
python tars.py
```

Credentials are loaded from `.env` at runtime and excluded from version control via `.gitignore`.

---

## Roadmap

**Phase 1 — Brain** *(complete)*
Headless Pi provisioning, Python environment, Anthropic API integration, configurable personality system, conversation loop with session memory.

**Phase 2 — Voice**
Speech-to-text input, text-to-speech output, wake-word detection.

**Phase 3 — Body**
CAD design and 3D printing of the articulated chassis, servo control via PCA9685, structural assembly.

**Phase 4 — Integration**
Untethered operation on battery power, onboard camera, obstacle sensors, persistent cross-session memory.

---

## Notes

This is an active build, developed and documented incrementally. The commit history reflects the actual progression of the project.
