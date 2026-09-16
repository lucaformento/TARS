# Voice setup notes

The voice prototype runs on the builder's Raspberry Pi 5 with Debian 13, Python 3.13, a USB microphone/audio interface, and a speaker. The text interface is the simplest entry point; voice setup currently requires manual model provisioning and device configuration.

Piper is the installed default. An opt-in ElevenLabs replacement and short name
audition are now available; see [cloud voice setup](cloud-voice.md). That engine
is pending audible selection and Pi validation.

## Runtime dependencies

Start with the virtual environment and API key from the [README](../README.md#run-the-text-interface).

The voice application imports `numpy`, `sounddevice`, `faster-whisper`, `openwakeword`, and `piper-tts` in addition to the brain's dependencies. It also requires a working PortAudio installation and the ALSA `aplay` command. See the [sounddevice installation guide](https://python-sounddevice.readthedocs.io/en/latest/installation.html) for the platform audio dependency.

Versions verified in the working environment:

| Package | Version |
| :--- | :--- |
| `anthropic` | 0.111.0 |
| `openwakeword` | 0.4.0 |
| `piper-tts` | 1.8.0 |

The complete environment has not yet been captured in a lockfile. Use these notes as the current configuration reference; a clean voice installation on another machine has not been validated. In particular, the openWakeWord API used by this project is the 0.4.0 API (`wakeword_model_paths`).

## Model files

The repository excludes downloaded voice and wake-word models. Provision this layout inside the project:

```text
TARS/
├── voices/
│   └── tars-community/
│       ├── TARS.onnx
│       └── TARS.onnx.json
└── wakeword/
    ├── hey_tars.onnx
    ├── melspectrogram.onnx
    └── embedding_model.onnx
```

- **Piper:** the working Pi uses the community TARS ONNX model and matching JSON configuration in `voices/tars-community/`. The model exposes multiple speakers; `tars_voice.py` explicitly selects its `neutral` speaker. These downloaded files are excluded from Git. The reviewed package does not include a model card or documented training provenance, so this remains a prototype voice. Verify its source and reuse terms, or replace it with a documented voice, before distributing the model or presenting the voice as a portfolio asset.
- **Wake word:** provision the project's custom `hey_tars.onnx` candidate and its matching `melspectrogram.onnx` and `embedding_model.onnx` files. The files are excluded from Git. The custom detector's measured tradeoffs and accepted limitations are recorded in the [wake-word decision](wake-word.md).
- **Transcription:** the code selects faster-whisper's `base` model with `device="cpu"` and `compute_type="int8"`. Loading a named model downloads it on first use if it is not cached. See the [upstream model-loading documentation](https://github.com/SYSTRAN/faster-whisper#model-conversion).

## Match the configuration to your hardware

Edit these constants near the top of [`tars_voice.py`](../tars_voice.py):

| Constant | What to set |
| :--- | :--- |
| `WAKE_MODEL` | Absolute path to the wake-word ONNX file |
| `MELSPEC_MODEL` | Absolute path to the matching mel-spectrogram ONNX file |
| `EMBEDDING_MODEL` | Absolute path to the matching embedding ONNX file |
| `VOICE` | Absolute path to the Piper ONNX file |
| `VOICE_CONFIG` | Absolute path to the matching Piper JSON file |
| `MIC_NAME` | A distinctive substring of your input device's PortAudio name |

The committed paths currently point to `/home/lucadev/TARS/`, and `MIC_NAME` is `USB PnP`. Playback uses ALSA's `default` output device. Confirm that the microphone can capture audio and that `aplay` can play through that output.

Before Piper synthesis, the voice interface applies the Italian pronunciation
`ˈluːka` (“LOO-kah”) to the builder's name. This speech-only override does not
change the response text stored in conversation history. Audition it without
starting the microphone loop by running
`python tars_voice.py --test-name`.
Luca reports that it still sounds poor. The earlier stress-marker edit was an
attempt, not a verified pronunciation fix. Cloud speech starts with ordinary
spelling and offers a separate optional alias.

## Start a conversation

From the project directory with the virtual environment active:

```bash
python tars_voice.py
```

Normal operation shows the conversation and wake/sleep state without the
per-stage timing report. To temporarily restore the development measurements,
run `python tars_voice.py --diagnostics`.

1. Stay quiet during room-noise calibration.
2. Say **“Hey TARS.”** Wait for the terminal's `[wake]` indication, then ask a question.
3. Let TARS finish speaking. Ask a follow-up within eight seconds to continue without another wake word.
4. After `[sleep]`, use the wake word again. Press **Ctrl+C** to stop the program.

`[sleep]` leaves the process and its conversation history running. Ctrl+C exits
cleanly. Background noise can delay endpointing; the audio loop does not yet
support interrupting TARS while he speaks.
