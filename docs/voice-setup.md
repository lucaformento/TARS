# Voice setup notes

The voice prototype runs on the builder's Raspberry Pi 5 with Debian 13, Python 3.13, a USB microphone/audio interface, and a speaker. The text interface is the simplest entry point; voice setup currently requires manual model provisioning and device configuration.

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
│   ├── en_US-ryan-medium.onnx
│   └── en_US-ryan-medium.onnx.json
└── wakeword/
    └── hey_jarvis_v0.1.onnx
```

- **Piper:** obtain the ONNX model and its matching JSON configuration from the [Ryan medium voice directory](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/medium).
- **Wake word:** use the “Hey Jarvis” ONNX model from the [openWakeWord 0.4.0 resources](https://github.com/dscripka/openWakeWord/tree/v0.4.0/openwakeword/resources/models). Its supporting feature models must also be available to the installed package.
- **Transcription:** the code selects faster-whisper's `base` model with `device="cpu"` and `compute_type="int8"`. Loading a named model downloads it on first use if it is not cached. See the [upstream model-loading documentation](https://github.com/SYSTRAN/faster-whisper#model-conversion).

## Match the configuration to your hardware

Edit these constants near the top of [`tars_voice.py`](../tars_voice.py):

| Constant | What to set |
| :--- | :--- |
| `WAKE_MODEL` | Absolute path to the wake-word ONNX file |
| `VOICE` | Absolute path to the Piper ONNX file |
| `VOICE_CONFIG` | Absolute path to the matching Piper JSON file |
| `MIC_NAME` | A distinctive substring of your input device's PortAudio name |

The committed paths currently point to `/home/lucadev/TARS/`, and `MIC_NAME` is `USB PnP`. Playback uses ALSA's `default` output device. Confirm that the microphone can capture audio and that `aplay` can play through that output.

## Start a conversation

From the project directory with the virtual environment active:

```bash
python tars_voice.py
```

1. Stay quiet during room-noise calibration.
2. Say **“Hey Jarvis.”** Wait for the terminal's `[wake]` indication, then ask a question.
3. Let TARS finish speaking. Ask a follow-up within eight seconds to continue without another wake word.
4. After `[sleep]`, use the wake word again. Press **Ctrl+C** to stop the program.

`[sleep]` leaves the process and its conversation history running. A `KeyboardInterrupt` traceback after Ctrl+C is currently expected. Background noise can delay endpointing; the audio loop does not yet support interrupting TARS while he speaks.
