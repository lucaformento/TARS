"""Voice front-end for TARS: wake word -> record -> transcribe -> brain -> speak."""

import subprocess
import tempfile

import numpy as np
import sounddevice as sd
import openwakeword
from faster_whisper import WhisperModel

from brain import TARS

WAKE_MODEL = "/home/lucadev/TARS/wakeword/hey_jarvis_v0.1.onnx"
VOICE = "/home/lucadev/TARS/voices/en_US-ryan-medium.onnx"
VOICE_CONFIG = "/home/lucadev/TARS/voices/en_US-ryan-medium.onnx.json"
PIPER = "/home/lucadev/TARS/venv/bin/piper"
MIC_NAME = "USB PnP"
THRESHOLD = 0.5
RECORD_SECONDS = 5


def speak(text):
    """Piper reads text on stdin, writes a WAV, aplay plays it."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        subprocess.run([PIPER, "-m", VOICE, "-c", VOICE_CONFIG, "-f", wav.name],
                       input=text.encode(), capture_output=True)
        subprocess.run(["aplay", "-q", "-D", "default", wav.name])


dev = next(i for i, d in enumerate(sd.query_devices())
           if MIC_NAME in d["name"] and d["max_input_channels"] > 0)

oww = openwakeword.Model(wakeword_model_paths=[WAKE_MODEL])
stt = WhisperModel("base", device="cpu", compute_type="int8")
tars = TARS()

print("TARS is listening. Say 'hey jarvis'. Ctrl+C to stop.")

with sd.InputStream(device=dev, samplerate=16000, channels=1,
                    dtype="int16", blocksize=1280) as stream:
    while True:
        audio, _ = stream.read(1280)
        if max(oww.predict(audio.flatten()).values()) <= THRESHOLD:
            continue

        print("  [wake]")
        frames = [stream.read(1280)[0]
                  for _ in range(int(RECORD_SECONDS * 16000 / 1280))]
        clip = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
        segments, _ = stt.transcribe(clip, language="en")
        heard = " ".join(s.text for s in segments).strip()

        if not heard:
            print("  (nothing heard)")
        else:
            print(f"  Luca: {heard}")
            reply = tars.respond(heard)     # same brain as the text front-end
            print(f"  TARS: {reply}\n")
            speak(reply)

        if hasattr(oww, "reset"):
            oww.reset()
