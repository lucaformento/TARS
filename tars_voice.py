"""Voice front-end for TARS.

Wake word starts a conversation; after that he keeps listening until you
go quiet for a while, so follow-ups need no wake word.
"""

import re
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

SR = 16000
FRAME = 1280              # 80ms, the size openWakeWord expects
WAKE_THRESHOLD = 0.5
SILENCE_END = 1.0         # quiet needed to call your sentence finished
MAX_UTTERANCE = 15.0      # hard cap so a noisy room can't record forever
MIN_SPEECH = 0.4          # shorter than this is a cough, not a sentence
FIRST_WAIT = 6.0          # after the wake word, how long to wait for you
FOLLOWUP_WAIT = 8.0       # after he answers, how long before he sleeps

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF"
                   "\U0001F1E6-\U0001F1FF\U0000FE0F\U00002190-\U000021FF]+")
MARKDOWN = re.compile(r"[*_`#>~\[\]]")


def sanitize(text):
    """Strip anything Piper would read aloud as punctuation. Backstop only —
    the real fix is the voice style block in the system prompt."""
    text = EMOJI.sub("", text)
    text = MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def level(frame):
    """Loudness of one frame (RMS)."""
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def calibrate(stream, seconds=1.0):
    """Measure the room, set the speech threshold above it."""
    levels = [level(stream.read(FRAME)[0])
              for _ in range(int(seconds * SR / FRAME))]
    ambient = sorted(levels)[len(levels) // 2]      # median ignores clicks
    return max(ambient * 3.0, 150.0)


def flush(stream):
    """Drop buffered audio so TARS never transcribes his own voice."""
    while stream.read_available >= FRAME:
        stream.read(FRAME)


def record_utterance(stream, threshold, wait):
    """Wait up to `wait` seconds for speech, then record until you stop.
    Returns a float32 clip, or None if you never spoke."""
    frames, speaking, silence, waited = [], False, 0.0, 0.0
    step = FRAME / SR

    while True:
        frame, _ = stream.read(FRAME)
        loud = level(frame) > threshold

        if not speaking:
            waited += step
            frames.append(frame)                     # keep a little pre-roll
            if len(frames) > int(0.5 * SR / FRAME):
                frames.pop(0)
            if loud:
                speaking = True
            elif waited >= wait:
                return None
        else:
            frames.append(frame)
            silence = 0.0 if loud else silence + step
            if silence >= SILENCE_END or len(frames) * step >= MAX_UTTERANCE:
                break

    clip = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
    return clip if len(clip) / SR >= MIN_SPEECH else None


def speak(text):
    text = sanitize(text)
    if not text:
        return
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        subprocess.run([PIPER, "-m", VOICE, "-c", VOICE_CONFIG, "-f", wav.name],
                       input=text.encode(), capture_output=True)
        subprocess.run(["aplay", "-q", "-D", "default", wav.name])


dev = next(i for i, d in enumerate(sd.query_devices())
           if MIC_NAME in d["name"] and d["max_input_channels"] > 0)

oww = openwakeword.Model(wakeword_model_paths=[WAKE_MODEL])
stt = WhisperModel("base", device="cpu", compute_type="int8")
tars = TARS(voice=True)          # short, speech-shaped replies

with sd.InputStream(device=dev, samplerate=SR, channels=1,
                    dtype="int16", blocksize=FRAME) as stream:

    print("Calibrating room noise, stay quiet...")
    threshold = calibrate(stream)
    print(f"Threshold: {threshold:.0f}")
    print("Say 'hey jarvis' to start. Ctrl+C to stop.\n")

    while True:
        # --- asleep: nothing but wake-word detection ---
        audio, _ = stream.read(FRAME)
        if max(oww.predict(audio.flatten()).values()) <= WAKE_THRESHOLD:
            continue

        print("[wake]")
        if hasattr(oww, "reset"):
            oww.reset()
        flush(stream)
        wait = FIRST_WAIT

        # --- awake: keep talking until silence sends him back to sleep ---
        while True:
            clip = record_utterance(stream, threshold, wait)
            if clip is None:
                print("[sleep]\n")
                break

            segments, _ = stt.transcribe(clip, language="en")
            heard = " ".join(s.text for s in segments).strip()
            if not heard:
                print("[sleep]\n")
                break

            print(f"  Luca: {heard}")
            reply = tars.respond(heard)
            print(f"  TARS: {reply}\n")
            speak(reply)

            flush(stream)
            if hasattr(oww, "reset"):
                oww.reset()
            wait = FOLLOWUP_WAIT
