"""Voice front-end for TARS.

Wake word starts a conversation; after that he keeps listening until you
go quiet for a while, so follow-ups need no wake word.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
import wave
import warnings
from contextlib import closing, ExitStack

import numpy as np
import sounddevice as sd
from cloud_speech import CloudSpeechError, ElevenLabsVoice, MODELS, NAME_TEST, load_environment

WAKE_MODEL = "/home/lucadev/TARS/wakeword/hey_tars.onnx"
MELSPEC_MODEL = "/home/lucadev/TARS/wakeword/melspectrogram.onnx"
EMBEDDING_MODEL = "/home/lucadev/TARS/wakeword/embedding_model.onnx"
VOICE = "/home/lucadev/TARS/voices/tars-community/TARS.onnx"
VOICE_CONFIG = "/home/lucadev/TARS/voices/tars-community/TARS.onnx.json"
MIC_NAME = "USB PnP"

STT_MODEL = "base"        # swap to "tiny.en" for ~3x speed, some accuracy loss

SR = 16000
FRAME = 1280              # 80ms, the size openWakeWord expects
WAKE_THRESHOLD = 0.5
WAKE_VAD_THRESHOLD = 0.5  # suppress scores when the VAD does not detect speech
WAKE_COOLDOWN = 2.0       # ignore a stale/duplicate trigger after going to sleep
SILENCE_END = 0.7         # quiet needed to call your sentence finished
MAX_UTTERANCE = 15.0      # hard cap so a noisy room can't record forever
MIN_SPEECH = 0.4          # shorter than this is a cough, not a sentence
TRIM_PAD = 2              # frames of padding kept around speech (160ms)
FIRST_WAIT = 6.0          # after the wake word, how long to wait for you
FOLLOWUP_WAIT = 8.0       # after he answers, how long before he sleeps
LUCA_PRONUNCIATION = "[[\u02c8lu\u02d0ka]]"  # Italian: LOO-kah, stress first

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF"
                   "\U0001F1E6-\U0001F1FF\U0000FE0F\U00002190-\U000021FF]+")
MARKDOWN = re.compile(r"[*_`#>~\[\]]")


def sanitize(text):
    """Strip formatting the speech engine would read aloud. Backstop only —
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
    Silence is trimmed off both ends before returning, because Whisper's
    cost scales with clip length and dead air is pure wasted latency.
    Returns (clip, last_loud_read_at), or (None, None).
    The timestamp estimates speech end from the last loud frame read;
    input buffering and background noise can shift it."""
    frames, levels = [], []
    speaking, silence, waited = False, 0.0, 0.0
    step = FRAME / SR
    last_loud_read_at = None

    while True:
        frame, _ = stream.read(FRAME)
        loud = level(frame) > threshold
        if loud:
            last_loud_read_at = time.perf_counter()

        if not speaking:
            waited += step
            frames.append(frame)                     # keep a little pre-roll
            levels.append(loud)
            if len(frames) > int(0.5 * SR / FRAME):
                frames.pop(0)
                levels.pop(0)
            if loud:
                speaking = True
            elif waited >= wait:
                return None, None
        else:
            frames.append(frame)
            levels.append(loud)
            silence = 0.0 if loud else silence + step
            if silence >= SILENCE_END or len(frames) * step >= MAX_UTTERANCE:
                break

    # trim to the loud region, padded, so we don't transcribe silence
    speech = [i for i, was_loud in enumerate(levels) if was_loud]
    if not speech:
        return None, None
    start = max(0, speech[0] - TRIM_PAD)
    end = min(len(frames), speech[-1] + TRIM_PAD + 1)

    clip = np.concatenate(frames[start:end]).flatten().astype(np.float32) / 32768.0
    if len(clip) / SR < MIN_SPEECH:
        return None, None
    return clip, last_loud_read_at


def speak(voice, text):
    """Return preparation time, playback-phase time, and request timestamp.
    Cloud preparation ends at the first PCM write; later transfer overlaps play.
    The request timestamp is not acoustic onset for either engine.
    """
    text = sanitize(text)
    if isinstance(voice, ElevenLabsVoice):
        return voice.speak(text)
    from piper.config import SynthesisConfig

    # Keep the written name unchanged while guiding the selected voice's pronunciation.
    text = re.sub(r"(?i)\bluca\b", LUCA_PRONUNCIATION, text)
    if not text:
        return 0.0, 0.0, None
    with tempfile.TemporaryDirectory(prefix="tars-speech-") as folder:
        output = f"{folder}/reply.wav"
        t0 = time.perf_counter()
        with wave.open(output, "wb") as wav:
            voice.synthesize_wav(
                text,
                wav,
                syn_config=SynthesisConfig(
                    speaker_id=voice.config.speaker_id_map.get("neutral")
                ),
            )
        playback_requested_at = time.perf_counter()
        subprocess.run(["aplay", "-q", "-D", "default", output], check=True)
        playback_finished_at = time.perf_counter()
    return (playback_requested_at - t0,
            playback_finished_at - playback_requested_at,
            playback_requested_at)


def split_sentences(buffer, final=False):
    """Extract speech chunks; wait for whitespace to resolve split-token decimals.
    Common abbreviations are conservative: they may keep two sentences together.
    """
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
    chunks, start = [], 0
    for match in re.finditer(r"""[.!?]+["'\u201d\u2019)\]]*(?=\s)""", buffer):
        if match.group().startswith("."):
            prefix = buffer[:match.start() + 1]
            word = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", prefix)
            if word and (word[1].lower() in abbreviations or len(word[1]) == 1):
                continue
            if re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix):
                continue
        chunk = buffer[start:match.end()].strip()
        if chunk:
            chunks.append(chunk)
        start = match.end()
    remainder = buffer[start:].lstrip()
    if final and remainder.strip():
        chunks.append(remainder.strip())
        remainder = ""
    return chunks, remainder


def speak_stream(tars, voice, user_input):
    """Read and speak sequentially. Cycle time includes synthesis and playback."""
    if isinstance(voice, ElevenLabsVoice):
        voice.begin_response()
    started = time.perf_counter()
    metrics = {"first_text": None, "first_chunk": None, "first_play_request": None,
               "tts": 0.0, "playback": 0.0, "chunks": 0, "gaps": [],
               "cloud": isinstance(voice, ElevenLabsVoice), "output_underflows": 0}
    previous_play_end = None

    def emit(chunk):
        nonlocal previous_play_end
        if not sanitize(chunk):
            return
        ready = time.perf_counter()
        synth_s, play_s, requested = speak(voice, chunk)
        if requested is None:
            return
        if metrics["first_play_request"] is None:
            metrics["first_chunk"] = ready - started
            metrics["first_play_request"] = requested
        if previous_play_end is not None:
            metrics["gaps"].append(max(0.0, requested - previous_play_end))
        previous_play_end = requested + play_s
        metrics["tts"] += synth_s
        metrics["playback"] += play_s
        metrics["chunks"] += 1
        if metrics["cloud"]:
            metrics["output_underflows"] += voice.last_underflows

    buffer = ""
    print("  TARS: ", end="", flush=True)
    try:
        with closing(tars.respond_stream(user_input)) as response:
            for delta in response:
                if not delta:
                    continue
                if metrics["first_text"] is None:
                    metrics["first_text"] = time.perf_counter() - started
                print(delta, end="", flush=True)
                buffer += delta
                chunks, buffer = split_sentences(buffer)
                for chunk in chunks:
                    emit(chunk)
            chunks, buffer = split_sentences(buffer, final=True)
            for chunk in chunks:
                emit(chunk)
    finally:
        print(flush=True)
    metrics["cycle"] = time.perf_counter() - started
    return metrics


def print_stream_timing(metrics, clip_s, stt_s, last_loud_read_at):
    """Report client-side observations without calling playback time API latency."""
    def seconds(value):
        return "n/a" if value is None else f"{value:.2f}s"

    preparation = "wait before playback" if metrics["cloud"] else "tts total"
    playback = "stream/playback phase" if metrics["cloud"] else "playback calls"
    print(f"  [clip {clip_s:.1f}s] stt {stt_s:.1f}s | chunks {metrics['chunks']} | "
          f"{preparation} {metrics['tts']:.2f}s | {playback} {metrics['playback']:.1f}s")
    if metrics["cloud"]:
        print("  Cloud stream/playback includes overlapping download and generation; "
              f"output underflows: {metrics['output_underflows']}")
    print(f"  Request -> first text: {seconds(metrics['first_text'])} | "
          f"first speech chunk: {seconds(metrics['first_chunk'])}")
    requested = metrics["first_play_request"]
    if requested is not None:
        print(f"  Detected end -> first playback request: ~{requested - last_loud_read_at:.2f}s "
              "(estimate; excludes playback startup)")
    else:
        print("  No spoken output.")
    if metrics["gaps"]:
        gaps = ", ".join(f"{gap:.2f}s" for gap in metrics["gaps"])
        print(f"  Gaps before next playback requests: {gaps} (excludes device startup)")
    print(f"  Response cycle: {metrics['cycle']:.2f}s (includes synthesis and playback)\n")


def main():
    parser = argparse.ArgumentParser(description="Talk with TARS.")
    parser.add_argument("--tts", choices=("piper", "elevenlabs"), default="piper",
                        help="Speech engine; ElevenLabs requires account configuration.")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--test-name", action="store_true",
                        help="Speak a short name test, then exit without starting the microphone.")
    parser.add_argument("--elevenlabs-model", choices=MODELS)
    parser.add_argument("--name-alias", help='ElevenLabs speech-only name spelling, e.g. "Loo kah".')
    args = parser.parse_args()
    if args.tts != "elevenlabs" and (args.elevenlabs_model or args.name_alias is not None):
        parser.error("--elevenlabs-model and --name-alias require --tts elevenlabs")
    with ExitStack() as cleanup:
        if args.tts == "elevenlabs":
            load_environment()
            voice = ElevenLabsVoice(model_id=args.elevenlabs_model, name_alias=args.name_alias)
            cleanup.callback(voice.close)
        else:
            from piper.voice import PiperVoice
            voice = PiperVoice.load(VOICE, config_path=VOICE_CONFIG)
        if args.test_name:
            print("Name audition: listen for LOO-kah.")
            speak(voice, NAME_TEST)
            return
        run_conversation(voice, args.diagnostics)


def run_conversation(voice, diagnostics=False):
    import openwakeword
    from faster_whisper import WhisperModel
    from brain import TARS

    warnings.filterwarnings(
        "ignore",
        message="Specified provider 'CUDAExecutionProvider' is not in available provider names.*",
        module="onnxruntime.*",
    )

    dev = next(i for i, d in enumerate(sd.query_devices())
               if MIC_NAME in d["name"] and d["max_input_channels"] > 0)

    oww = openwakeword.Model(
        wakeword_model_paths=[WAKE_MODEL],
        melspec_onnx_model_path=MELSPEC_MODEL,
        embedding_onnx_model_path=EMBEDDING_MODEL,
        vad_threshold=WAKE_VAD_THRESHOLD,
    )
    stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    tars = TARS(voice=True)          # short, speech-shaped replies

    with sd.InputStream(device=dev, samplerate=SR, channels=1,
                        dtype="int16", blocksize=FRAME) as stream:

        print("Calibrating room noise, stay quiet...")
        threshold = calibrate(stream)
        print(f"Threshold: {threshold:.0f}  |  STT: {STT_MODEL}")
        print("Say 'hey tars' to start. Ctrl+C to stop.\n")

        wake_ready_at = 0.0
        while True:
            # --- asleep: nothing but wake-word detection ---
            audio, _ = stream.read(FRAME)
            wake_score = max(oww.predict(audio.flatten()).values())
            if time.monotonic() < wake_ready_at or wake_score <= WAKE_THRESHOLD:
                continue

            print("[wake]")
            if hasattr(oww, "reset"):
                oww.reset()
            flush(stream)
            wait = FIRST_WAIT

            # --- awake: keep talking until silence sends him back to sleep ---
            while True:
                clip, last_loud_read_at = record_utterance(stream, threshold, wait)
                if clip is None:
                    print("[sleep]\n")
                    wake_ready_at = time.monotonic() + WAKE_COOLDOWN
                    break

                clip_s = len(clip) / SR

                t0 = time.perf_counter()
                segments, _ = stt.transcribe(clip, language="en")
                heard = " ".join(s.text for s in segments).strip()
                stt_s = time.perf_counter() - t0

                if not heard:
                    print("[sleep]\n")
                    break

                print(f"  Luca: {heard}")

                try:
                    metrics = speak_stream(tars, voice, heard)
                except CloudSpeechError as exc:
                    print(f"[speech unavailable] {exc}")
                    print("[sleep]\n")
                    flush(stream)
                    if hasattr(oww, "reset"):
                        oww.reset()
                    wake_ready_at = time.monotonic() + WAKE_COOLDOWN
                    break
                else:
                    if diagnostics:
                        print_stream_timing(metrics, clip_s, stt_s, last_loud_read_at)

                flush(stream)
                if hasattr(oww, "reset"):
                    oww.reset()
                wait = FOLLOWUP_WAIT


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTARS stopped.")
    except CloudSpeechError as exc:
        print(f"Speech unavailable: {exc}", file=sys.stderr)
        sys.exit(1)
