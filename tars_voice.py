"""Voice front-end for TARS.

Wake word starts a conversation; after that he keeps listening until you
go quiet for a while, so follow-ups need no wake word.
"""

import argparse
from contextlib import closing, contextmanager, ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
import warnings
from statistics import median

import numpy as np
import sounddevice as sd
from cloud_speech import (CloudSpeechError, ElevenLabsVoice, MODELS, NAME_TEST,
                          SpeechPreparationCancelled, load_environment)
from sentences import split_sentences

WAKE_MODEL = "/home/lucadev/TARS/wakeword/hey_tars.onnx"
MELSPEC_MODEL = "/home/lucadev/TARS/wakeword/melspectrogram.onnx"
EMBEDDING_MODEL = "/home/lucadev/TARS/wakeword/embedding_model.onnx"
VOICE = "/home/lucadev/TARS/voices/tars-community/TARS.onnx"
VOICE_CONFIG = "/home/lucadev/TARS/voices/tars-community/TARS.onnx.json"
MIC_NAME = "USB PnP"

# The October 5 same-clip benchmark: tiny.en with greedy decoding matched
# base/beam 5 on all 11 clips at about half the decode time. TARS_STT_MODEL and
# TARS_STT_BEAM switch back (for example base and 5) without a code change.
STT_MODEL = "tiny.en"
STT_BEAM = 1
STT_MODELS = ("tiny.en", "tiny", "base.en", "base")
STT_BEAMS = range(1, 6)

SR = 16000
FRAME = 1280              # 80ms, the size openWakeWord expects
WAKE_THRESHOLD = 0.5
WAKE_VAD_THRESHOLD = 0.5  # suppress scores when the VAD does not detect speech
WAKE_COOLDOWN = 2.0       # ignore a stale/duplicate trigger after going to sleep
SILENCE_END = 1.2         # conservative pause before calling a sentence finished
MAX_UTTERANCE = 30.0      # long enough for a natural thought; still bounds noise
MIN_SPEECH = 0.4          # minimum accumulated above-threshold speech
PRE_ROLL = 0.5            # audio retained before the first above-threshold frame
TAIL_PAD = 0.5            # quiet audio retained after the final loud frame
FIRST_WAIT = 6.0          # after the wake word, how long to wait for you
FOLLOWUP_WAIT = 8.0       # after he answers, how long before he sleeps
INPUT_BUFFER_SECONDS = 3.0  # keep draining USB while wake inference briefly stalls
LUCA_PRONUNCIATION = "[[\u02c8lu\u02d0ka]]"  # Italian: LOO-kah, stress first

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF"
                   "\U0001F1E6-\U0001F1FF\U0000FE0F\U00002190-\U000021FF]+")
MARKDOWN = re.compile(r"[*_`#>~\[\]]")


@dataclass
class InputReadStats:
    reads: int = 0
    overflows: int = 0


class TrackedInput:
    """Count every PortAudio input overflow instead of silently discarding it."""

    def __init__(self, stream, warn=print):
        self.stream = stream
        self.stats = InputReadStats()
        self.warn = warn

    def read(self, frames=FRAME):
        audio, overflowed = self.stream.read(frames)
        self.stats.reads += 1
        overflow_count = int(overflowed)
        if overflow_count:
            first = self.stats.overflows + 1
            self.stats.overflows += overflow_count
            self.warn(
                f"[audio warning] microphone input overflow #{first}"
                + (f" through #{self.stats.overflows}" if overflow_count > 1 else "")
                + "; "
                "captured speech may be incomplete."
            )
        return audio


class BufferedInput:
    """Continuously drain PortAudio into a small, bounded frame queue.

    openWakeWord inference runs in the main thread. A callback keeps the USB
    capture endpoint serviced during brief inference stalls. Hardware and queue
    overflows are carried to the reader so damaged speech can be rejected.
    """

    def __init__(self, max_seconds=INPUT_BUFFER_SECONDS):
        depth = max(2, round(max_seconds * SR / FRAME))
        self._frames = queue.Queue(maxsize=depth)
        self._pending_overflows = 0
        self._lock = threading.Lock()
        self._stream = None

    def attach(self, stream):
        self._stream = stream

    def callback(self, indata, _frames, _time_info, status):
        overflow_count = int(bool(getattr(status, "input_overflow", False)))
        frame = indata.copy()
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            # Keep the newest audio. Losing an old frame still invalidates the
            # current utterance, so report it through the overflow path.
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                pass
            overflow_count += 1
        if overflow_count:
            with self._lock:
                self._pending_overflows += overflow_count

    def read(self, frames=FRAME):
        if frames != FRAME:
            raise ValueError(f"BufferedInput requires {FRAME}-sample reads")
        try:
            frame = self._frames.get(timeout=2.0)
        except queue.Empty as exc:
            raise MicrophoneStreamError(
                "The microphone stopped delivering audio."
            ) from exc
        with self._lock:
            overflows = self._pending_overflows
            self._pending_overflows = 0
        return frame, overflows

    def clear(self, reset_overflows=False):
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        if reset_overflows:
            with self._lock:
                self._pending_overflows = 0

    def stop(self, ignore_errors=False):
        if self._stream is None:
            raise MicrophoneStreamError("The microphone buffer is not attached.")
        self._stream.stop(ignore_errors=ignore_errors)
        self.clear(reset_overflows=True)

    def start(self):
        if self._stream is None:
            raise MicrophoneStreamError("The microphone buffer is not attached.")
        self.clear(reset_overflows=True)
        self._stream.start()


@dataclass
class CaptureResult:
    clip: object | None
    last_loud_read_at: float | None
    endpoint_reason: str
    rms_levels: list[float]
    input_overflows: int


class MicrophoneStreamError(RuntimeError):
    """The input device could not be paused or restarted safely."""


def sanitize(text):
    """Strip formatting the speech engine would read aloud. Backstop only —
    the real fix is the voice style block in the system prompt."""
    text = EMOJI.sub("", text)
    text = MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def level(frame):
    """Loudness of one frame (RMS)."""
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def calibrate(input_reader, seconds=1.0):
    """Measure the room, set the speech threshold above it."""
    levels = [level(input_reader.read(FRAME))
              for _ in range(int(seconds * SR / FRAME))]
    ambient = sorted(levels)[len(levels) // 2]      # median ignores clicks
    return max(ambient * 3.0, 150.0)


def record_utterance(input_reader, threshold, wait):
    """Wait up to `wait` seconds for speech, then record until you stop.
    Keep the full pre-roll and a generous tail so quiet syllables survive.
    The result records why capture ended and any input overflows.
    The timestamp estimates speech end from the last loud frame read;
    input buffering and background noise can shift it."""
    frames, rms_levels = [], []
    speaking, silence, waited = False, 0.0, 0.0
    step = FRAME / SR
    last_loud_read_at = None
    endpoint_reason = None
    overflow_start = input_reader.stats.overflows
    pre_roll_frames = max(1, round(PRE_ROLL / step))
    tail_frames = max(1, round(TAIL_PAD / step))

    while True:
        frame = input_reader.read(FRAME)
        rms = level(frame)
        loud = rms > threshold
        if loud:
            last_loud_read_at = time.perf_counter()

        if not speaking:
            waited += step
            frames.append(frame)                     # keep a little pre-roll
            rms_levels.append(rms)
            if len(frames) > pre_roll_frames:
                frames.pop(0)
                rms_levels.pop(0)
            if loud:
                speaking = True
            elif waited >= wait:
                return CaptureResult(
                    None, None, "speech_timeout", rms_levels,
                    input_reader.stats.overflows - overflow_start,
                )
        else:
            frames.append(frame)
            rms_levels.append(rms)
            silence = 0.0 if loud else silence + step
            if silence >= SILENCE_END:
                endpoint_reason = "silence"
                break
            if len(frames) * step >= MAX_UTTERANCE:
                endpoint_reason = "max_utterance"
                break

    speech = [i for i, rms in enumerate(rms_levels) if rms > threshold]
    if not speech:
        return CaptureResult(
            None, None, "no_speech", rms_levels,
            input_reader.stats.overflows - overflow_start,
        )
    # The list already contains no more than PRE_ROLL seconds before speech.
    # Preserve it all; trim only excess endpoint silence beyond TAIL_PAD.
    end = min(len(frames), speech[-1] + tail_frames + 1)

    # Judge the cough/noise guard using loud frames, not the padded clip. A
    # single 80 ms spike must not become a valid sentence merely because the
    # saved clip also contains pre-roll and endpoint silence.
    if len(speech) * step < MIN_SPEECH:
        return CaptureResult(
            None, last_loud_read_at, "too_short", rms_levels,
            input_reader.stats.overflows - overflow_start,
        )
    clip = np.concatenate(frames[:end]).flatten().astype(np.float32) / 32768.0
    return CaptureResult(
        clip, last_loud_read_at, endpoint_reason, rms_levels[:end],
        input_reader.stats.overflows - overflow_start,
    )


def stt_settings():
    """Return (model, beam) from the environment, falling back to the defaults."""
    model = os.environ.get("TARS_STT_MODEL", "").strip() or STT_MODEL
    if model not in STT_MODELS:
        raise ValueError(f"TARS_STT_MODEL must be one of: {', '.join(STT_MODELS)}.")
    raw_beam = os.environ.get("TARS_STT_BEAM", "").strip() or str(STT_BEAM)
    try:
        beam = int(raw_beam)
    except ValueError:
        beam = None
    if beam not in STT_BEAMS:
        raise ValueError(f"TARS_STT_BEAM must be a whole number from {STT_BEAMS.start} "
                         f"to {STT_BEAMS.stop - 1}.")
    return model, beam


def warm_stt(stt, beam=STT_BEAM):
    """Run and fully consume one silent transcription before live speech."""
    silence = np.zeros(SR // 2, dtype=np.float32)
    segments, _ = stt.transcribe(silence, language="en", beam_size=beam)
    for _ in segments:
        pass


@contextmanager
def paused_input(stream):
    """Prevent USB capture overruns and speaker echo during processing/playback."""
    try:
        # sounddevice defaults to ignore_errors=True for stop(), which would
        # make a failed pause look successful and recreate the unread-input
        # overrun this lifecycle is meant to prevent.
        stream.stop(ignore_errors=False)
    except Exception as exc:
        raise MicrophoneStreamError("Could not pause microphone input safely.") from exc
    try:
        yield
    except BaseException:
        try:
            stream.start()
        except Exception as restart_exc:
            # A CloudSpeechError is normally recoverable, but continuing the
            # wake loop with a stopped input stream is not. Escalate the device
            # failure so the outer handler terminates cleanly.
            raise MicrophoneStreamError(
                "Microphone input did not restart after a processing failure; "
                "TARS stopped instead of entering a dead listening loop."
            ) from restart_exc
        raise
    else:
        # Starting a stopped PortAudio stream gives the follow-up turn a clean
        # capture buffer instead of several seconds of unread speaker audio.
        try:
            stream.start()
        except Exception as exc:
            raise MicrophoneStreamError(
                "Microphone input did not restart; TARS stopped instead of listening silently."
            ) from exc


def report_turn_failure(exc, diagnostics=False):
    """Describe a recoverable turn failure; the caller returns TARS to sleep."""
    if isinstance(exc, CloudSpeechError):
        print(f"[speech unavailable] {exc}")
        return
    lines = str(exc).strip().splitlines()
    detail = f": {lines[0][:200]}" if lines else ""
    print(f"[turn failed] {type(exc).__name__}{detail}")
    if diagnostics:
        traceback.print_exc()


def prepare_diagnostic_directory(value):
    base_path = Path(value).expanduser().resolve()
    project_root = Path(__file__).resolve().parent
    try:
        base_path.relative_to(project_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "--capture-diagnostics must point outside the TARS Git folder because "
            "its WAV and JSON files contain private speech."
        )
    if base_path.exists() and not base_path.is_dir():
        raise ValueError("--capture-diagnostics must name a directory, not a file.")
    base_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("session-%Y%m%dT%H%M%SZ")
    path = base_path / stamp
    suffix = 1
    while path.exists():
        path = base_path / f"{stamp}-{suffix}"
        suffix += 1
    path.mkdir(mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)
    return path


def save_capture_diagnostics(directory, turn, capture, threshold, transcript,
                             stt_seconds, stt_info, segments, total_overflows,
                             stt_model=STT_MODEL, stt_beam=STT_BEAM):
    """Save only microphone/STT evidence explicitly requested by the operator."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"turn-{turn:03d}-{stamp}"
    audio_path = directory / f"{stem}.wav"
    json_path = directory / f"{stem}.json"

    pcm = (np.clip(capture.clip, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())

    levels = capture.rms_levels
    details = {
        "created_at_utc": stamp,
        "audio_file": audio_path.name,
        "sample_rate_hz": SR,
        "clip_seconds": round(len(capture.clip) / SR, 4),
        "endpoint_reason": capture.endpoint_reason,
        "threshold_rms": round(float(threshold), 4),
        "above_threshold_seconds": round(
            sum(value > threshold for value in levels) * FRAME / SR, 4
        ),
        "rms": {
            "minimum": round(min(levels), 4) if levels else None,
            "median": round(float(median(levels)), 4) if levels else None,
            "maximum": round(max(levels), 4) if levels else None,
            "frames": [round(value, 4) for value in levels],
        },
        "input_overflows_this_capture": capture.input_overflows,
        "input_overflows_total": total_overflows,
        "stt_model": stt_model,
        "stt_beam": stt_beam,
        "stt_seconds": round(stt_seconds, 4),
        "transcript": transcript,
        "stt_info": {
            "language": getattr(stt_info, "language", None),
            "language_probability": getattr(stt_info, "language_probability", None),
            "duration": getattr(stt_info, "duration", None),
            "duration_after_vad": getattr(stt_info, "duration_after_vad", None),
        },
        "segments": [
            {
                "start": getattr(segment, "start", None),
                "end": getattr(segment, "end", None),
                "text": getattr(segment, "text", ""),
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
            }
            for segment in segments
        ],
    }
    json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    if os.name == "posix":
        audio_path.chmod(0o600)
        json_path.chmod(0o600)
    print(f"  [diagnostics] saved private microphone capture: {audio_path}")


def speak(voice, text):
    """Return preparation time, playback-phase time, and request timestamp.
    Cloud preparation includes the complete sentence download before playback.
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


def _stream_metrics(cloud):
    return {"first_text": None, "first_sentence_ready": None,
            "first_play_request": None,
            "tts": 0.0, "playback": 0.0, "chunks": 0, "gaps": [],
            "cloud": cloud, "output_underflows": 0,
            "cloud_sentences": []}


def _record_playback(metrics, voice, started, ready, timing, previous_play_end):
    synth_s, play_s, requested = timing
    if requested is None:
        return previous_play_end
    if metrics["first_play_request"] is None:
        metrics["first_sentence_ready"] = ready - started
        metrics["first_play_request"] = requested
    if previous_play_end is not None:
        metrics["gaps"].append(max(0.0, requested - previous_play_end))
    metrics["tts"] += synth_s
    metrics["playback"] += play_s
    metrics["chunks"] += 1
    if metrics["cloud"]:
        metrics["output_underflows"] += voice.last_underflows
        metrics["cloud_sentences"].append(dict(voice.last_stats))
    return requested + play_s


def _speak_stream_cloud(tars, voice, user_input, started, metrics):
    """Prepare exactly one future sentence while the current one plays."""
    prepared_queue = queue.Queue(maxsize=1)
    lookahead_slot = threading.Semaphore(1)
    cancel = threading.Event()
    finished = object()
    model_response = []
    model_response_lock = threading.Lock()

    def acquire_slot():
        while not cancel.is_set():
            if lookahead_slot.acquire(timeout=0.05):
                return True
        return False

    def put(item):
        while not cancel.is_set():
            try:
                prepared_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def prepare_chunk(chunk, ready):
        text = sanitize(chunk)
        if not text or not acquire_slot():
            return
        try:
            prepared = voice.prepare(text, cancel_event=cancel)
        except BaseException:
            lookahead_slot.release()
            raise
        if prepared is None:
            lookahead_slot.release()
        elif not put(("speech", prepared, ready)):
            lookahead_slot.release()

    def produce():
        buffer = ""
        try:
            response = tars.respond_stream(user_input)
            with model_response_lock:
                model_response.append(response)
            try:
                with closing(response):
                    for delta in response:
                        if cancel.is_set():
                            return
                        if not delta:
                            continue
                        if metrics["first_text"] is None:
                            metrics["first_text"] = time.perf_counter() - started
                        print(delta, end="", flush=True)
                        buffer += delta
                        chunks, buffer = split_sentences(buffer)
                        for chunk in chunks:
                            ready = time.perf_counter()
                            prepare_chunk(chunk, ready)
                    chunks, buffer = split_sentences(buffer, final=True)
                    for chunk in chunks:
                        ready = time.perf_counter()
                        prepare_chunk(chunk, ready)
            finally:
                with model_response_lock:
                    model_response.clear()
        except SpeechPreparationCancelled:
            if not cancel.is_set():
                put(("error", CloudSpeechError("Cloud speech preparation was cancelled.")))
        except BaseException as exc:
            put(("error", exc))
        finally:
            put(("finished", finished))

    # This thread owns the output device. Open it before any paid request;
    # the worker only downloads, because a PortAudio abort from the worker
    # during this thread's blocking write left TARS permanently deaf.
    voice.ensure_output()
    worker = threading.Thread(target=produce, name="tars-elevenlabs-lookahead", daemon=True)
    worker.start()
    previous_play_end = None
    try:
        while True:
            item = prepared_queue.get()
            if item[0] == "finished":
                break
            if item[0] == "error":
                raise item[1]
            _, prepared, ready = item
            # Consuming the prepared sentence opens the single lookahead slot.
            # The worker can download N+1 while this thread plays N, but it
            # cannot start N+2 until N+1 has been consumed for playback.
            lookahead_slot.release()
            timing = voice.play(prepared)
            previous_play_end = _record_playback(
                metrics, voice, started, ready, timing, previous_play_end
            )
    finally:
        cancel.set()
        voice.cancel_pending()
        with model_response_lock:
            response = model_response[0] if model_response else None
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        worker.join(timeout=2.0)
        if worker.is_alive() and sys.exc_info()[0] is None:
            raise CloudSpeechError("The response worker did not stop cleanly.")


def speak_stream(tars, voice, user_input):
    """Stream a response; cloud speech overlaps one future sentence."""
    cloud = isinstance(voice, ElevenLabsVoice)
    if cloud:
        voice.begin_response()
    started = time.perf_counter()
    metrics = _stream_metrics(cloud)
    previous_play_end = None

    def emit(chunk):
        nonlocal previous_play_end
        if not sanitize(chunk):
            return
        ready = time.perf_counter()
        previous_play_end = _record_playback(
            metrics, voice, started, ready, speak(voice, chunk), previous_play_end
        )

    print("  TARS: ", end="", flush=True)
    try:
        if cloud:
            _speak_stream_cloud(tars, voice, user_input, started, metrics)
        else:
            buffer = ""
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


def print_stream_timing(metrics, capture, stt_s):
    """Report client-side observations without calling playback time API latency."""
    def seconds(value):
        return "n/a" if value is None else f"{value:.2f}s"

    preparation = "download/prep total" if metrics["cloud"] else "tts total"
    playback = "playback phase" if metrics["cloud"] else "playback calls"
    clip_s = len(capture.clip) / SR
    print(f"  [clip {clip_s:.1f}s; endpoint {capture.endpoint_reason}; "
          f"input overflows {capture.input_overflows}] stt {stt_s:.1f}s | "
          f"chunks {metrics['chunks']} | "
          f"{preparation} {metrics['tts']:.2f}s | {playback} {metrics['playback']:.1f}s")
    if metrics["cloud"]:
        print("  Cloud audio is fully downloaded before playback; "
              "lookahead is capped at one sentence; "
              f"output underflows: {metrics['output_underflows']}")
    print(f"  Request -> first text: {seconds(metrics['first_text'])} | "
          f"first complete sentence: {seconds(metrics['first_sentence_ready'])}")
    for index, stats in enumerate(metrics["cloud_sentences"], start=1):
        first_pcm = seconds(stats.get("time_to_first_pcm_s"))
        print(
            f"  Cloud sentence {index}: first PCM {first_pcm}; "
            f"download {stats.get('download_s', 0.0):.2f}s; "
            f"audio {stats.get('audio_s', 0.0):.2f}s; "
            "largest PCM chunk interval "
            f"{stats.get('max_pcm_chunk_interval_s', 0.0):.2f}s; "
            f"device write {stats.get('write_s', 0.0):.2f}s; "
            f"drain {stats.get('drain_s', 0.0):.2f}s"
        )
    requested = metrics["first_play_request"]
    if requested is not None:
        print(f"  Detected end -> first playback request: "
              f"~{requested - capture.last_loud_read_at:.2f}s "
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
    parser.add_argument(
        "--capture-diagnostics", metavar="BASE_DIRECTORY",
        help=("Opt in to saving microphone WAVs and STT JSON in a new private "
              "session folder under BASE_DIRECTORY. Files are never saved when omitted."),
    )
    parser.add_argument("--test-name", action="store_true",
                        help="Speak a short name test, then exit without starting the microphone.")
    parser.add_argument("--elevenlabs-model", choices=MODELS)
    parser.add_argument("--name-alias", help='ElevenLabs speech-only name spelling, e.g. "Loo kah".')
    parser.add_argument(
        "--brain-model",
        help=("Anthropic response model. Supported values are validated by brain.py; "
              "omit to use its default or ANTHROPIC_MODEL."),
    )
    args = parser.parse_args()
    if args.test_name and args.capture_diagnostics:
        parser.error("--capture-diagnostics requires a microphone conversation, not --test-name")
    if args.tts != "elevenlabs" and (args.elevenlabs_model or args.name_alias is not None):
        parser.error("--elevenlabs-model and --name-alias require --tts elevenlabs")
    # Speech settings may live in .env, so read it before choosing the model.
    load_environment()
    try:
        stt_model, stt_beam = stt_settings()
    except ValueError as exc:
        parser.error(str(exc))
    with ExitStack() as cleanup:
        if args.tts == "elevenlabs":
            voice = ElevenLabsVoice(model_id=args.elevenlabs_model, name_alias=args.name_alias)
            cleanup.callback(voice.close)
        else:
            from piper.voice import PiperVoice
            voice = PiperVoice.load(VOICE, config_path=VOICE_CONFIG)
        if args.test_name:
            print("Name audition: listen for LOO-kah.")
            speak(voice, NAME_TEST)
            return
        try:
            diagnostic_dir = (prepare_diagnostic_directory(args.capture_diagnostics)
                              if args.capture_diagnostics else None)
        except ValueError as exc:
            parser.error(str(exc))
        run_conversation(voice, args.diagnostics, diagnostic_dir, args.brain_model,
                         stt_model, stt_beam)


def run_conversation(voice, diagnostics=False, diagnostic_dir=None, brain_model=None,
                     stt_model=STT_MODEL, stt_beam=STT_BEAM):
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
    stt = WhisperModel(stt_model, device="cpu", compute_type="int8")
    tars = TARS(model=brain_model, voice=True)  # short, speech-shaped replies
    notes_on = tars.enable_memory_notes()

    print("Warming speech recognition...")
    warm_started = time.perf_counter()
    warm_stt(stt, stt_beam)
    warm_seconds = time.perf_counter() - warm_started
    if diagnostics:
        print(f"Speech recognition ready ({warm_seconds:.1f}s startup warm-up).")
        print(f"Response model: {tars.model}")
        print(f"Memory notes: {'on' if notes_on else 'off'}")
    else:
        print("Speech recognition ready.")
    if diagnostic_dir is not None:
        print(
            "DIAGNOSTIC CAPTURE ENABLED: microphone WAVs and transcripts will be "
            f"saved in {diagnostic_dir}. Treat these files as private."
        )

    buffered_input = BufferedInput()
    # closing(tars) stops the note-taker on every exit, including Ctrl+C.
    with closing(tars), sd.InputStream(device=dev, samplerate=SR, channels=1,
                        dtype="int16", blocksize=FRAME, latency=0.2,
                        callback=buffered_input.callback) as stream:
        buffered_input.attach(stream)
        input_reader = TrackedInput(buffered_input)

        print("Calibrating room noise, stay quiet...")
        threshold = calibrate(input_reader)
        print(f"Threshold: {threshold:.0f}  |  STT: {stt_model}, beam {stt_beam}")
        print("Say 'hey tars' to start. Ctrl+C to stop.\n")

        wake_ready_at = 0.0
        turn = 0
        while True:
            # --- asleep: nothing but wake-word detection ---
            audio = input_reader.read(FRAME)
            wake_score = max(oww.predict(audio.flatten()).values())
            if time.monotonic() < wake_ready_at or wake_score <= WAKE_THRESHOLD:
                continue

            print("[wake]")
            tars.begin_session()
            if hasattr(oww, "reset"):
                oww.reset()
            # Do not flush here: queued frames can contain the beginning of a
            # natural question spoken immediately after the wake phrase.
            wait = FIRST_WAIT

            # --- awake: keep talking until silence sends him back to sleep ---
            while True:
                capture = record_utterance(input_reader, threshold, wait)
                if capture.clip is None:
                    if diagnostics:
                        print(f"  [capture ended: {capture.endpoint_reason}; "
                              f"input overflows {capture.input_overflows}]")
                    print("[sleep]\n")
                    wake_ready_at = time.monotonic() + WAKE_COOLDOWN
                    break

                if capture.input_overflows:
                    # Missing samples can turn a clear question into unrelated
                    # text. Never send known-damaged audio to Whisper or Claude.
                    print("  [audio retry] Part of that recording was lost; please repeat it.")
                    try:
                        with paused_input(buffered_input):
                            speak(voice, "I lost part of that. Please say it again.")
                    except MicrophoneStreamError:
                        raise
                    except Exception as exc:
                        report_turn_failure(exc, diagnostics)
                        print("[sleep]\n")
                        if hasattr(oww, "reset"):
                            oww.reset()
                        wake_ready_at = time.monotonic() + WAKE_COOLDOWN
                        break
                    wait = FOLLOWUP_WAIT
                    continue

                turn += 1
                metrics = None
                heard = ""
                try:
                    # Capture stops before CPU-heavy STT, both cloud requests,
                    # and playback. Restarting afterward gives the next turn a
                    # clean buffer and prevents known USB input overruns.
                    with paused_input(buffered_input):
                        t0 = time.perf_counter()
                        segment_iter, stt_info = stt.transcribe(capture.clip, language="en",
                                                                beam_size=stt_beam)
                        segments = list(segment_iter)
                        heard = " ".join(segment.text for segment in segments).strip()
                        stt_s = time.perf_counter() - t0

                        if diagnostic_dir is not None:
                            save_capture_diagnostics(
                                diagnostic_dir, turn, capture, threshold, heard,
                                stt_s, stt_info, segments, input_reader.stats.overflows,
                                stt_model, stt_beam,
                            )

                        if heard:
                            print(f"  Luca: {heard}")
                            metrics = speak_stream(tars, voice, heard)
                            # Only a reply that was generated and played in
                            # full may become a quiet memory note.
                            tars.submit_note()
                except MicrophoneStreamError:
                    raise
                except Exception as exc:
                    # paused_input has already restarted the microphone, so a
                    # failed cloud request, audio device, or transcription ends
                    # only this turn. A microphone that cannot restart is fatal.
                    report_turn_failure(exc, diagnostics)
                    print("[sleep]\n")
                    if hasattr(oww, "reset"):
                        oww.reset()
                    wake_ready_at = time.monotonic() + WAKE_COOLDOWN
                    break

                if not heard:
                    print("[sleep]\n")
                    wake_ready_at = time.monotonic() + WAKE_COOLDOWN
                    break

                if diagnostics:
                    print_stream_timing(metrics, capture, stt_s)

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
    except MicrophoneStreamError as exc:
        print(f"Microphone unavailable: {exc}", file=sys.stderr)
        sys.exit(1)
