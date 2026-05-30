from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import logging
import os
import shutil
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import torch
except Exception:  # pragma: no cover - torch is optional until inference starts.
    torch = None  # type: ignore[assignment]

try:
    import ctranslate2
except Exception:  # pragma: no cover - installed with faster-whisper in normal use.
    ctranslate2 = None  # type: ignore[assignment]

from faster_whisper import WhisperModel


logger = logging.getLogger("transcript_direct")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_SECONDS = 0.2
DEFAULT_PHRASE_MAX_SECONDS = 4.0
DEFAULT_PHRASE_SILENCE_SECONDS = 0.55
DEFAULT_PARAGRAPH_SILENCE_SECONDS = 1.2
DEFAULT_SPEECH_RMS_THRESHOLD = 0.0025
DEFAULT_ADAPTIVE_RMS_MULTIPLIER = 3.0
DEFAULT_BEAM_SIZE = 5
DEFAULT_CONTEXT_WORDS = 32
CUDA_REQUIRED_LIBRARIES = (
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
)

BUILTIN_MODELS = [
    {
        "id": "tiny",
        "label": "Whisper tiny",
        "detail": "Lightest option, prioritizes latency over accuracy.",
        "recommended": False,
        "local": False,
    },
    {
        "id": "base",
        "label": "Whisper base",
        "detail": "More accurate than tiny, still lightweight.",
        "recommended": False,
        "local": False,
    },
    {
        "id": "small",
        "label": "Whisper small",
        "detail": "Better accuracy with higher latency.",
        "recommended": False,
        "local": False,
    },
    {
        "id": "large-v3",
        "label": "Whisper large-v3",
        "detail": "Highest accuracy, more GPU use and latency.",
        "recommended": True,
        "local": False,
    },
]


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    label: str
    detail: str
    recommended: bool
    local: bool
    path: Path | None = None

    def public(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "label": self.label,
            "detail": self.detail,
            "recommended": self.recommended,
            "local": self.local,
        }
        if self.path is not None:
            payload["path"] = str(self.path)
        return payload


def _split_paths(raw: str) -> list[Path]:
    return [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]


def _default_model_roots() -> list[Path]:
    roots = [
        PROJECT_ROOT / "models" / "whisper",
    ]
    extra = os.getenv("TRANSCRIPT_MODEL_ROOTS", "")
    roots.extend(_split_paths(extra))
    return roots


def _looks_like_ctranslate2_model(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model.bin").exists()
        and (path / "config.json").exists()
        and ((path / "tokenizer.json").exists() or (path / "vocabulary.json").exists())
    )


def _human_model_label(name: str) -> str:
    if name.startswith("large"):
        return f"Whisper {name}"
    return name.replace("_", " ")


def _hf_cache_model_id(path: Path) -> str | None:
    name = path.name
    prefixes = (
        "models--Systran--faster-whisper-",
        "models--guillaumekln--faster-whisper-",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return None


def _resolve_hf_cache_snapshot(path: Path) -> Path | None:
    refs_main = path / "refs" / "main"
    snapshots_dir = path / "snapshots"
    snapshot: Path | None = None

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot = snapshots_dir / revision
    if snapshot is None or not snapshot.exists():
        candidates = sorted(
            (candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        ) if snapshots_dir.exists() else []
        snapshot = candidates[0] if candidates else None

    if snapshot is not None and _looks_like_ctranslate2_model(snapshot):
        return snapshot.resolve()
    return None


def build_model_catalog() -> dict[str, CatalogEntry]:
    catalog: dict[str, CatalogEntry] = {
        item["id"]: CatalogEntry(**item) for item in BUILTIN_MODELS
    }

    for root in _default_model_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            model_id = child.name
            model_path: Path | None = None
            if _looks_like_ctranslate2_model(child):
                model_path = child.resolve()
            else:
                hf_model_id = _hf_cache_model_id(child)
                if hf_model_id:
                    model_id = hf_model_id
                    model_path = _resolve_hf_cache_snapshot(child)
            if model_path is None:
                continue

            existing = catalog.get(model_id)
            label = existing.label if existing is not None else _human_model_label(model_id)
            recommended = existing.recommended if existing is not None else False
            catalog[model_id] = CatalogEntry(
                id=model_id,
                label=label,
                detail=f"Local model at {model_path}",
                recommended=recommended,
                local=True,
                path=model_path,
            )

    return catalog


def _select_device() -> str:
    requested = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if ctranslate2 is not None:
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def _cuda_available() -> bool:
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            pass
    if ctranslate2 is not None:
        try:
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            pass
    return False


def _select_compute_type(device: str) -> str:
    raw = os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
    if raw:
        return raw
    return "float16" if device == "cuda" else "int8"


def _missing_shared_libraries(names: tuple[str, ...]) -> list[str]:
    missing = []
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def _missing_cuda_libraries() -> list[str]:
    return _missing_shared_libraries(CUDA_REQUIRED_LIBRARIES)


def _cuda_setup_error(missing: list[str]) -> str:
    libraries = ", ".join(missing)
    return (
        "CUDA is selected but required CUDA shared libraries are missing: "
        f"{libraries}. Install them with "
        "`python -m pip install -r requirements-cuda.txt`, then restart "
        "`./run-webapp.sh` so the launcher can add the NVIDIA library paths."
    )


def _download_root() -> Path:
    raw = os.getenv("WHISPER_DOWNLOAD_ROOT", "models/whisper-cache").strip()
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def _run_text_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _pulse_sources() -> list[str]:
    if shutil.which("pactl") is None:
        return []
    raw = _run_text_command(["pactl", "list", "short", "sources"])
    if not raw:
        return []
    sources = []
    for line in raw.splitlines():
        columns = line.split()
        if len(columns) >= 2:
            sources.append(columns[1])
    return sources


def _default_speaker_source() -> str:
    configured = (
        os.getenv("TRANSCRIPT_SPEAKER_SOURCE")
        or os.getenv("PULSE_MONITOR_SOURCE")
        or os.getenv("SPEAKER_SOURCE")
    )
    if configured:
        return configured

    sources = _pulse_sources()
    default_sink = _run_text_command(["pactl", "get-default-sink"])
    if default_sink:
        monitor = f"{default_sink}.monitor"
        if monitor in sources:
            return monitor

    for source in sources:
        if source.endswith(".monitor"):
            return source

    return "@DEFAULT_MONITOR@"


def _speaker_capture_available() -> bool:
    return shutil.which("parec") is not None


def _speaker_capture_command(source: str) -> list[str]:
    executable = shutil.which("parec")
    if executable is None:
        raise RuntimeError("Could not find 'parec'. Install pulseaudio-utils.")
    return [
        executable,
        "--record",
        "--device",
        source,
        "--format",
        "s16le",
        "--rate",
        str(DEFAULT_SAMPLE_RATE),
        "--channels",
        "1",
        "--latency-msec",
        "100",
        "--raw",
    ]


class WhisperRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: WhisperModel | None = None
        self._model_id: str | None = None
        self._device: str | None = None
        self._compute_type: str | None = None

    def get(self, model_id: str) -> WhisperModel:
        catalog = build_model_catalog()
        if model_id not in catalog:
            raise ValueError(f"Model is not available: {model_id}")

        entry = catalog[model_id]
        model_ref = str(entry.path) if entry.path is not None else entry.id
        device = _select_device()
        compute_type = _select_compute_type(device)
        if device == "cuda":
            missing = _missing_cuda_libraries()
            if missing:
                raise RuntimeError(_cuda_setup_error(missing))

        with self._lock:
            if (
                self._model is not None
                and self._model_id == model_ref
                and self._device == device
                and self._compute_type == compute_type
            ):
                return self._model

            self.unload_locked()
            self._model = WhisperModel(
                model_ref,
                device=device,
                compute_type=compute_type,
                cpu_threads=max(1, int(os.getenv("WHISPER_CPU_THREADS", "4"))),
                download_root=str(_download_root()),
            )
            self._model_id = model_ref
            self._device = device
            self._compute_type = compute_type
            return self._model

    def unload_locked(self) -> None:
        self._model = None
        self._model_id = None
        gc.collect()
        if torch is not None:
            try:
                if _cuda_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        device = _select_device()
        cuda_missing_libraries = _missing_cuda_libraries() if device == "cuda" else []
        payload: dict[str, Any] = {
            "status": "ok",
            "device": device,
            "compute_type": _select_compute_type(device),
            "model_loaded": self._model_id,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "speaker_capture_available": _speaker_capture_available(),
            "speaker_source": _default_speaker_source(),
            "cuda_ready": device != "cuda" or not cuda_missing_libraries,
            "cuda_missing_libraries": cuda_missing_libraries,
        }
        cuda_available = _cuda_available()
        payload["cuda_available"] = cuda_available
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    payload["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        return payload


runtime = WhisperRuntime()
app = FastAPI(title="Transcript Direct", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return runtime.status()


@app.get("/api/models")
def models() -> dict[str, Any]:
    entries = [entry.public() for entry in build_model_catalog().values()]
    return {
        "default_model": os.getenv("WHISPER_MODEL_NAME", "large-v3"),
        "models": entries,
    }


@app.get("/api/audio-sources")
def audio_sources() -> dict[str, Any]:
    return {
        "speaker_capture_available": _speaker_capture_available(),
        "speaker_source": _default_speaker_source(),
        "pulse_sources": _pulse_sources(),
    }


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def _decode_pcm16(payload: bytes) -> np.ndarray:
    if len(payload) < 2:
        return np.zeros(0, dtype=np.float32)
    if len(payload) % 2:
        payload = payload[:-1]
    audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _transcribe_chunk(
    model: WhisperModel,
    audio: np.ndarray,
    language: str | None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    segments_iter, info = model.transcribe(
        audio,
        language=language or None,
        beam_size=max(1, _env_int("WHISPER_BEAM_SIZE", DEFAULT_BEAM_SIZE)),
        temperature=0.0,
        vad_filter=_env_bool("WHISPER_INTERNAL_VAD", True),
        vad_parameters={
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 120,
        },
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.2,
        log_prob_threshold=-0.8,
        no_repeat_ngram_size=3,
        repetition_penalty=1.08,
        hallucination_silence_threshold=1.0,
    )
    segments = [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        for segment in segments_iter
        if segment.text.strip()
    ]
    return {
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
        "language": getattr(info, "language", language),
        "duration": float(getattr(info, "duration", audio.size / DEFAULT_SAMPLE_RATE)),
    }


def _normalize_language(language: str | None) -> str:
    normalized = (language or "en").strip().lower()
    if normalized not in {"es", "en"}:
        return "en"
    return normalized


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


class TranscriptContext:
    def __init__(self, *, max_words: int) -> None:
        self.max_words = max(0, int(max_words))
        self.words: list[str] = []

    def prompt(self) -> str | None:
        if self.max_words <= 0 or not self.words:
            return None
        return " ".join(self.words[-self.max_words :])

    def remember(self, text: str) -> None:
        if self.max_words <= 0:
            return
        new_words = text.strip().split()
        if not new_words:
            return
        self.words.extend(new_words)
        keep = self.max_words * 2
        if len(self.words) > keep:
            self.words = self.words[-keep:]


@dataclass(slots=True)
class PhraseSegment:
    index: int
    start: float
    end: float
    audio: np.ndarray
    rms: float
    reason: str
    paragraph_break_before: bool


class NaturalPhraseBuffer:
    def __init__(
        self,
        *,
        max_seconds: float,
        silence_seconds: float,
        paragraph_silence_seconds: float,
        speech_rms_threshold: float,
        adaptive_rms_multiplier: float,
    ) -> None:
        self.max_seconds = max(1.0, float(max_seconds))
        self.silence_seconds = max(0.3, float(silence_seconds))
        self.paragraph_silence_seconds = max(
            self.silence_seconds,
            float(paragraph_silence_seconds),
        )
        self.speech_rms_threshold = max(0.0001, float(speech_rms_threshold))
        self.adaptive_rms_multiplier = max(1.5, float(adaptive_rms_multiplier))
        self.noise_floor = self.speech_rms_threshold / self.adaptive_rms_multiplier
        self.min_speech_seconds = 0.15
        self.elapsed_sec = 0.0
        self.phrase_index = 0
        self.phrase_start_sec: float | None = None
        self.buffers: list[np.ndarray] = []
        self.frame_rms_values: list[float] = []
        self.speech_sec = 0.0
        self.trailing_silence_sec = 0.0
        self.idle_silence_sec = 0.0
        self.has_emitted = False
        self.paragraph_break_before_current = False

    @property
    def active(self) -> bool:
        return bool(self.buffers)

    def _speech_threshold(self) -> float:
        return max(
            self.speech_rms_threshold,
            self.noise_floor * self.adaptive_rms_multiplier,
        )

    def _learn_noise(self, rms: float) -> None:
        self.noise_floor = (self.noise_floor * 0.95) + (rms * 0.05)

    def append(self, audio: np.ndarray) -> PhraseSegment | None:
        duration_sec = audio.size / DEFAULT_SAMPLE_RATE
        started_at = self.elapsed_sec
        self.elapsed_sec += duration_sec

        if audio.size == 0:
            return None

        rms = _rms(audio)
        is_speech = rms >= self._speech_threshold()
        if not is_speech:
            self._learn_noise(rms)

        if not self.active and not is_speech:
            if self.has_emitted:
                self.idle_silence_sec += duration_sec
            return None

        if not self.active:
            self.phrase_start_sec = started_at
            self.paragraph_break_before_current = (
                self.has_emitted
                and self.idle_silence_sec >= self.paragraph_silence_seconds
            )
            self.idle_silence_sec = 0.0

        self.buffers.append(audio)
        self.frame_rms_values.append(rms)

        if is_speech:
            self.speech_sec += duration_sec
            self.trailing_silence_sec = 0.0
        else:
            self.trailing_silence_sec += duration_sec

        phrase_duration = self.elapsed_sec - (self.phrase_start_sec or started_at)
        if (
            self.speech_sec >= self.min_speech_seconds
            and self.trailing_silence_sec >= self.silence_seconds
        ):
            return self.flush(reason="pause")
        if phrase_duration >= self.max_seconds:
            return self.flush(reason="max_duration")
        return None

    def flush(self, *, reason: str = "flush") -> PhraseSegment | None:
        if not self.active:
            return None

        audio = np.concatenate(self.buffers).astype(np.float32, copy=False)
        start = self.phrase_start_sec if self.phrase_start_sec is not None else 0.0
        end = start + audio.size / DEFAULT_SAMPLE_RATE
        frame_rms = float(max(self.frame_rms_values or [0.0]))
        should_emit = (
            self.speech_sec >= self.min_speech_seconds
            and audio.size >= DEFAULT_SAMPLE_RATE // 2
        )

        segment = None
        if should_emit:
            self.phrase_index += 1
            segment = PhraseSegment(
                index=self.phrase_index,
                start=start,
                end=end,
                audio=audio,
                rms=frame_rms,
                reason=reason,
                paragraph_break_before=self.paragraph_break_before_current,
            )
            self.has_emitted = True

        idle_after_flush = self.trailing_silence_sec if reason == "pause" else 0.0
        self.phrase_start_sec = None
        self.buffers = []
        self.frame_rms_values = []
        self.speech_sec = 0.0
        self.trailing_silence_sec = 0.0
        self.idle_silence_sec = idle_after_flush
        self.paragraph_break_before_current = False
        return segment

    def discard(self) -> None:
        self.phrase_start_sec = None
        self.buffers = []
        self.frame_rms_values = []
        self.speech_sec = 0.0
        self.trailing_silence_sec = 0.0
        self.paragraph_break_before_current = False


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


async def _transcribe_phrase_segment(
    websocket: WebSocket,
    whisper: WhisperModel,
    segment: PhraseSegment,
    *,
    language: str | None,
    transcript_context: TranscriptContext,
) -> None:
    await _send_json(
        websocket,
        {
            "type": "processing",
            "chunk": segment.index,
            "phrase": segment.index,
            "start": segment.start,
            "end": segment.end,
            "message": "Processing phrase",
            "reason": segment.reason,
            "paragraph_break_before": segment.paragraph_break_before,
        },
    )
    result = await asyncio.to_thread(
        _transcribe_chunk,
        whisper,
        segment.audio,
        language,
        transcript_context.prompt(),
    )
    if not result["text"]:
        await _send_json(
            websocket,
            {
                "type": "empty_phrase",
                "chunk": segment.index,
                "phrase": segment.index,
                "start": segment.start,
                "end": segment.end,
                "rms": segment.rms,
                "paragraph_break_before": segment.paragraph_break_before,
            },
        )
        return
    transcript_context.remember(result["text"])

    result.update(
        {
            "type": "transcript",
            "chunk": segment.index,
            "phrase": segment.index,
            "start": segment.start,
            "end": segment.end,
            "rms": segment.rms,
            "reason": segment.reason,
            "paragraph_break_before": segment.paragraph_break_before,
        }
    )
    await _send_json(websocket, result)


async def _process_audio_payload(
    websocket: WebSocket,
    whisper: WhisperModel,
    phrase_buffer: NaturalPhraseBuffer,
    payload: bytes,
    *,
    language: str | None,
    transcript_context: TranscriptContext,
) -> None:
    audio = _decode_pcm16(payload)
    segment = phrase_buffer.append(audio)
    if segment is not None:
        await _transcribe_phrase_segment(
            websocket,
            whisper,
            segment,
            language=language,
            transcript_context=transcript_context,
        )


async def _flush_phrase_buffer(
    websocket: WebSocket,
    whisper: WhisperModel,
    phrase_buffer: NaturalPhraseBuffer,
    *,
    language: str | None,
    transcript_context: TranscriptContext,
) -> None:
    segment = phrase_buffer.flush(reason="stop")
    if segment is not None:
        await _transcribe_phrase_segment(
            websocket,
            whisper,
            segment,
            language=language,
            transcript_context=transcript_context,
        )


def _terminate_capture(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


async def _watch_stop_command(websocket: WebSocket, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                stop_event.set()
                return
            text_message = message.get("text")
            if not text_message:
                continue
            try:
                command = json.loads(text_message)
            except json.JSONDecodeError:
                continue
            if command.get("type") == "stop":
                stop_event.set()
                return
    except WebSocketDisconnect:
        stop_event.set()


async def _terminate_on_stop(
    stop_event: asyncio.Event,
    process: subprocess.Popen[bytes],
) -> None:
    await stop_event.wait()
    _terminate_capture(process)


async def _capture_speakers(
    websocket: WebSocket,
    whisper: WhisperModel,
    *,
    language: str | None,
    phrase_max_seconds: float,
) -> None:
    source = _default_speaker_source()
    command = _speaker_capture_command(source)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        _terminate_capture(process)
        raise RuntimeError("Could not open stdout for speaker capture.")

    stop_event = asyncio.Event()
    stop_task = asyncio.create_task(_watch_stop_command(websocket, stop_event))
    terminate_task = asyncio.create_task(_terminate_on_stop(stop_event, process))
    frame_bytes = max(1, int(DEFAULT_FRAME_SECONDS * DEFAULT_SAMPLE_RATE) * 2)
    phrase_buffer = NaturalPhraseBuffer(
        max_seconds=phrase_max_seconds,
        silence_seconds=_env_float(
            "TRANSCRIPT_PHRASE_SILENCE_SECONDS",
            DEFAULT_PHRASE_SILENCE_SECONDS,
        ),
        paragraph_silence_seconds=_env_float(
            "TRANSCRIPT_PARAGRAPH_SILENCE_SECONDS",
            DEFAULT_PARAGRAPH_SILENCE_SECONDS,
        ),
        speech_rms_threshold=_env_float(
            "TRANSCRIPT_SPEECH_RMS_THRESHOLD",
            DEFAULT_SPEECH_RMS_THRESHOLD,
        ),
        adaptive_rms_multiplier=_env_float(
            "TRANSCRIPT_ADAPTIVE_RMS_MULTIPLIER",
            DEFAULT_ADAPTIVE_RMS_MULTIPLIER,
        ),
    )
    transcript_context = TranscriptContext(
        max_words=_env_int("WHISPER_CONTEXT_WORDS", DEFAULT_CONTEXT_WORDS),
    )

    await _send_json(
        websocket,
        {
            "type": "status",
            "message": f"Capturing speakers from {source}",
            "speaker_source": source,
        },
    )

    try:
        while not stop_event.is_set():
            payload = await asyncio.to_thread(process.stdout.read, frame_bytes)
            if stop_event.is_set():
                break
            if not payload:
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().decode("utf-8", errors="ignore").strip()
                raise RuntimeError(stderr or "Speaker capture closed.")
            await _process_audio_payload(
                websocket,
                whisper,
                phrase_buffer,
                payload,
                language=language,
                transcript_context=transcript_context,
            )
        phrase_buffer.discard()
        with suppress(Exception):
            await _send_json(websocket, {"type": "stopped"})
    finally:
        stop_event.set()
        stop_task.cancel()
        terminate_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        with suppress(asyncio.CancelledError):
            await terminate_task
        _terminate_capture(process)


@app.websocket("/ws/transcribe")
async def transcribe_ws(
    websocket: WebSocket,
    model: str = Query(default="large-v3"),
    language: str = Query(default="en"),
    source: str = Query(default="browser"),
    chunk_seconds: float = Query(default=DEFAULT_PHRASE_MAX_SECONDS),
) -> None:
    await websocket.accept()
    safe_phrase_max_seconds = max(1.0, min(float(chunk_seconds), 30.0))
    selected_language = _normalize_language(language)
    phrase_buffer = NaturalPhraseBuffer(
        max_seconds=safe_phrase_max_seconds,
        silence_seconds=_env_float(
            "TRANSCRIPT_PHRASE_SILENCE_SECONDS",
            DEFAULT_PHRASE_SILENCE_SECONDS,
        ),
        paragraph_silence_seconds=_env_float(
            "TRANSCRIPT_PARAGRAPH_SILENCE_SECONDS",
            DEFAULT_PARAGRAPH_SILENCE_SECONDS,
        ),
        speech_rms_threshold=_env_float(
            "TRANSCRIPT_SPEECH_RMS_THRESHOLD",
            DEFAULT_SPEECH_RMS_THRESHOLD,
        ),
        adaptive_rms_multiplier=_env_float(
            "TRANSCRIPT_ADAPTIVE_RMS_MULTIPLIER",
            DEFAULT_ADAPTIVE_RMS_MULTIPLIER,
        ),
    )
    transcript_context = TranscriptContext(
        max_words=_env_int("WHISPER_CONTEXT_WORDS", DEFAULT_CONTEXT_WORDS),
    )

    try:
        await _send_json(
            websocket,
            {
                "type": "status",
                "message": f"Loading model {model}",
            },
        )
        whisper = await asyncio.to_thread(runtime.get, model)
        await _send_json(
            websocket,
            {
                "type": "ready",
                "message": "Ready to transcribe",
                "model": model,
                "language": selected_language,
                "sample_rate": DEFAULT_SAMPLE_RATE,
            },
        )

        if source == "speakers":
            await _capture_speakers(
                websocket,
                whisper,
                language=selected_language,
                phrase_max_seconds=safe_phrase_max_seconds,
            )
            return

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_message = message.get("text")
            if text_message:
                try:
                    command = json.loads(text_message)
                except json.JSONDecodeError:
                    continue
                if command.get("type") == "stop":
                    phrase_buffer.discard()
                    await _send_json(websocket, {"type": "stopped"})
                    break
                continue

            payload = message.get("bytes")
            if not payload:
                continue

            await _process_audio_payload(
                websocket,
                whisper,
                phrase_buffer,
                payload,
                language=selected_language,
                transcript_context=transcript_context,
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("Transcription WebSocket failed")
        await _send_json(
            websocket,
            {
                "type": "error",
                "message": str(exc),
            },
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
