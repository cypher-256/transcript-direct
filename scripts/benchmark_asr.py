from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    build_model_catalog,
    _select_compute_type,
    _select_device,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "benchmark_data" / "ami_2speaker_test" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"
WORD_RE = re.compile(r"[a-z0-9']+")

CONFIGS: dict[str, dict[str, Any]] = {
    "live-3s-beam5": {
        "mode": "chunked",
        "phrase_seconds": 3.0,
        "beam_size": 5,
        "vad_filter": True,
        "context_words": 0,
        "description": "3s baseline without text context.",
    },
    "live-3s-context24": {
        "mode": "chunked",
        "phrase_seconds": 3.0,
        "beam_size": 5,
        "vad_filter": True,
        "context_words": 24,
        "description": "Current recommended live default.",
    },
    "live-3s-beam8": {
        "mode": "chunked",
        "phrase_seconds": 3.0,
        "beam_size": 8,
        "vad_filter": True,
        "context_words": 0,
        "description": "3s chunks with wider beam search.",
    },
    "live-4s-beam5": {
        "mode": "chunked",
        "phrase_seconds": 4.0,
        "beam_size": 5,
        "vad_filter": True,
        "context_words": 0,
        "description": "Longer chunks for more context and slightly higher latency.",
    },
    "live-3s-no-vad": {
        "mode": "chunked",
        "phrase_seconds": 3.0,
        "beam_size": 5,
        "vad_filter": False,
        "context_words": 0,
        "description": "3s chunks without faster-whisper internal VAD.",
    },
    "live-2s-beam5": {
        "mode": "chunked",
        "phrase_seconds": 2.0,
        "beam_size": 5,
        "vad_filter": True,
        "context_words": 0,
        "description": "Lower latency, usually less context.",
    },
    "live-1s-beam5": {
        "mode": "chunked",
        "phrase_seconds": 1.0,
        "beam_size": 5,
        "vad_filter": True,
        "context_words": 0,
        "description": "Very low latency, expected to lose accuracy.",
    },
    "offline-full-beam5": {
        "mode": "full",
        "phrase_seconds": None,
        "beam_size": 5,
        "vad_filter": True,
        "context_words": 0,
        "description": "Upper-bound single-pass transcription for each clip.",
    },
}


def read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + substitution_cost,
            )
        previous = current
    return previous[-1]


def word_metrics(reference_text: str, hypothesis_text: str) -> dict[str, float | int]:
    reference = normalize_words(reference_text)
    hypothesis = normalize_words(hypothesis_text)
    errors = edit_distance(reference, hypothesis)
    ref_count = len(reference)
    hyp_count = len(hypothesis)

    ref_counter = Counter(reference)
    hyp_counter = Counter(hypothesis)
    overlap = sum((ref_counter & hyp_counter).values())

    precision = overlap / hyp_count if hyp_count else 0.0
    recall = overlap / ref_count if ref_count else 0.0
    return {
        "wer": errors / ref_count if ref_count else math.inf,
        "edit_errors": errors,
        "reference_words": ref_count,
        "hypothesis_words": hyp_count,
        "bag_precision": precision,
        "bag_recall": recall,
        "bag_f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }


def load_whisper_model(model_id: str) -> tuple[WhisperModel, dict[str, str]]:
    catalog = build_model_catalog()
    if model_id not in catalog:
        available = ", ".join(sorted(catalog))
        raise SystemExit(f"Model '{model_id}' is not available. Available: {available}")

    entry = catalog[model_id]
    model_ref = str(entry.path) if entry.path is not None else entry.id
    device = _select_device()
    compute_type = _select_compute_type(device)
    model = WhisperModel(
        model_ref,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(1, int(os.getenv("WHISPER_CPU_THREADS", "4"))),
        download_root=str(PROJECT_ROOT / "models" / "whisper-cache"),
    )
    return model, {"model_ref": model_ref, "device": device, "compute_type": compute_type}


def transcribe_audio(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    language: str,
    beam_size: int,
    vad_filter: bool,
    initial_prompt: str | None = None,
) -> str:
    segments_iter, _info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        temperature=0.0,
        vad_filter=vad_filter,
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
    return " ".join(segment.text.strip() for segment in segments_iter if segment.text.strip()).strip()


def run_sample(model: WhisperModel, row: dict[str, Any], config: dict[str, Any], language: str) -> dict[str, Any]:
    audio_path = PROJECT_ROOT / row["audio_path"]
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise RuntimeError(f"{audio_path} sample rate is {sample_rate}, expected {DEFAULT_SAMPLE_RATE}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    chunk_times: list[float] = []
    chunks = 1
    started = time.perf_counter()
    if config["mode"] == "full":
        hypothesis = transcribe_audio(
            model,
            audio,
            language=language,
            beam_size=config["beam_size"],
            vad_filter=config["vad_filter"],
            initial_prompt=None,
        )
        chunk_times.append(time.perf_counter() - started)
    else:
        chunk_samples = int(float(config["phrase_seconds"]) * sample_rate)
        chunks_text: list[str] = []
        context_words: list[str] = []
        max_context_words = max(0, int(config.get("context_words", 0)))
        chunks = 0
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            if len(chunk) < int(0.5 * sample_rate):
                continue
            chunks += 1
            chunk_started = time.perf_counter()
            text = transcribe_audio(
                model,
                chunk,
                language=language,
                beam_size=config["beam_size"],
                vad_filter=config["vad_filter"],
                initial_prompt=" ".join(context_words[-max_context_words:]) if max_context_words else None,
            )
            chunk_times.append(time.perf_counter() - chunk_started)
            if text:
                chunks_text.append(text)
                if max_context_words:
                    context_words.extend(normalize_words(text))
        hypothesis = " ".join(chunks_text).strip()

    elapsed = time.perf_counter() - started
    duration = len(audio) / float(sample_rate)
    metrics = word_metrics(row["reference_text"], hypothesis)
    return {
        "id": row["id"],
        "duration_seconds": duration,
        "chunks": chunks,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / duration if duration else math.inf,
        "avg_chunk_latency_seconds": statistics.mean(chunk_times) if chunk_times else 0.0,
        "max_chunk_latency_seconds": max(chunk_times) if chunk_times else 0.0,
        "reference_text": row["reference_text"],
        "hypothesis_text": hypothesis,
        "metrics": metrics,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def aggregate(config_name: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_errors = sum(sample["metrics"]["edit_errors"] for sample in samples)
    total_ref_words = sum(sample["metrics"]["reference_words"] for sample in samples)
    total_duration = sum(sample["duration_seconds"] for sample in samples)
    total_elapsed = sum(sample["elapsed_seconds"] for sample in samples)
    chunk_latencies = [sample["avg_chunk_latency_seconds"] for sample in samples]
    return {
        "config": config_name,
        "samples": len(samples),
        "duration_seconds": total_duration,
        "elapsed_seconds": total_elapsed,
        "wer": total_errors / total_ref_words if total_ref_words else math.inf,
        "rtf": total_elapsed / total_duration if total_duration else math.inf,
        "avg_chunk_latency_seconds": statistics.mean(chunk_latencies) if chunk_latencies else 0.0,
        "p95_chunk_latency_seconds": percentile(chunk_latencies, 0.95),
        "bag_f1": statistics.mean(sample["metrics"]["bag_f1"] for sample in samples) if samples else 0.0,
        "bag_recall": statistics.mean(sample["metrics"]["bag_recall"] for sample in samples) if samples else 0.0,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ASR benchmark",
        "",
        f"Created: {payload['created_at']}",
        f"Model: `{payload['model']}`",
        f"Runtime: `{payload['runtime']['device']} / {payload['runtime']['compute_type']}`",
        f"Dataset: `{payload['dataset']}`",
        "",
        "| Config | Samples | WER | Bag F1 | RTF | Avg chunk latency | P95 chunk latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["summary"]:
        lines.append(
            "| {config} | {samples} | {wer:.3f} | {bag_f1:.3f} | {rtf:.3f} | {avg:.3f}s | {p95:.3f}s |".format(
                config=item["config"],
                samples=item["samples"],
                wer=item["wer"],
                bag_f1=item["bag_f1"],
                rtf=item["rtf"],
                avg=item["avg_chunk_latency_seconds"],
                p95=item["p95_chunk_latency_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "WER is strict and order-sensitive. Bag F1 is looser and useful here",
            "because overlapping speakers can produce valid words in a different order.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local faster-whisper settings on overlapping English speech.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="en")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--configs",
        default="live-3s-context24,live-3s-beam5,live-2s-beam5,offline-full-beam5",
        help=f"Comma-separated config names. Available: {', '.join(CONFIGS)}",
    )
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    rows = read_manifest(manifest_path, args.limit if args.limit > 0 else None)
    if not rows:
        raise SystemExit(f"No benchmark rows found in {manifest_path}")

    requested_configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    unknown_configs = [item for item in requested_configs if item not in CONFIGS]
    if unknown_configs:
        raise SystemExit(f"Unknown config(s): {', '.join(unknown_configs)}")

    model, runtime = load_whisper_model(args.model)
    results: dict[str, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []

    for config_name in requested_configs:
        config = CONFIGS[config_name]
        config_results = []
        print(f"\n[{config_name}] {config['description']}", flush=True)
        for index, row in enumerate(rows, start=1):
            sample = run_sample(model, row, config, args.language)
            config_results.append(sample)
            print(
                f"  {index:02d}/{len(rows)} {row['id']} "
                f"wer={sample['metrics']['wer']:.3f} rtf={sample['rtf']:.3f} "
                f"chunks={sample['chunks']}",
                flush=True,
            )
        results[config_name] = config_results
        summary.append(aggregate(config_name, config_results))

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "created_at": created_at,
        "model": args.model,
        "runtime": runtime,
        "dataset": str(manifest_path.relative_to(PROJECT_ROOT)),
        "configs": {name: CONFIGS[name] for name in requested_configs},
        "summary": summary,
        "results": results,
    }

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = DEFAULT_OUTPUT_DIR / f"asr_benchmark_{stamp}.json"
    md_path = DEFAULT_OUTPUT_DIR / f"asr_benchmark_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)

    print("\nSummary")
    for item in summary:
        print(
            f"{item['config']}: WER={item['wer']:.3f} "
            f"BagF1={item['bag_f1']:.3f} RTF={item['rtf']:.3f} "
            f"AvgLatency={item['avg_chunk_latency_seconds']:.3f}s"
        )
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
