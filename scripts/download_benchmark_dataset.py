from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download


DATASET_REPO = "Trelis/ami-2speaker-test"
DATASET_FILE = "data/train-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = Path("benchmark_data") / "ami_2speaker_test"
TIMED_TEXT_RE = re.compile(r"<\|([0-9.]+)\|>\s*(.*?)\s*<\|([0-9.]+)\|>", re.DOTALL)


def timed_target_to_utterances(target: str, fallback: str) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    for match in TIMED_TEXT_RE.finditer(target or ""):
        text = " ".join(match.group(2).split()).strip()
        if not text:
            continue
        utterances.append(
            {
                "start": float(match.group(1)),
                "end": float(match.group(3)),
                "text": text,
            }
        )
    if utterances:
        return utterances
    fallback_text = " ".join((fallback or "").split()).strip()
    return [{"start": 0.0, "end": 0.0, "text": fallback_text}] if fallback_text else []


def chronological_reference(row: dict[str, Any]) -> str:
    utterances = []
    utterances.extend(timed_target_to_utterances(row.get("speaker1_target", ""), row.get("speaker1_text", "")))
    utterances.extend(timed_target_to_utterances(row.get("speaker2_target", ""), row.get("speaker2_text", "")))
    utterances.sort(key=lambda item: (item["start"], item["end"]))
    return " ".join(item["text"] for item in utterances).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small overlapping-speaker ASR benchmark dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=DATASET_FILE,
    )
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index, row in enumerate(rows):
            audio_bytes = row["audio"]["bytes"]
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            audio_path = audio_dir / f"{index:04d}.wav"
            sf.write(audio_path, audio, sample_rate)

            duration_seconds = len(audio) / float(sample_rate)
            record = {
                "id": f"ami-2speaker-{index:04d}",
                "audio_path": str(audio_path.relative_to(project_root)),
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
                "reference_text": chronological_reference(row),
                "speaker1_text": row.get("speaker1_text", ""),
                "speaker2_text": row.get("speaker2_text", ""),
                "speaker1_start": row.get("speaker1_start"),
                "speaker1_end": row.get("speaker1_end"),
                "speaker2_start": row.get("speaker2_start"),
                "speaker2_end": row.get("speaker2_end"),
                "overlap_ratio": row.get("overlap_ratio"),
                "source_dataset": DATASET_REPO,
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# AMI 2-speaker ASR benchmark",
                "",
                f"Source: https://huggingface.co/datasets/{DATASET_REPO}",
                f"Rows: {len(rows)}",
                "",
                "The manifest joins both speaker transcripts in chronological order.",
                "This is intentionally harder than clean single-speaker speech and is",
                "useful for checking how Whisper behaves with overlapping conversation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Downloaded {len(rows)} clips")
    print(f"Manifest: {manifest_path}")
    print(f"Audio dir: {audio_dir}")


if __name__ == "__main__":
    main()
