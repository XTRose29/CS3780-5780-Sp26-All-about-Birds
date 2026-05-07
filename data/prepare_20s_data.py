#!/usr/bin/env python3

import argparse
import hashlib
import math
import subprocess
from pathlib import Path


TARGET_SECONDS = 20.0
DEFAULT_SAMPLE_RATE = 32000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("new_train_audio"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("new_train_audio_20s"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
    )
    parser.add_argument(
        "--files-per-label",
        type=int,
        default=None,
    )
    return parser.parse_args()


def run_command(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def get_duration_seconds(audio_path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
    )
    return float(result.stdout.decode("utf-8").strip())


def stable_random_start(audio_path: Path, duration: float, seed: int) -> float:
    max_start = max(duration - TARGET_SECONDS, 0.0)
    if max_start <= 0:
        return 0.0

    key = f"{audio_path.as_posix()}::{seed}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    random_u64 = int.from_bytes(digest[:8], "big")
    unit_value = random_u64 / float(2**64 - 1)
    return unit_value * max_start


def normalize_audio_to_output(
    source_path: Path,
    output_path: Path,
    sample_rate: int,
    seed: int,
) -> None:
    duration = get_duration_seconds(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_cmd = ["ffmpeg", "-y", "-loglevel", "error"]

    if duration > TARGET_SECONDS:
        start = stable_random_start(source_path, duration, seed)
        cmd = base_cmd + [
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source_path),
            "-t",
            f"{TARGET_SECONDS:.6f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "libvorbis",
            str(output_path),
        ]
    else:
        if duration <= 0:
            raise ValueError(f"Invalid non-positive duration for {source_path}")
        extra_loops = max(math.ceil(TARGET_SECONDS / duration) - 1, 0)
        cmd = base_cmd + [
            "-stream_loop",
            str(extra_loops),
            "-i",
            str(source_path),
            "-t",
            f"{TARGET_SECONDS:.6f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "libvorbis",
            str(output_path),
        ]

    run_command(cmd)


def gather_audio_files(data_root: Path, files_per_label: int | None) -> list[Path]:
    selected_files: list[Path] = []

    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label_files = sorted(
            label_dir.rglob("*.ogg"),
            key=lambda path: hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest(),
        )
        if files_per_label is not None:
            label_files = label_files[:files_per_label]
        selected_files.extend(label_files)

    return selected_files


def build_normalized_audio_files(
    audio_files: list[Path],
    data_root: Path,
    output_root: Path,
    sample_rate: int,
    seed: int,
) -> None:
    total = len(audio_files)

    for index, audio_path in enumerate(audio_files, start=1):
        relative_path = audio_path.relative_to(data_root)
        output_path = output_root / relative_path
        normalize_audio_to_output(
            source_path=audio_path,
            output_path=output_path,
            sample_rate=sample_rate,
            seed=seed,
        )
        if index % 100 == 0 or index == total:
            print(f"Prepared {index}/{total}: {output_path}")


def main() -> None:
    args = parse_args()
    audio_files = gather_audio_files(args.data_root, args.files_per_label)
    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found under {args.data_root}")

    print(f"Found {len(audio_files)} source audio files under {args.data_root}")
    print(
        f"Writing 20-second audio to {args.output_root} "
        f"at sample_rate={args.sample_rate}"
    )
    build_normalized_audio_files(
        audio_files=audio_files,
        data_root=args.data_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
