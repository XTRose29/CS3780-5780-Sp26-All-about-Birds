#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import stft


DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 320
TARGET_SECONDS = 20.0
SHORT_SECONDS = 10.0
LOW_FREQ_DROP_RATIO = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("new_train_audio"))
    parser.add_argument("--output-root", type=Path, default=Path("ab_spectrogram_export"))
    parser.add_argument("--files-per-label", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--seed", type=int, default=2026)
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


def stable_random_start(audio_path: Path, duration: float, seed: int, target_seconds: float) -> float:
    max_start = max(duration - target_seconds, 0.0)
    if max_start <= 0:
        return 0.0
    key = f"{audio_path.as_posix()}::{seed}::{target_seconds}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    random_u64 = int.from_bytes(digest[:8], "big")
    unit_value = random_u64 / float(2**64 - 1)
    return unit_value * max_start


def load_fixed_length_audio(
    audio_path: Path,
    sample_rate: int,
    target_seconds: float,
    seed: int,
) -> np.ndarray:
    duration = get_duration_seconds(audio_path)
    target_samples = int(round(sample_rate * target_seconds))
    base_cmd = ["ffmpeg", "-loglevel", "error"]

    if duration > target_seconds:
        start = stable_random_start(audio_path, duration, seed, target_seconds)
        cmd = base_cmd + [
            "-ss",
            f"{start:.6f}",
            "-i",
            str(audio_path),
            "-t",
            f"{target_seconds:.6f}",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]
    else:
        if duration <= 0:
            raise ValueError(f"Invalid non-positive duration for {audio_path}")
        extra_loops = max(math.ceil(target_seconds / duration) - 1, 0)
        cmd = base_cmd + [
            "-stream_loop",
            str(extra_loops),
            "-i",
            str(audio_path),
            "-t",
            f"{target_seconds:.6f}",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]

    result = run_command(cmd, capture_output=True)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"Decoded empty audio for {audio_path}")
    if samples.size < target_samples:
        samples = np.pad(samples, (0, target_samples - samples.size))
    elif samples.size > target_samples:
        samples = samples[:target_samples]
    return samples


def compute_log_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    noverlap = n_fft - hop_length
    _, _, zxx = stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    return np.log1p(np.abs(zxx) ** 2).astype(np.float32)


def save_grayscale_image(spec: np.ndarray, path: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec_min = float(spec.min())
    spec_max = float(spec.max())
    if spec_max > spec_min:
        image_array = (255 * (spec - spec_min) / (spec_max - spec_min)).astype(np.uint8)
    else:
        image_array = np.zeros(spec.shape, dtype=np.uint8)
    image_array = np.flipud(image_array)
    image = Image.fromarray(image_array, mode="L")
    image.save(path)
    return image.size


def gather_audio_files(data_root: Path, files_per_label: int) -> list[Path]:
    selected_files: list[Path] = []
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label_files = sorted(
            label_dir.rglob("*.ogg"),
            key=lambda path: hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest(),
        )
        selected_files.extend(label_files[:files_per_label])
    return selected_files


def main() -> None:
    args = parse_args()
    audio_files = gather_audio_files(args.data_root, args.files_per_label)
    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found under {args.data_root}")

    rows: list[dict[str, str | int | float]] = []

    for index, audio_path in enumerate(audio_files, start=1):
        samples_10 = load_fixed_length_audio(
            audio_path=audio_path,
            sample_rate=args.sample_rate,
            target_seconds=SHORT_SECONDS,
            seed=args.seed,
        )
        spec_10 = compute_log_spectrogram(
            samples=samples_10,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )

        samples_20 = load_fixed_length_audio(
            audio_path=audio_path,
            sample_rate=args.sample_rate,
            target_seconds=TARGET_SECONDS,
            seed=args.seed,
        )
        spec_20 = compute_log_spectrogram(
            samples=samples_20,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        drop_rows = int(spec_20.shape[0] * LOW_FREQ_DROP_RATIO)
        spec_crop = spec_20[drop_rows:, :]

        relative_png = audio_path.relative_to(args.data_root).with_suffix(".png")
        path_a = args.output_root / "a_10sec" / relative_png
        path_b = args.output_root / "b_crop_low_freq" / relative_png
        size_a = save_grayscale_image(spec_10, path_a)
        size_b = save_grayscale_image(spec_crop, path_b)

        rows.append(
            {
                "label": audio_path.parent.name,
                "audio_file": str(audio_path),
                "option_a_image": str(path_a),
                "option_a_width": size_a[0],
                "option_a_height": size_a[1],
                "option_b_image": str(path_b),
                "option_b_width": size_b[0],
                "option_b_height": size_b[1],
                "low_freq_drop_ratio": LOW_FREQ_DROP_RATIO,
            }
        )

        print(
            f"{index}/{len(audio_files)} {audio_path.parent.name}: "
            f"a={size_a[0]}x{size_a[1]}, b={size_b[0]}x{size_b[1]}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "audio_file",
                "option_a_image",
                "option_a_width",
                "option_a_height",
                "option_b_image",
                "option_b_width",
                "option_b_height",
                "low_freq_drop_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report to {args.output_root / 'sizes.csv'}")


if __name__ == "__main__":
    main()
