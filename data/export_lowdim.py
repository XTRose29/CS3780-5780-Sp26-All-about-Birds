#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.signal import stft


DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 320
TARGET_SECONDS = 20.0
REDUCED_SHAPES = [
    (32, 64),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("new_train_audio"))
    parser.add_argument("--output-root", type=Path, default=Path("lowdim_export"))
    parser.add_argument("--files-per-label", type=int, default=None)
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


def stable_random_start(audio_path: Path, duration: float, seed: int) -> float:
    max_start = max(duration - TARGET_SECONDS, 0.0)
    if max_start <= 0:
        return 0.0
    key = f"{audio_path.as_posix()}::{seed}::{TARGET_SECONDS}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    random_u64 = int.from_bytes(digest[:8], "big")
    unit_value = random_u64 / float(2**64 - 1)
    return unit_value * max_start


def load_fixed_length_audio(audio_path: Path, sample_rate: int, seed: int) -> np.ndarray:
    duration = get_duration_seconds(audio_path)
    target_samples = int(round(sample_rate * TARGET_SECONDS))
    base_cmd = ["ffmpeg", "-loglevel", "error"]

    if duration > TARGET_SECONDS:
        start = stable_random_start(audio_path, duration, seed)
        cmd = base_cmd + [
            "-ss",
            f"{start:.6f}",
            "-i",
            str(audio_path),
            "-t",
            f"{TARGET_SECONDS:.6f}",
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
        extra_loops = max(math.ceil(TARGET_SECONDS / duration) - 1, 0)
        cmd = base_cmd + [
            "-stream_loop",
            str(extra_loops),
            "-i",
            str(audio_path),
            "-t",
            f"{TARGET_SECONDS:.6f}",
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


def dct2(spec: np.ndarray) -> np.ndarray:
    return dct(dct(spec, axis=0, norm="ortho"), axis=1, norm="ortho")


def reduce_spectrogram(spec: np.ndarray, rows: int, cols: int) -> np.ndarray:
    coeffs = dct2(spec)
    return coeffs[:rows, :cols].astype(np.float32)


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


def main() -> None:
    args = parse_args()
    audio_files = gather_audio_files(args.data_root, args.files_per_label)
    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found under {args.data_root}")

    rows: list[dict[str, str | int]] = []

    for index, audio_path in enumerate(audio_files, start=1):
        samples = load_fixed_length_audio(audio_path, args.sample_rate, args.seed)
        original_spec = compute_log_spectrogram(
            samples=samples,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        relative_png = audio_path.relative_to(args.data_root).with_suffix(".png")
        original_path = args.output_root / "original_spectrogram" / relative_png

        original_size = save_grayscale_image(original_spec, original_path)
        for rows_dim, cols_dim in REDUCED_SHAPES:
            reduced_spec = reduce_spectrogram(original_spec, rows_dim, cols_dim)
            reduced_path = args.output_root / f"reduced_{cols_dim}x{rows_dim}" / relative_png
            reduced_size = save_grayscale_image(reduced_spec, reduced_path)
            rows.append(
                {
                    "label": audio_path.parent.name,
                    "audio_file": str(audio_path),
                    "variant": f"{cols_dim}x{rows_dim}",
                    "original_image": str(original_path),
                    "original_width": original_size[0],
                    "original_height": original_size[1],
                    "reduced_image": str(reduced_path),
                    "reduced_width": reduced_size[0],
                    "reduced_height": reduced_size[1],
                    "reduced_feature_width": cols_dim,
                    "reduced_feature_height": rows_dim,
                }
            )

        print(
            f"{index}/{len(audio_files)} {audio_path.parent.name}: "
            f"original={original_size[0]}x{original_size[1]}, "
            "reduced=64*128, 64*256, 128*256, 256*512"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "audio_file",
                "variant",
                "original_image",
                "original_width",
                "original_height",
                "reduced_image",
                "reduced_width",
                "reduced_height",
                "reduced_feature_width",
                "reduced_feature_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report to {args.output_root / 'sizes.csv'}")


if __name__ == "__main__":
    main()
