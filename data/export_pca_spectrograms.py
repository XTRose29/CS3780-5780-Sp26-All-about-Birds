#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import stft
from sklearn.decomposition import PCA


DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 320
TARGET_SECONDS = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("new_train_audio"))
    parser.add_argument("--output-root", type=Path, default=Path("pca_spectrogram_export"))
    parser.add_argument("--files-per-label", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pca-components", type=int, default=64)
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

    spectrograms: list[np.ndarray] = []
    rows: list[dict[str, str | int | float]] = []

    print(f"Selected {len(audio_files)} files from {args.data_root}")
    for index, audio_path in enumerate(audio_files, start=1):
        samples = load_fixed_length_audio(audio_path, args.sample_rate, args.seed)
        spec = compute_log_spectrogram(
            samples=samples,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        spectrograms.append(spec)
        if index % 20 == 0 or index == len(audio_files):
            print(f"Loaded {index}/{len(audio_files)}: {audio_path}")

    spec_array = np.stack(spectrograms)
    flat_specs = spec_array.reshape(spec_array.shape[0], -1)
    max_valid_components = min(flat_specs.shape[0], flat_specs.shape[1])
    n_components = min(args.pca_components, max_valid_components)
    pca = PCA(n_components=n_components, random_state=args.seed)
    transformed = pca.fit_transform(flat_specs).astype(np.float32)
    reconstructed = pca.inverse_transform(transformed).reshape(spec_array.shape)

    for audio_path, raw_spec, pca_spec in zip(audio_files, spec_array, reconstructed):
        relative_png = audio_path.relative_to(args.data_root).with_suffix(".png")
        raw_path = args.output_root / "raw_spectrogram" / relative_png
        pca_path = args.output_root / f"pca_{n_components}" / relative_png
        raw_size = save_grayscale_image(raw_spec, raw_path)
        pca_size = save_grayscale_image(pca_spec.astype(np.float32), pca_path)
        rows.append(
            {
                "label": audio_path.parent.name,
                "audio_file": str(audio_path),
                "raw_image": str(raw_path),
                "raw_width": raw_size[0],
                "raw_height": raw_size[1],
                "pca_image": str(pca_path),
                "pca_width": pca_size[0],
                "pca_height": pca_size[1],
                "pca_components": n_components,
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "audio_file",
                "raw_image",
                "raw_width",
                "raw_height",
                "pca_image",
                "pca_width",
                "pca_height",
                "pca_components",
                "explained_variance_ratio_sum",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": str(args.data_root),
                "files_per_label": args.files_per_label,
                "sample_rate": args.sample_rate,
                "n_fft": args.n_fft,
                "hop_length": args.hop_length,
                "target_seconds": TARGET_SECONDS,
                "pca_components": n_components,
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            },
            f,
            indent=2,
        )

    print(f"Saved report to {args.output_root / 'sizes.csv'}")
    print(f"Saved metadata to {args.output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
