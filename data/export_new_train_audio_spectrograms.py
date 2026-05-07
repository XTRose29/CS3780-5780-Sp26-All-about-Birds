#!/usr/bin/env python3

import argparse
import csv
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import stft

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
from matplotlib import colormaps


DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("new_train_audio"))
    parser.add_argument("--output-root", type=Path, default=Path("new_train_audio_spectrograms"))
    parser.add_argument("--report-out", type=Path, default=Path("new_train_audio_spectrogram_sizes.csv"))
    parser.add_argument("--files-per-label", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--cmap", type=str, default="magma")
    return parser.parse_args()


def run_command(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def load_audio_with_ffmpeg(audio_path: Path, sample_rate: int) -> np.ndarray:
    result = run_command(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ],
        capture_output=True,
    )
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"Decoded empty audio for {audio_path}")
    return samples


def compute_raw_spectrogram(
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
    return np.abs(zxx).astype(np.float32)


def save_colored_spectrogram(
    spec: np.ndarray,
    image_path: Path,
    cmap_name: str,
) -> tuple[int, int]:
    image_path.parent.mkdir(parents=True, exist_ok=True)

    spec_min = float(spec.min())
    spec_max = float(spec.max())
    if spec_max > spec_min:
        norm = (spec - spec_min) / (spec_max - spec_min)
    else:
        norm = np.zeros(spec.shape, dtype=np.float32)

    norm = np.flipud(norm)
    rgba = colormaps[cmap_name](norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    image.save(image_path)
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

    rows: list[dict[str, str | int]] = []

    for index, audio_path in enumerate(audio_files, start=1):
        samples = load_audio_with_ffmpeg(audio_path, sample_rate=args.sample_rate)
        spec = compute_raw_spectrogram(
            samples=samples,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        image_path = args.output_root / audio_path.relative_to(args.data_root).with_suffix(".png")
        size = save_colored_spectrogram(spec, image_path=image_path, cmap_name=args.cmap)
        rows.append(
            {
                "label": audio_path.parent.name,
                "audio_file": str(audio_path),
                "image_file": str(image_path),
                "width": size[0],
                "height": size[1],
            }
        )
        print(f"{index}/{len(audio_files)} {audio_path.parent.name}: {size[0]} x {size[1]} -> {image_path}")

    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "audio_file", "image_file", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report to {args.report_out}")


if __name__ == "__main__":
    main()
