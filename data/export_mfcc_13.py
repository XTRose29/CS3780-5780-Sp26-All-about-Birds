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
DEFAULT_N_MELS = 40
DEFAULT_N_MFCC = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("new_train_audio"))
    parser.add_argument("--output-root", type=Path, default=Path("mfcc_13_export"))
    parser.add_argument("--files-per-label", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--n-mfcc", type=int, default=DEFAULT_N_MFCC)
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


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(np.array([0.0]))[0]
    mel_max = hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        if center == left:
            center = min(center + 1, n_freqs - 1)
        if right == center:
            right = min(right + 1, n_freqs)

        for j in range(left, center):
            filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            filters[i - 1, j] = (right - j) / max(right - center, 1)
    return filters


def compute_power_spectrogram(
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
    return (np.abs(zxx) ** 2).astype(np.float32)


def compute_mfcc(
    power_spec: np.ndarray,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    n_mfcc: int,
) -> np.ndarray:
    filters = mel_filterbank(sample_rate, n_fft, n_mels)
    mel_spec = filters @ power_spec
    log_mel = np.log1p(mel_spec)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc].astype(np.float32)


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
        power_spec = compute_power_spectrogram(
            samples=samples,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        mfcc = compute_mfcc(
            power_spec=power_spec,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
        )

        image_path = args.output_root / "mfcc_spectrogram" / audio_path.relative_to(args.data_root).with_suffix(".png")
        size = save_grayscale_image(mfcc, image_path)
        rows.append(
            {
                "label": audio_path.parent.name,
                "audio_file": str(audio_path),
                "image_file": str(image_path),
                "width": size[0],
                "height": size[1],
            }
        )
        if index % 100 == 0 or index == len(audio_files):
            print(f"{index}/{len(audio_files)} {audio_path.parent.name}: {size[0]}x{size[1]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "audio_file", "image_file", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report to {args.output_root / 'sizes.csv'}")


if __name__ == "__main__":
    main()
