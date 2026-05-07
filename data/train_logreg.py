#!/usr/bin/env python3

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.signal import stft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 320
DEFAULT_N_CEPSTRAL = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
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
        "--n-fft",
        type=int,
        default=DEFAULT_N_FFT,
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
    )
    parser.add_argument(
        "--n-cepstral",
        type=int,
        default=DEFAULT_N_CEPSTRAL,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("logreg_bird_model_split80_20.pkl"),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("logreg_metrics_split80_20.json"),
    )
    parser.add_argument(
        "--spectrogram-root",
        type=Path,
        default=Path("spectrogram_images"),
    )
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

    power_spec = np.abs(zxx) ** 2
    return np.log1p(power_spec)


def extract_feature_vector(
    log_spec: np.ndarray,
    n_cepstral: int,
) -> np.ndarray:
    cepstral = dct(log_spec, type=2, axis=0, norm="ortho")[:n_cepstral]

    spec_mean = log_spec.mean(axis=1)
    spec_std = log_spec.std(axis=1)
    cep_mean = cepstral.mean(axis=1)
    cep_std = cepstral.std(axis=1)

    return np.concatenate([spec_mean, spec_std, cep_mean, cep_std]).astype(np.float32)


def save_spectrogram_image(
    log_spec: np.ndarray,
    audio_path: Path,
    data_root: Path,
    spectrogram_root: Path,
) -> tuple[int, int]:
    spec_min = float(log_spec.min())
    spec_max = float(log_spec.max())
    if spec_max > spec_min:
        image_array = (255 * (log_spec - spec_min) / (spec_max - spec_min)).astype(np.uint8)
    else:
        image_array = np.zeros(log_spec.shape, dtype=np.uint8)

    image_array = np.flipud(image_array)
    relative_path = audio_path.relative_to(data_root).with_suffix(".png")
    image_path = spectrogram_root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(image_array, mode="L")
    image.save(image_path)
    return image.size


def gather_audio_files(data_root: Path) -> list[Path]:
    selected_files: list[Path] = []

    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label_files = sorted(label_dir.rglob("*.ogg"))
        selected_files.extend(label_files)

    return selected_files


def build_dataset(
    audio_files: list[Path],
    data_root: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_cepstral: int,
    spectrogram_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    image_sizes: list[tuple[int, int]] = []
    total = len(audio_files)

    for index, audio_path in enumerate(audio_files, start=1):
        samples = load_audio_with_ffmpeg(audio_path, sample_rate=sample_rate)
        log_spec = compute_log_spectrogram(
            samples=samples,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        feature_vector = extract_feature_vector(
            log_spec=log_spec,
            n_cepstral=n_cepstral,
        )
        image_size = save_spectrogram_image(
            log_spec=log_spec,
            audio_path=audio_path,
            data_root=data_root,
            spectrogram_root=spectrogram_root,
        )
        features.append(feature_vector)
        labels.append(audio_path.parent.name)
        image_sizes.append(image_size)

        if index % 100 == 0 or index == total:
            print(f"Extracted features {index}/{total}: {audio_path}")

    return np.stack(features), np.array(labels), image_sizes


def train_classifier(
    x: np.ndarray,
    y_labels: np.ndarray,
    test_size: float,
    max_iter: int,
    seed: int,
) -> tuple[Pipeline, LabelEncoder, dict]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=seed,
                    n_jobs=1,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(label_encoder.classes_)),
        "classification_report": report,
    }
    return model, label_encoder, metrics


def save_outputs(
    model: Pipeline,
    label_encoder: LabelEncoder,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    model_bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "config": {
            "data_root": str(args.data_root),
            "sample_rate": args.sample_rate,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_cepstral": args.n_cepstral,
            "seed": args.seed,
            "test_size": args.test_size,
            "spectrogram_root": str(args.spectrogram_root),
        },
    }

    with args.model_out.open("wb") as f:
        pickle.dump(model_bundle, f)

    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    audio_files = gather_audio_files(args.data_root)
    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found under {args.data_root}")

    print(f"Found {len(audio_files)} audio files under {args.data_root}")
    print(
        "Feature setup: "
        f"sample_rate={args.sample_rate}, n_fft={args.n_fft}, "
        f"hop_length={args.hop_length}, n_cepstral={args.n_cepstral}"
    )

    x, y_labels, image_sizes = build_dataset(
        audio_files,
        data_root=args.data_root,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_cepstral=args.n_cepstral,
        spectrogram_root=args.spectrogram_root,
    )
    model, label_encoder, metrics = train_classifier(
        x=x,
        y_labels=y_labels,
        test_size=args.test_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    unique_image_sizes = sorted(set(image_sizes))
    metrics["spectrogram_image_size"] = {
        "unique_sizes": [list(size) for size in unique_image_sizes],
        "all_same": len(unique_image_sizes) == 1,
    }
    save_outputs(model, label_encoder, metrics, args)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Spectrogram image size(s): {metrics['spectrogram_image_size']['unique_sizes']}")
    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
