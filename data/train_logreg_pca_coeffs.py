#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from export_pca_spectrograms import (
    TARGET_SECONDS,
    compute_log_spectrogram,
    gather_audio_files,
    load_fixed_length_audio,
)


DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("new_train_audio"))
    parser.add_argument("--output", type=Path, default=Path("pca_coeff_logreg_metrics.json"))
    parser.add_argument("--files-per-label", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--pca-components", type=int, default=64)
    return parser.parse_args()


def train_eval(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_size: float,
    max_iter: int,
) -> dict:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    print(f"[start] training logistic regression on PCA coefficients with d={x.shape[1]}", flush=True)
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
                    verbose=1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    print(f"[done] training logistic regression on PCA coefficients with d={x.shape[1]}", flush=True)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    report = classification_report(
        y_test,
        test_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    return {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(label_encoder.classes_)),
        "classification_report": report,
    }


def main() -> None:
    args = parse_args()
    audio_files = gather_audio_files(args.data_root, args.files_per_label)
    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found under {args.data_root}")

    labels = np.array([path.parent.name for path in audio_files])
    flat_specs: list[np.ndarray] = []

    print(f"Selected {len(audio_files)} files from {args.data_root}")
    for audio_path in tqdm(audio_files, desc="Extracting spectrograms", unit="file"):
        samples = load_fixed_length_audio(audio_path, args.sample_rate, args.seed)
        spec = compute_log_spectrogram(
            samples=samples,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        flat_specs.append(spec.reshape(-1).astype(np.float32))

    x_raw = np.stack(flat_specs)
    max_valid_components = min(x_raw.shape[0], x_raw.shape[1])
    n_components = min(args.pca_components, max_valid_components)

    print(f"[start] fitting PCA with n_components={n_components}", flush=True)
    pca = PCA(n_components=n_components, random_state=args.seed)
    x_pca = pca.fit_transform(x_raw).astype(np.float32)
    print(f"[done] fitting PCA with n_components={n_components}", flush=True)

    metrics = train_eval(
        x=x_pca,
        labels=labels,
        seed=args.seed,
        test_size=args.test_size,
        max_iter=args.max_iter,
    )

    result = {
        "config": {
            "data_root": str(args.data_root),
            "files_per_label": args.files_per_label,
            "sample_rate": args.sample_rate,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "target_seconds": TARGET_SECONDS,
            "pca_components": n_components,
        },
        "metrics": {
            **metrics,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"train={metrics['train_accuracy']:.4f}, "
        f"test={metrics['test_accuracy']:.4f}, "
        f"d={metrics['n_features']}"
    )
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
