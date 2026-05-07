#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("lowdim_export"))
    parser.add_argument("--output", type=Path, default=Path("lowdim_export_logreg_metrics.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def gather_variant_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def load_variant_dataset(variant_dir: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    image_size: tuple[int, int] | None = None
    image_paths = sorted(variant_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found under {variant_dir}")

    for image_path in tqdm(image_paths, desc=f"Loading {variant_dir.name}", unit="img"):
        with Image.open(image_path) as image:
            gray = image.convert("L")
            if image_size is None:
                image_size = gray.size
            elif gray.size != image_size:
                raise ValueError(
                    f"Inconsistent image sizes in {variant_dir}: "
                    f"{gray.size} vs {image_size} at {image_path}"
                )
            features.append(np.asarray(gray, dtype=np.float32).reshape(-1))
        labels.append(image_path.parent.name)

    return np.stack(features), np.array(labels), image_size


def train_eval(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_size: float,
    max_iter: int,
) -> dict:
    print(f"[start] training {x.shape[1]}-dim logistic regression", flush=True)
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
    print(f"[done] training {x.shape[1]}-dim logistic regression", flush=True)

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
    variant_dirs = gather_variant_dirs(args.data_root)
    if not variant_dirs:
        raise FileNotFoundError(f"No variant folders found under {args.data_root}")

    results = {
        "config": {
            "data_root": str(args.data_root),
            "seed": args.seed,
            "test_size": args.test_size,
            "max_iter": args.max_iter,
        },
        "variants": {},
    }

    variant_bar = tqdm(variant_dirs, desc="Variants", unit="variant")
    for variant_dir in variant_bar:
        variant_bar.set_postfix_str(variant_dir.name)
        x, labels, image_size = load_variant_dataset(variant_dir)
        metrics = train_eval(
            x=x,
            labels=labels,
            seed=args.seed,
            test_size=args.test_size,
            max_iter=args.max_iter,
        )
        metrics["image_size"] = list(image_size)
        results["variants"][variant_dir.name] = metrics
        print(
            f"{variant_dir.name}: "
            f"train={metrics['train_accuracy']:.4f}, "
            f"test={metrics['test_accuracy']:.4f}, "
            f"d={metrics['n_features']}"
        )

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
