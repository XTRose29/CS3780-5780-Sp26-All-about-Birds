#!/usr/bin/env python3

import argparse
import csv
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("lowdim_export/reduced_64x32"),
    )
    parser.add_argument(
        "--label-source-root",
        type=Path,
        default=Path("new_train_audio"),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def gather_labeled_files(data_root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    ignored_dirs = {"train", "test"}

    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir() and path.name not in ignored_dirs):
        for image_path in sorted(label_dir.rglob("*.png")):
            files.append((label_dir.name, image_path))

    return files


def ensure_clean_targets(data_root: Path) -> tuple[Path, Path, Path]:
    train_root = data_root / "train"
    test_root = data_root / "test"
    solution_path = data_root / "solution.csv"

    if train_root.exists() or test_root.exists() or solution_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split outputs under {data_root}. "
            "Remove train/, test/, and solution.csv first if you want to rerun."
        )

    train_root.mkdir(parents=True, exist_ok=False)
    test_root.mkdir(parents=True, exist_ok=False)
    return train_root, test_root, solution_path


def has_existing_split(data_root: Path) -> bool:
    train_root = data_root / "train"
    test_root = data_root / "test"
    return train_root.is_dir() and test_root.is_dir()


def load_existing_solution(solution_path: Path) -> dict[str, str]:
    if not solution_path.exists():
        return {}

    with solution_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "file_name" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError(f"{solution_path} must contain file_name and label columns")
        return {row["file_name"]: row["label"] for row in reader}


def build_label_lookup(label_source_root: Path) -> dict[str, str]:
    if not label_source_root.exists():
        raise FileNotFoundError(f"Label source root not found: {label_source_root}")

    file_to_label: dict[str, str] = {}
    duplicates: list[str] = []

    for label_dir in sorted(path for path in label_source_root.iterdir() if path.is_dir()):
        for audio_path in sorted(label_dir.rglob("*.ogg")):
            image_name = audio_path.with_suffix(".png").name
            if image_name in file_to_label and file_to_label[image_name] != label_dir.name:
                duplicates.append(image_name)
                continue
            file_to_label[image_name] = label_dir.name

    if duplicates:
        duplicate_preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"Found non-unique file basenames in {label_source_root}: {duplicate_preview}"
        )

    return file_to_label


def rebuild_solution_rows_from_existing_split(
    data_root: Path,
    label_source_root: Path,
    seed: int,
) -> list[dict[str, str]]:
    test_root = data_root / "test"
    solution_path = data_root / "solution.csv"
    existing_labels = load_existing_solution(solution_path)
    test_files = sorted(path.name for path in test_root.glob("*.png"))

    missing = [file_name for file_name in test_files if file_name not in existing_labels]
    if missing:
        lookup = build_label_lookup(label_source_root)
        unresolved = [file_name for file_name in missing if file_name not in lookup]
        if unresolved:
            raise ValueError(
                "Could not infer labels for some test files from the label source root. "
                f"Missing {len(unresolved)} files."
            )
        for file_name in missing:
            existing_labels[file_name] = lookup[file_name]

    rows = [{"file_name": file_name, "label": existing_labels[file_name]} for file_name in test_files]
    return assign_usage(rows, seed)


def move_files(
    labeled_files: list[tuple[str, Path]],
    train_root: Path,
    test_root: Path,
    test_fraction: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    shuffled = labeled_files[:]
    rng.shuffle(shuffled)

    test_count = int(round(len(shuffled) * test_fraction))
    test_files = shuffled[:test_count]
    train_files = shuffled[test_count:]

    solution_rows: list[dict[str, str]] = []

    for label, source_path in train_files:
        destination = train_root / label / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination))

    for label, source_path in test_files:
        destination = test_root / source_path.name
        shutil.move(str(source_path), str(destination))
        solution_rows.append({"file_name": destination.name, "label": label})

    solution_rows.sort(key=lambda row: row["file_name"])
    return solution_rows


def remove_empty_label_dirs(data_root: Path) -> None:
    for label_dir in sorted(path for path in data_root.iterdir() if path.is_dir() and path.name not in {"train", "test"}):
        if any(label_dir.iterdir()):
            continue
        label_dir.rmdir()


def write_solution_csv(solution_path: Path, rows: list[dict[str, str]]) -> None:
    with solution_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "Usage", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_submission_csv(submission_path: Path, rows: list[dict[str, str]], include_labels: bool) -> None:
    with submission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "label": row["label"] if include_labels else "",
                }
            )


def assign_usage(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed + 1)
    shuffled_rows = rows[:]
    rng.shuffle(shuffled_rows)

    public_count = len(shuffled_rows) // 2
    public_ids = {row["file_name"] for row in shuffled_rows[:public_count]}

    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        enriched_rows.append(
            {
                "file_name": row["file_name"],
                "Usage": "Public" if row["file_name"] in public_ids else "Private",
                "label": row["label"],
            }
        )
    return enriched_rows


def main() -> None:
    args = parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")

    solution_path = args.data_root / "solution.csv"
    sample_submission_path = args.data_root / "sample_submission.csv"
    benchmark_submission_path = args.data_root / "benchmark_submission.csv"

    if has_existing_split(args.data_root):
        kaggle_solution_rows = rebuild_solution_rows_from_existing_split(
            args.data_root,
            args.label_source_root,
            args.seed,
        )
        write_solution_csv(solution_path, kaggle_solution_rows)
        write_submission_csv(sample_submission_path, kaggle_solution_rows, include_labels=False)
        write_submission_csv(benchmark_submission_path, kaggle_solution_rows, include_labels=True)
        print(f"Rebuilt Kaggle solution file for existing split at {solution_path}")
        print(f"Wrote {sample_submission_path}")
        print(f"Wrote {benchmark_submission_path}")
        return

    labeled_files = gather_labeled_files(args.data_root)
    if not labeled_files:
        raise FileNotFoundError(f"No labeled .png files found under {args.data_root}")

    train_root, test_root, solution_path = ensure_clean_targets(args.data_root)
    solution_rows = move_files(
        labeled_files=labeled_files,
        train_root=train_root,
        test_root=test_root,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    remove_empty_label_dirs(args.data_root)
    kaggle_solution_rows = assign_usage(solution_rows, args.seed)
    write_solution_csv(solution_path, kaggle_solution_rows)
    write_submission_csv(sample_submission_path, kaggle_solution_rows, include_labels=False)
    write_submission_csv(benchmark_submission_path, kaggle_solution_rows, include_labels=True)

    print(f"Moved {len(labeled_files) - len(solution_rows)} files into {train_root}")
    print(f"Moved {len(solution_rows)} files into {test_root}")
    print(f"Wrote {solution_path}")
    print(f"Wrote {sample_submission_path}")
    print(f"Wrote {benchmark_submission_path}")


if __name__ == "__main__":
    main()
