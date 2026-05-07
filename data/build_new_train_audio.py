#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path

LABEL_OVERRIDES = {
    "Antbird / Antshrike": "Antbird",
    "Tanager / Finch": "Tanager",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("train_with_class.csv"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("train_audio"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("new_train_audio"),
    )
    return parser.parse_args()


def should_skip(label: str) -> bool:
    return not label or label.strip().lower() == "non-bird"


def normalize_class_name(label: str) -> str:
    label = label.strip()
    return LABEL_OVERRIDES.get(label, label)


def main() -> None:
    args = parse_args()

    copied = 0
    skipped_non_bird = 0
    missing = 0

    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            new_class_name = normalize_class_name(row.get("new_class_name") or "")
            relative_filename = (row.get("filename") or "").strip()

            if should_skip(new_class_name):
                skipped_non_bird += 1
                continue

            if not relative_filename:
                missing += 1
                continue

            source_path = args.source_root / relative_filename
            if not source_path.exists():
                missing += 1
                continue

            destination_dir = args.output_root / new_class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / source_path.name

            shutil.copy2(source_path, destination_path)
            copied += 1

    print(f"Copied bird files: {copied}")
    print(f"Skipped non-bird rows: {skipped_non_bird}")
    print(f"Skipped missing/invalid rows: {missing}")


if __name__ == "__main__":
    main()
