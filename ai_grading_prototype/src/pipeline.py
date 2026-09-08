"""Minimal AI grading prototype pipeline.

This file is intentionally small. It shows the shape of the system without
locking you into a specific implementation.
"""

from pathlib import Path

from . import config


def ensure_dirs() -> None:
    for p in [
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.BATCH_DIR,
        config.RESULTS_DIR,
        config.MANIFEST_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    print("Prototype scaffold is ready.")
    print(f"Raw input: {config.RAW_DIR}")
    print(f"Questions: {config.QUESTIONS_XLSX}")
    print(f"Results: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()

