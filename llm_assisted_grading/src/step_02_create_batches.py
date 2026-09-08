"""Create grading batch JSONL and optionally launch batch jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

try:
    from . import config
    from . import utils
except ImportError:  # Allows running as a script: python src/02_create_batches.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    import config  # type: ignore
    import utils  # type: ignore

SKIP_LOG_PATH = config.BATCH_DIR / f"skipped_images_{config.NAME}_{config.LANGUAGE}_batch.txt"


def ensure_dirs() -> None:
    """Ensure batch output directory exists."""
    config.BATCH_DIR.mkdir(parents=True, exist_ok=True)


def resolve_latest_mapping_path() -> Path:
    """Resolve the latest anonymized IDs file path."""
    _manifest_path, manifest = utils.load_latest_manifest(config.RUN_MANIFEST_DIR)
    mapping_path = None
    if manifest and manifest.get("anonymized_ids_file"):
        mapping_path = Path(str(manifest.get("anonymized_ids_file")))
    if mapping_path and mapping_path.exists():
        return mapping_path

    existing_maps = sorted(
        Path(".").glob(f"{config.ANON_ID_PREFIX}_*.txt"),
        key=lambda p: p.stat().st_mtime,
    )
    if existing_maps:
        return existing_maps[-1]

    legacy_map = Path(config.ANON_ID_FILENAME)
    if legacy_map.exists():
        return legacy_map

    raise SystemExit("Aborted: no anonymized IDs file found. Run step 1 first.")


def create_batch_jsonl() -> Tuple[List[str], Path]:
    """Create the JSONL batch file and return skip log entries + JSONL path."""
    ensure_dirs()
    skip_log: List[str] = []

    mapping_path = resolve_latest_mapping_path()
    allowed_keys = set(utils.load_random_key_mapping(mapping_path).keys())
    if not allowed_keys:
        raise SystemExit(f"Aborted: anonymized IDs file is empty: {mapping_path}")

    timestamp = utils.extract_timestamp_from_mapping(mapping_path)
    batch_jsonl_path = (
        config.BATCH_DIR / f"{config.NAME}_{config.LANGUAGE}_{timestamp}.jsonl"
        if timestamp
        else config.BATCH_JSONL
    )

    if not utils.confirm_overwrite(batch_jsonl_path, "Batch JSONL"):
        raise SystemExit("Aborted: existing batch JSONL not overwritten.")

    df = utils.load_questions_table(config.QUESTIONS_XLSX)
    bonus_test_dictionary = utils.build_bonus_test_dictionary(df)

    with open(batch_jsonl_path, "w", encoding="utf-8") as handle:
        for img_path in sorted(config.OUT_PROCESSED_DIR.glob("*.jpg")):
            parsed = utils.parse_processed_filename(img_path.stem)
            if parsed is None:
                skip_log.append(f"Unexpected filename format: {img_path.name}")
                continue

            random_key, version, group, q_label = parsed
            if random_key not in allowed_keys:
                skip_log.append(
                    f"SKIP {img_path.name}: random key {random_key} not in {mapping_path.name}"
                )
                continue

            try:
                q_info = bonus_test_dictionary[group]
            except KeyError:
                skip_log.append(f"No entry for group={group} ({img_path.name})")
                continue

            version_key = config.QUESTION1_VERSION if q_label == "QUESTION1" else version
            try:
                entry = q_info[version_key]
            except KeyError:
                skip_log.append(
                    f"No entry for group={group} version={version_key} ({img_path.name})"
                )
                continue

            prompt_text = utils.build_grading_prompt(
                entry["Question"],
                entry["Solution"],
                entry["Grading Key"],
            )
            img_b64 = utils.encode_image_to_data_url(img_path)

            payload = {
                "custom_id": img_path.stem,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": config.BATCH_MODEL,
                    "reasoning": {"effort": config.BATCH_REASONING_EFFORT},
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text},
                                {"type": "input_image", "image_url": img_b64},
                            ],
                        }
                    ],
                },
            }
            utils.write_jsonl_line(handle, payload)

    utils.update_manifest_fields(
        config.RUN_MANIFEST_DIR, {"batch_jsonl": str(batch_jsonl_path)}
    )

    return skip_log, batch_jsonl_path


def write_skip_log(skip_log: List[str]) -> None:
    """Write skip log if present."""
    if not skip_log:
        return
    with open(SKIP_LOG_PATH, "w", encoding="utf-8") as handle:
        for line in skip_log:
            handle.write(f"{line}\n")


def prompt_launch(batch_jsonl_path: Path) -> bool:
    """Prompt the user to review the JSONL and decide whether to launch."""
    source, _ = utils.resolve_openai_api_key()
    print(f"Batch file written to: {batch_jsonl_path}")
    print("Please open and review the batch JSONL for completeness before launching.")
    if source == "config":
        print("API key source: config.OPENAI_API_KEY")
    elif source == "environment":
        print("API key source: OPENAI_API_KEY environment variable")
    else:
        print("API key source: missing")
    choice = input("Launch batch jobs now? (y/n): ").strip().lower()
    return choice == "y"


def launch_batches(batch_jsonl_path: Path) -> None:
    """Upload JSONL and launch batch jobs."""
    source, api_key = utils.resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or config.OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)
    batch_ids: List[str] = []

    for i in range(config.BATCH_ITERATIONS):
        batch_input_file = client.files.create(
            file=open(batch_jsonl_path, "rb"),
            purpose="batch",
        )

        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window=config.BATCH_COMPLETION_WINDOW,
            metadata={"description": f"grading student answers iteration {i}"},
        )

        batch_ids.append(batch_job.id)
        print(f"Batch job created for iteration {i}: {batch_job.id}")

    manifest_path, manifest = utils.load_latest_manifest(config.RUN_MANIFEST_DIR)
    mapping_path = None
    if manifest and manifest.get("anonymized_ids_file"):
        mapping_path = Path(str(manifest.get("anonymized_ids_file")))
    timestamp = utils.extract_timestamp_from_mapping(mapping_path) if mapping_path else None
    batch_ids_path = (
        config.BATCH_DIR / f"{config.BATCH_IDS_PREFIX}_{timestamp}.txt"
        if timestamp
        else config.BATCH_IDS_FILENAME
    )

    if not utils.confirm_overwrite(batch_ids_path, "Batch IDs file"):
        raise SystemExit("Aborted: existing batch IDs file not overwritten.")

    with open(batch_ids_path, "w", encoding="utf-8") as handle:
        for batch_id in batch_ids:
            handle.write(batch_id + "\n")

    utils.update_manifest_fields(
        config.RUN_MANIFEST_DIR, {"batch_ids_file": str(batch_ids_path)}
    )


if __name__ == "__main__":
    skipped, batch_jsonl_path = create_batch_jsonl()
    write_skip_log(skipped)
    if prompt_launch(batch_jsonl_path):
        launch_batches(batch_jsonl_path)
