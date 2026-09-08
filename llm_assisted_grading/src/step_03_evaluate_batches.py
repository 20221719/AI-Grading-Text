"""Download completed batch results and compute grading summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from openai import OpenAI

try:
    from . import config
    from . import utils
except ImportError:  # Allows running as a script: python src/03_evaluate_batches.py
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent))
    import config  # type: ignore
    import utils  # type: ignore

EVAL_LOG_PATH = config.BATCH_DIR / f"batch_eval_{config.NAME}_{config.LANGUAGE}.txt"


def ensure_dirs() -> None:
    """Ensure batch output directory exists."""
    config.BATCH_DIR.mkdir(parents=True, exist_ok=True)


def load_batch_ids() -> List[str]:
    """Load batch IDs from disk."""
    manifest_path, manifest = utils.load_latest_manifest(config.RUN_MANIFEST_DIR)
    batch_ids_path = None
    if manifest and manifest.get("batch_ids_file"):
        batch_ids_path = Path(str(manifest.get("batch_ids_file")))
    if batch_ids_path is None:
        # Fallback to latest timestamped batch_ids file if manifest is missing
        candidates = sorted(
            config.BATCH_DIR.glob(f"{config.BATCH_IDS_PREFIX}_*.txt"),
            key=lambda p: p.stat().st_mtime,
        )
        batch_ids_path = candidates[-1] if candidates else config.BATCH_IDS_FILENAME
    if not batch_ids_path.exists():
        raise FileNotFoundError(f"Missing batch ids file: {batch_ids_path}")
    with open(batch_ids_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def download_completed_batches(batch_ids: List[str], log: List[str]) -> None:
    """Download completed batch outputs into the batch directory."""
    _source, api_key = utils.resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or config.OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)

    for i, batch_id in enumerate(batch_ids):
        job = client.batches.retrieve(batch_id)
        if job.status != "completed":
            log.append(f"Batch {i} ({batch_id}) status: {job.status}")
            continue

        output_file_id = job.output_file_id
        if not output_file_id:
            log.append(f"Batch {i} ({batch_id}) completed but has no output file.")
            continue

        result_content = client.files.content(output_file_id)
        out_path = config.BATCH_DIR / f"{config.NAME}_{config.LANGUAGE}_batch{i}.jsonl"
        with open(out_path, "wb") as handle:
            handle.write(result_content.read())
        log.append(f"Downloaded batch {i} to {out_path}")


def extract_output_text(data: Dict) -> str:
    """Extract the assistant output text from a batch result entry."""
    try:
        return utils.first_text_from_output_list(data["response"]["body"]["output"])
    except (KeyError, TypeError):
        try:
            return utils.first_text_from_output_list(data["response"]["output"])
        except (KeyError, TypeError):
            return ""


def parse_batch_files(batch_count: int, log: List[str]) -> pd.DataFrame:
    """Parse batch output files into a DataFrame of records."""
    records = []
    for i in range(batch_count):
        batch_file = config.BATCH_DIR / f"{config.NAME}_{config.LANGUAGE}_batch{i}.jsonl"
        if not batch_file.exists():
            log.append(f"Missing batch output file: {batch_file}")
            continue

        with open(batch_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                identifier = data.get("custom_id", "")
                output_text = extract_output_text(data)
                score, flag = utils.extract_score_flag(output_text)

                records.append(
                    {
                        "iteration": i,
                        "identifier": identifier,
                        "score": score,
                        "flag": flag,
                        "explanation": output_text.strip(),
                    }
                )

    return pd.DataFrame(records)


def aggregate_results(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Aggregate results per identifier using the configured method."""
    if method == "max":
        idx = df.groupby("identifier")["score"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    if method == "min":
        idx = df.groupby("identifier")["score"].idxmin()
        return df.loc[idx].reset_index(drop=True)
    if method == "mean":
        return (
            df.groupby("identifier", as_index=False)
            .agg(
                score=("score", "mean"),
                flag=("flag", "first"),
                explanation=("explanation", "first"),
                iteration=("iteration", "first"),
            )
        )
    if method == "median":
        return (
            df.groupby("identifier", as_index=False)
            .agg(
                score=("score", "median"),
                flag=("flag", "first"),
                explanation=("explanation", "first"),
                iteration=("iteration", "first"),
            )
        )
    raise ValueError(f"Unknown aggregation method: {method}")


def postprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated results and variability metrics."""
    if df.empty:
        return df

    std_dev = df.groupby("identifier")["score"].std().reset_index()
    std_dev.columns = ["identifier", "score_std"]

    mode_flag = (
        df.groupby("identifier")["flag"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
    )
    mode_flag.columns = ["identifier", "flag_mode"]

    df = df.dropna(subset=["score"]).reset_index(drop=True)

    df = aggregate_results(df, config.AGGREGATION_METHOD)

    df = df.drop(columns=["flag"])
    merged = df.merge(std_dev, on="identifier").merge(mode_flag, on="identifier")

    def categorize_std(x):
        if pd.isna(x):
            return "low"
        if x <= 1:
            return "low"
        if x <= 2:
            return "medium"
        return "high"

    merged["score_std_cat"] = merged["score_std"].apply(categorize_std)
    merged = merged.rename(columns={"flag_mode": "flag"})

    parsed_rows = merged["identifier"].apply(utils.parse_identifier)
    parsed_df = pd.json_normalize(parsed_rows)
    for col in ("random_key", "version", "group", "question"):
        if col not in parsed_df.columns:
            parsed_df[col] = None
    merged = pd.concat([merged, parsed_df], axis=1)

    merged = merged[
        [
            "identifier",
            "random_key",
            "version",
            "group",
            "question",
            "score",
            "score_std",
            "score_std_cat",
            "flag",
            "explanation",
        ]
    ]

    merged["group"] = merged["group"].astype(str)
    merged["group"] = pd.Categorical(merged["group"], categories=["1", "2", "3", "4", "5"], ordered=True)
    merged["question"] = pd.Categorical(merged["question"], categories=[1, 2], ordered=True)
    merged["version"] = pd.Categorical(merged["version"], categories=["A", "B"], ordered=True)

    merged = merged.sort_values(by=["question", "group", "version"])
    mask = merged["question"] == 2
    df_question2 = merged[mask].sort_values(by=["version", "group"])
    df_question1 = merged[~mask]
    merged = pd.concat([df_question1, df_question2], ignore_index=True)

    return merged.reset_index(drop=True)


def write_eval_log(log: List[str]) -> None:
    """Write evaluation log messages to disk."""
    if not log:
        return
    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as handle:
        for line in log:
            handle.write(f"{line}\n")


def process() -> None:
    """Download batch results and compute summary CSVs."""
    ensure_dirs()
    log: List[str] = []

    batch_ids = load_batch_ids()
    source, api_key = utils.resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or config.OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)
    statuses = [client.batches.retrieve(bid).status for bid in batch_ids]
    if any(status != "completed" for status in statuses):
        print(
            "Not all batches have completed. Please check the OpenAI dashboard at "
            "https://platform.openai.com for batch status."
        )
        return

    existing_outputs = list(
        config.BATCH_DIR.glob(f"{config.NAME}_{config.LANGUAGE}_batch*.jsonl")
    )
    if existing_outputs:
        choice = input(
            f"Found {len(existing_outputs)} existing batch output files. "
            "Delete them before downloading new results? (y/n): "
        ).strip().lower()
        if choice == "y":
            for path in existing_outputs:
                path.unlink()
        else:
            raise SystemExit("Aborted: existing batch output files not cleared.")

    download_completed_batches(batch_ids, log)

    df = parse_batch_files(len(batch_ids), log)
    if not df.empty:
        if not utils.confirm_overwrite(config.BATCH_RESULTS_CSV, "Batch results CSV"):
            raise SystemExit("Aborted: existing results CSV not overwritten.")
        df.sort_values(by="identifier").reset_index(drop=True).to_csv(
            config.BATCH_RESULTS_CSV, index=False
        )
        agg_df = postprocess_results(df)
        if not utils.confirm_overwrite(
            config.BATCH_RESULTS_AGG_CSV, "Aggregated results CSV"
        ):
            raise SystemExit("Aborted: existing aggregated CSV not overwritten.")
        agg_df.to_csv(config.BATCH_RESULTS_AGG_CSV, index=False)
        utils.update_manifest_fields(
            config.RUN_MANIFEST_DIR,
            {
                "results_csv": str(config.BATCH_RESULTS_CSV),
                "aggregated_results_csv": str(config.BATCH_RESULTS_AGG_CSV),
            },
        )
    else:
        log.append("No records found in batch outputs.")

    write_eval_log(log)


if __name__ == "__main__":
    process()
