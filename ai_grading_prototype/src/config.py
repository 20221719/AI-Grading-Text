from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
BATCH_DIR = BASE_DIR / "data" / "batches"
RESULTS_DIR = BASE_DIR / "data" / "results"
MANIFEST_DIR = BASE_DIR / "data" / "manifests"

QUESTIONS_XLSX = BASE_DIR / "data" / "questions.xlsx"
OPENAI_MODEL = "gpt-5.1"
AGGREGATION_METHOD = "max"

