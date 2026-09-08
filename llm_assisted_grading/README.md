# Bonus Test Processing Pipeline

This repository contains a script-based workflow to:
- process scanned bonus tests,
- extract metadata from OMR regions,
- crop question images,
- create and optionally launch multiple iterations of OpenAI batch grading jobs,
- aggregate results across iterations,
- generate a manual-check PDF.

## Quick Start

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate bonus-test
```

### Option B: venv

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

## API Key

Set your OpenAI API key using one of these methods:
- Environment variable (recommended): `export OPENAI_API_KEY=...`
- `src/config.py`: `OPENAI_API_KEY = "..."` (avoid committing keys)

The repository includes `.env.example` as a template.

## Configuration

Edit `src/config.py` before running:
- `LANGUAGE`, `NAME`
- `IMAGE_PATH` (raw scanned submissions folder)
- `AGGREGATION_METHOD` (`"max"`, `"mean"`, `"median"`, `"min"`)
- Batch settings (`BATCH_MODEL`, `BATCH_ITERATIONS`, `BATCH_COMPLETION_WINDOW`)
- ROI and PDF layout settings if needed

## Run Order

Run these scripts in order:

1. Process scans, extract metadata, crop questions
```bash
python -m src.step_01_process_bonus_test
```

2. Create batch JSONL and optionally launch batches
```bash
python -m src.step_02_create_batches
```

3. Download completed batches and compute results
```bash
python -m src.step_03_evaluate_batches
```

4. Create manual-check PDF
```bash
python -m src.step_04_create_manual_check_pdf
```

## Step Details

### Step 1: `src.step_01_process_bonus_test`

Input:
- raw page images in `IMAGE_PATH` (default: `TEMPLATE_BONUS_TEST_ENG_RAW/`)

Key behavior:
- Attempts automatic corner detection and alignment
- Falls back to manual marker clicking when auto-detection fails
- Extracts `student_id`, `group`, and `version`
- Creates anonymized random keys and mapping file
- Crops and saves question images to the processed folder
- Writes OMR debug visuals and skip logs
- Creates a run manifest in `run_manifests/`

Main outputs:
- `TEMPLATE_BONUS_TEST_<LANG>_processed/*.jpg`
- `anonymized_student_ids_<NAME>_<LANG>_<timestamp>.txt`
- `skipped_images_<NAME>_<LANG>.txt` (if skips/warnings exist)
- `OMR_DEBUG/omr_debug_<LANG>/...`
- `run_manifests/<run_id>.json`

### Step 2: `src.step_02_create_batches`

Input:
- processed question crops from Step 1
- latest anonymized mapping file
- `BONUS_TEST_TEMPLATE_QUESTIONS.xlsx`

Key behavior:
- Builds a batch JSONL request file
- Prompts whether to launch batch jobs immediately
- If launched, creates `BATCH_ITERATIONS` batch jobs
- Writes batch IDs file and updates manifest

Main outputs:
- `TEMPLATE_BONUS_TEST_<LANG>_batches/TEMPLATE_BONUS_TEST_<LANG>_<timestamp>.jsonl`
- `TEMPLATE_BONUS_TEST_<LANG>_batches/batch_ids_<NAME>_<LANG>_<timestamp>.txt`
- `TEMPLATE_BONUS_TEST_<LANG>_batches/skipped_images_<NAME>_<LANG>_batch.txt` (if needed)

### Step 3: `src.step_03_evaluate_batches`

Input:
- batch IDs file from Step 2

Key behavior:
- Verifies all batches are completed before proceeding
- Downloads batch output JSONL files
- Parses score/flag/explanation per identifier
- Saves raw per-iteration results
- Aggregates results using `AGGREGATION_METHOD`
- Adds variability metrics and parsed identifier fields
- Updates run manifest

Main outputs:
- `TEMPLATE_BONUS_TEST_<LANG>_batches/TEMPLATE_BONUS_TEST_<LANG>_batch<i>.jsonl`
- `TEMPLATE_BONUS_TEST_<LANG>_batches/TEMPLATE_BONUS_TEST_<LANG>_results.csv`
- `TEMPLATE_BONUS_TEST_<LANG>_batches/TEMPLATE_BONUS_TEST_<LANG>_results_<aggregation>.csv`
- `TEMPLATE_BONUS_TEST_<LANG>_batches/batch_eval_<NAME>_<LANG>.txt`

### Step 4: `src.step_04_create_manual_check_pdf`

Input:
- aggregated CSV from Step 3
- raw results CSV from Step 3 (for flag review computation)
- processed images from Step 1
- latest anonymized mapping from manifest

Key behavior:
- Computes unsupervised review flags
- Merges metadata and explanations
- Renders one page per identifier with score and explanation
- Updates run manifest with PDF path

Main output:
- `TEMPLATE_BONUS_TEST_<LANG>_results/TEMPLATE_BONUS_TEST_<LANG>_results_explanation_flag_<aggregation>.pdf`

## Output Reference

Assuming:
- `NAME = TEMPLATE_BONUS_TEST`
- `LANGUAGE = ENG`
- `AGGREGATION_METHOD = max`

Expected key artifacts:
- Processed crops: `TEMPLATE_BONUS_TEST_ENG_processed/`
- Batch directory: `TEMPLATE_BONUS_TEST_ENG_batches/`
- Raw results CSV: `TEMPLATE_BONUS_TEST_ENG_batches/TEMPLATE_BONUS_TEST_ENG_results.csv`
- Aggregated results CSV: `TEMPLATE_BONUS_TEST_ENG_batches/TEMPLATE_BONUS_TEST_ENG_results_max.csv`
- Manual-check PDF: `TEMPLATE_BONUS_TEST_ENG_results/TEMPLATE_BONUS_TEST_ENG_results_explanation_flag_max.pdf`
- Manifests: `run_manifests/<run_id>.json`

## Repository Layout

Top-level items:
- `src/`: pipeline scripts and shared utilities
- `scripts/`: helper scripts (environment setup)
- `run_manifests/`: per-run JSON manifests linking produced artifacts
- `LaTex_Template/`: LaTeX source and PDF template
- `BONUS_TEST_TEMPLATE_QUESTIONS.xlsx`: question/solution/grading key source table
- `TEMPLATE_BONUS_TEST_<LANG>_RAW/`: raw scanned pages (input)
- `TEMPLATE_BONUS_TEST_<LANG>_processed/`: cropped question images (Step 1 output)
- `TEMPLATE_BONUS_TEST_<LANG>_batches/`: batch JSONL, batch outputs, CSV results, logs
- `TEMPLATE_BONUS_TEST_<LANG>_results/`: generated manual-check PDF
- `OMR_DEBUG/`: debug visualizations for OMR extraction
- `Test/`: bulk scan of all ten sample tests (save each page as image in respective folder to process)

## Run Manifests

Each run stores metadata in `run_manifests/<run_id>.json`.

The manifest links key artifacts across steps, including:
- mapping file,
- batch JSONL,
- batch IDs file,
- raw and aggregated result CSVs,
- generated PDF.

This is the main traceability mechanism across repeated runs.

## Notes

- Scripts may prompt before overwriting existing outputs.
- Step 3 stops if any batch is not completed yet.
- Reusing an old anonymized mapping preserves key continuity with previous runs.
- Keep `batch_ids_*.txt` private when possible; they are operational IDs (not credentials) but still internal metadata.
