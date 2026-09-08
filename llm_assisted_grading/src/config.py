"""Configuration for the bonus test processing pipeline."""

from pathlib import Path

# Input/output naming
LANGUAGE = "ENG"
# Set API key here if you do not want to use the OPENAI_API_KEY environment variable.
OPENAI_API_KEY = None
NAME = "TEMPLATE_BONUS_TEST"
# Folder containing the raw scanned submissions (input images).
IMAGE_PATH = Path(f"{NAME}_{LANGUAGE}_RAW")
FILE_END = f"{NAME}_{LANGUAGE}"
OUT_PROCESSED_DIR = Path(f"{NAME}_{LANGUAGE}_processed")
ANON_ID_PREFIX = f"anonymized_student_ids_{NAME}_{LANGUAGE}"
# Legacy default (current pipeline uses timestamped filenames per run).
ANON_ID_FILENAME = f"{ANON_ID_PREFIX}.txt"
BATCH_DIR = Path(f"{NAME}_{LANGUAGE}_batches")
BATCH_JSONL = BATCH_DIR / f"{NAME}_{LANGUAGE}.jsonl"
BATCH_IDS_PREFIX = f"batch_ids_{NAME}_{LANGUAGE}"
BATCH_IDS_FILENAME = BATCH_DIR / f"{BATCH_IDS_PREFIX}.txt"
BATCH_RESULTS_CSV = BATCH_DIR / f"{NAME}_{LANGUAGE}_results.csv"
# Aggregation method for final score per identifier: "max", "mean", "median", "min".
AGGREGATION_METHOD = "max"
BATCH_RESULTS_AGG_CSV = BATCH_DIR / f"{NAME}_{LANGUAGE}_results_{AGGREGATION_METHOD}.csv"
QUESTIONS_XLSX = Path("BONUS_TEST_TEMPLATE_QUESTIONS.xlsx")
RESULTS_DIR = Path(f"{NAME}_{LANGUAGE}_results")
RESULTS_PDF_EXPLANATION = RESULTS_DIR / (
    f"{NAME}_{LANGUAGE}_results_explanation_flag_{AGGREGATION_METHOD}.pdf"
)
RUN_MANIFEST_DIR = Path("run_manifests")

# Batch API settings
BATCH_MODEL = "gpt-5.1"
BATCH_REASONING_EFFORT = "high"
BATCH_COMPLETION_WINDOW = "24h"
BATCH_ITERATIONS = 5
QUESTION1_VERSION = "N"
GRADING_GUIDELINES_LIST = [
    "Real Numbers Only",
    "Steps",
    "No hallucinations",
    "Nonsensical segments",
]

# Debug output folders
OMR_DEBUG_DIR = Path(f"OMR_DEBUG/omr_debug_{LANGUAGE}")
OMR_DEBUG_STUDENT_ID_DIR = OMR_DEBUG_DIR / "student_id"
OMR_DEBUG_GROUP_DIR = OMR_DEBUG_DIR / "group"
OMR_DEBUG_VERSION_DIR = OMR_DEBUG_DIR / "version"

# Page size (A4 @ 300 dpi)
A4_W, A4_H = 2480, 3508

# ROI definitions (y1, y2, x1, x2)
ROIS = {
    "student_id": (118, 865, 108, 822),
    "group": (350, 450, 1050, 1800),
    "version": (692, 767, 1026, 1377),
    "question1": (900, 1800, 50, 2425),
    "question2": (1850, 3475, 50, 2425),
}

# Corner marker detection parameters
CORNER_FRACS = [0.05, 0.07, 0.10]
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.15
ASPECT_MIN, ASPECT_MAX = 0.75, 1.30
EXTENT_MIN = 0.55
SOLIDITY_MIN = 0.90

# OCR grid parameters
ID_ROWS = 10
ID_COLS = 7
CELL_MARGIN = 0.18
CIRCLE_SCALE = 0.5
FILL_THRESHOLD = 0.20
BINARIZE_OFFSET = 50

# Pick thresholds
ID_OFFSET_FROM_MEDIAN = 0.05
ID_MIN_MARGIN = 0.10
GROUP_OFFSET_FROM_MEDIAN = 0.05
GROUP_MIN_MARGIN = 0.15
VERSION_OFFSET_FROM_MEDIAN = 0.05
VERSION_MIN_MARGIN = 0.10

# Labels
GROUP_LABELS = ["1", "2", "3", "4", "5"]
VERSION_LABELS = ["A", "B"]

# Random anonymization
RANDOM_SEED = 42
RANDOM_KEY_COUNT = 1000
RANDOM_KEY_LENGTH = 6

# Manual check PDF layout (A4 mm)
PDF_PAGE_W = 210
PDF_PAGE_H = 297
PDF_MARGIN = 10
PDF_HEADER_BLOCK_H = 30
PDF_TITLE_FONT_SIZE = 18
PDF_DETAIL_FONT_SIZE = 13
PDF_SCORE_FONT_INC = 10
PDF_STD_FONT_INC = 5
PDF_V_GAP_BETWEEN_IMAGES = 8
PDF_GAP_INFO_TO_IMG = 10
PDF_INFO_BLOCK_H = 14
PDF_GAP_IMG_TO_EXPL = 2
PDF_EXPL_RESERVED_H = 40
PDF_CENTER_BLOCK = True

# Manual review flagging parameters (copied from config_ENG.py)
scaler = {
    "gpt_std": {"med": 0.0, "scale": 0.7760040183930035},
    "gpt_iqr": {"med": 0.0, "scale": 0.9198182819707031},
    "gpt_range": {"med": 0.0, "scale": 1.9442577579321474},
    "pairwise_madiff": {"med": 0.0, "scale": 0.8401469912038706},
    "bimodality": {"med": 0.0, "scale": 0.5559673832468496},
    "hist_entropy": {"med": 0.0, "scale": 0.2931716005214807},
    "loo_instability": {"med": 0.0, "scale": 0.25827297325740733},
}
global_thr = 1.3383576623662983

grading_prompt_dictionary = {
    "Intro": (
        "You are grading a student's solution. Make sure the question they are "
        "attempting to solve matches the question provided below. If it does not "
        "match, give them a score of 0."
    ),
    "Steps": (
        "Award credit for intermediate steps only if they are explicitly written by the "
        "student; do not infer non-trivial reasoning from a correct final answer except "
        "for trivial algebraic simplifications."
    ),
    "Real Numbers Only": (
        "Assume the course works over the real numbers R; do not penalize students for "
        "omitting complex solutions or incomplete complex solutions."
    ),
    "No hallucinations": (
        "Base all scoring strictly on evidence in the student's solution; if evidence is "
        "missing, assign zero for that criterion--do not guess or hallucinate it."
    ),
    "Nonsensical segments": (
        "Be aware of nonsensical segments; apply partial credit only to valid, supported "
        "steps, and do not award credit for reasoning that would not imply the subsequent "
        "result."
    ),
}
