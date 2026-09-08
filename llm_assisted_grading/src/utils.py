"""Utility functions for OMR processing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import base64
import json
import re
from datetime import datetime, timezone
from itertools import combinations

import cv2
import numpy as np
import os
import shutil

from PIL import Image

from . import config


def binarize(img_gray: np.ndarray, offset: int = config.BINARIZE_OFFSET) -> np.ndarray:
    """Binarize a grayscale image using Otsu + offset and return inverted binary."""
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(threshold + offset, 0)
    _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return binary


def get_corner_rois(shape: Tuple[int, int], frac: float) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Return corner ROI slices for a given fraction of the image."""
    height, width = shape
    frac_h, frac_w = int(height * frac), int(width * frac)
    return {
        "top_left": ((0, frac_h), (0, frac_w), (0, 0)),
        "top_right": ((0, frac_h), (width - frac_w, width), (width - 1, 0)),
        "bottom_left": ((height - frac_h, height), (0, frac_w), (0, height - 1)),
        "bottom_right": ((height - frac_h, height), (width - frac_w, width), (width - 1, height - 1)),
    }


def pick_marker_from_contours(
    roi_bin: np.ndarray,
    corner_name: str,
    roi_origin: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    min_area_ratio: float = config.MIN_AREA_RATIO,
    max_area_ratio: float = config.MAX_AREA_RATIO,
    aspect_min: float = config.ASPECT_MIN,
    aspect_max: float = config.ASPECT_MAX,
    extent_min: float = config.EXTENT_MIN,
    solidity_min: float = config.SOLIDITY_MIN,
    debug_canvas: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], Optional[int], Optional[np.ndarray]]:
    """
    Select the best contour in a corner ROI based on geometric constraints.

    Returns (cx_global, cy_global, contour) or (None, None, None).
    """
    contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    height, width = roi_bin.shape
    roi_area = height * width

    if corner_name == "top_left":
        corner_pt_roi = np.array([0, 0])
    elif corner_name == "top_right":
        corner_pt_roi = np.array([width - 1, 0])
    elif corner_name == "bottom_left":
        corner_pt_roi = np.array([0, height - 1])
    else:
        corner_pt_roi = np.array([width - 1, height - 1])

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        area_ratio = area / roi_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect = w_box / float(h_box) if h_box > 0 else 999
        if not (aspect_min <= aspect <= aspect_max):
            continue

        extent = area / float(w_box * h_box)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if extent < extent_min or solidity < solidity_min:
            continue

        cx = x + w_box // 2
        cy = y + h_box // 2
        distance = np.linalg.norm(np.array([cx, cy]) - corner_pt_roi)
        candidates.append((distance, (cx, cy), contour))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda item: item[0])
    _, (cx, cy), best_contour = candidates[0]

    (y0, _), (x0, _), _ = roi_origin
    cx_global = x0 + cx
    cy_global = y0 + cy

    if debug_canvas is not None:
        cv2.drawContours(debug_canvas, [best_contour], -1, 255, 2)
        cv2.circle(debug_canvas, (cx, cy), 8, 255, -1)

    return cx_global, cy_global, best_contour


def detect_four_markers(
    gray: np.ndarray,
    corner_fracs: Iterable[float] = config.CORNER_FRACS,
) -> Tuple[Optional[Dict[str, Tuple[int, int]]], np.ndarray, Optional[Dict[str, Tuple]], Optional[Dict[str, np.ndarray]], Optional[float]]:
    """Detect all four corner markers and return their positions."""
    bin_full = binarize(gray)
    height, width = bin_full.shape

    for frac in corner_fracs:
        rois = get_corner_rois((height, width), frac)
        found = {}
        debug_rois = {}

        for name, origin in rois.items():
            (y0, y1), (x0, x1), _ = origin
            roi_bin = bin_full[y0:y1, x0:x1].copy()
            debug_roi = roi_bin.copy()
            cx_g, cy_g, _ = pick_marker_from_contours(roi_bin, name, origin, debug_canvas=debug_roi)
            debug_rois[name] = debug_roi

            if cx_g is not None:
                found[name] = (cx_g, cy_g)

        if len(found) == 4:
            return found, bin_full, rois, debug_rois, frac

    return None, bin_full, None, None, None


def align_image(image: np.ndarray) -> Optional[np.ndarray]:
    """Align a scanned image to A4 coordinates based on the four corner markers."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, _, _, _, _ = detect_four_markers(gray)
    if found is None:
        return None

    src_pts = np.array(
        [
            found["top_left"],
            found["top_right"],
            found["bottom_left"],
            found["bottom_right"],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [
            [0, 0],
            [config.A4_W - 1, 0],
            [0, config.A4_H - 1],
            [config.A4_W - 1, config.A4_H - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (config.A4_W, config.A4_H))


def grid_fill_scores(
    bin_img: np.ndarray,
    rows: int = config.ID_ROWS,
    cols: int = config.ID_COLS,
    margin: float = config.CELL_MARGIN,
    circle_scale: float = config.CIRCLE_SCALE,
) -> np.ndarray:
    """Compute fill scores per grid cell using a circular sampling region."""
    height, width = bin_img.shape
    cell_h, cell_w = height / rows, width / cols
    scores = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * cell_h + margin * cell_h)
            y2 = int((r + 1) * cell_h - margin * cell_h)
            x1 = int(c * cell_w + margin * cell_w)
            x2 = int((c + 1) * cell_w - margin * cell_w)

            if y2 <= y1 or x2 <= x1:
                continue

            cell = bin_img[y1:y2, x1:x2]
            h, w = cell.shape

            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
            r_max = 0.5 * min(h, w)
            radius = circle_scale * r_max
            if radius <= 0:
                continue

            ys, xs = np.ogrid[:h, :w]
            dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
            mask = dist2 < (radius ** 2)
            if not np.any(mask):
                continue

            ink = cell[mask]
            scores[r, c] = ink.mean() / 255.0

    return scores


def pick_per_column(scores: np.ndarray, offset_from_median: float, min_margin: float) -> List[Optional[int]]:
    """Pick the best row per column, returning None for ambiguous columns."""
    rows, cols = scores.shape
    picks: List[Optional[int]] = []
    threshold = np.median(scores) + offset_from_median
    for c in range(cols):
        col = scores[:, c]
        sorted_idx = np.argsort(col)[::-1]
        best_r, second_r = sorted_idx[0], sorted_idx[1]
        best_val, second_val = col[best_r], col[second_r]

        if best_val >= threshold and (best_val - second_val) >= min_margin:
            picks.append(best_r)
        else:
            picks.append(None)
    return picks


def pick_one_from_1xN(scores: np.ndarray, offset_from_median: float, min_margin: float) -> Optional[int]:
    """Pick the best column for 1xN grids, returning None for ambiguous picks."""
    threshold = np.median(scores) + offset_from_median
    if scores.shape[0] != 1:
        raise ValueError("Expected scores with shape (1, N).")

    row = scores[0]
    sorted_idx = np.argsort(row)[::-1]
    best_c, second_c = sorted_idx[0], sorted_idx[1]
    best_val, second_val = row[best_c], row[second_c]

    if best_val >= threshold and (best_val - second_val) >= min_margin:
        return int(best_c)
    return None


def debug_grid(
    gray_roi: np.ndarray,
    scores: np.ndarray,
    picks: Optional[Iterable[int]] = None,
    title: Optional[str] = None,
    path: Optional[Path] = None,
    margin: float = config.CELL_MARGIN,
    circle_scale: float = config.CIRCLE_SCALE,
    draw_circles: bool = True,
) -> None:
    """Render a debug plot for a grid ROI with scores and selection markers."""
    import matplotlib.pyplot as plt

    height, width = gray_roi.shape
    rows, cols = scores.shape
    cell_h, cell_w = height / rows, width / cols

    fig, ax = plt.subplots(figsize=(max(4, cols * 0.6), max(4, rows * 0.6)))
    ax.imshow(gray_roi, cmap="gray")

    for c in range(cols + 1):
        x = c * cell_w
        ax.plot([x, x], [0, height], linewidth=1)
    for r in range(rows + 1):
        y = r * cell_h
        ax.plot([0, width], [y, y], linewidth=1)

    if draw_circles:
        inner_h = cell_h * (1 - 2 * margin)
        inner_w = cell_w * (1 - 2 * margin)
        radius = circle_scale * 0.5 * min(inner_h, inner_w)
        if radius > 0:
            for r in range(rows):
                for c in range(cols):
                    cy = (r + 0.5) * cell_h
                    cx = (c + 0.5) * cell_w
                    circ = plt.Circle((cx, cy), radius, fill=False, linewidth=0.8)
                    ax.add_patch(circ)

    for r in range(rows):
        for c in range(cols):
            ax.text(
                c * cell_w + 4,
                r * cell_h + 14,
                f"{scores[r, c]:.2f}",
                fontsize=8,
            )

    if picks is not None:
        if np.isscalar(picks):
            if picks is not None and 0 <= int(picks) < cols:
                c = int(picks)
                ax.plot((c + 0.5) * cell_w, 0.5 * cell_h, "o", markersize=8)
        else:
            for c, r in enumerate(picks):
                if r is not None:
                    ax.plot((c + 0.5) * cell_w, (int(r) + 0.5) * cell_h, "o", markersize=8)

    if title:
        ax.set_title(title)
    ax.set_axis_off()

    if path:
        fig.savefig(path, bbox_inches="tight", dpi=200)

    plt.close(fig)


def load_images(image_dir: Path) -> List[Tuple[str, np.ndarray]]:
    """Load, rotate, and return images from a directory, sorted by filename."""
    images: List[Tuple[str, np.ndarray]] = []
    for img_path in image_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}, skipping.")
            continue
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        images.append((img_path.name, img))

    images.sort(key=lambda item: item[0])
    return images


def load_questions_table(path: Path):
    """Load the questions table from Excel."""
    import pandas as pd

    return pd.read_excel(path)


def build_bonus_test_dictionary(df) -> Dict[int, Dict[str, Dict[str, str]]]:
    """Build nested dictionary by group and version."""
    bonus_test_dictionary: Dict[int, Dict[str, Dict[str, str]]] = {}
    for group, group_data in df.groupby("Group"):
        bonus_test_dictionary[group] = {}
        for version, version_data in group_data.groupby("Version"):
            bonus_test_dictionary[group][version] = {
                "Question": version_data["Question"].values[0],
                "Solution": version_data["Solution"].values[0],
                "Grading Key": version_data["Grading Key"].values[0],
            }
    return bonus_test_dictionary


def parse_processed_filename(stem: str) -> Optional[Tuple[str, str, int, str]]:
    """Parse processed image stem into (random_key, version, group, q_label)."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    random_key, version, group_raw, q_label = parts[0], parts[1], parts[2], parts[3]
    try:
        group = int(group_raw)
    except ValueError:
        return None
    return random_key, version, group, q_label


def parse_student_and_question(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract random key and question number from an image stem."""
    random_key = stem.split("_", 1)[0]
    match = re.search(r"QUESTION(?:_|-|\\s)*([12])", stem, re.IGNORECASE)
    qnum = int(match.group(1)) if match else None
    return random_key, qnum


def build_grading_prompt(question_text: str, solution_text: str, grading_key: str) -> str:
    """Build the grading prompt text for the batch request."""
    guidelines = " ".join(
        config.grading_prompt_dictionary[key] for key in config.GRADING_GUIDELINES_LIST
    )
    prompt_text = f"""
{config.grading_prompt_dictionary["Intro"]}

Grading guidelines:
{guidelines}

Question:
{question_text}

{solution_text}

Grading key:
{grading_key}
"""
    return prompt_text.strip()


def encode_image_to_data_url(img_path: Path) -> str:
    """Encode an image file as a data URL for the batch payload."""
    with open(img_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def write_jsonl_line(handle, payload: dict) -> None:
    """Write one JSONL line to a file handle."""
    handle.write(json.dumps(payload) + "\n")


def first_text_from_output_list(output_list) -> str:
    """Return the first text content from an output list, if any."""
    if not isinstance(output_list, list):
        return ""
    for item in output_list:
        try:
            return item["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue
    return ""


def extract_score_flag(output_text: str) -> Tuple[Optional[float], Optional[int]]:
    """Extract score and flag values from model output text."""
    score = None
    flag = None

    score_patterns = [
        r"Total[^0-9]*([0-9]+(?:\\.[0-9]+)?)\\s*/\\s*10",
        r"Score[^0-9]*([0-9]+(?:\\.[0-9]+)?)\\s*/\\s*10",
        r"Total[^0-9]*([0-9]+(?:\\.[0-9]+)?)",
    ]
    for pattern in score_patterns:
        m_score = re.search(pattern, output_text, flags=re.IGNORECASE)
        if m_score:
            score = float(m_score.group(1))
            break

    m_flag = re.search(r"Flag[^0-9]*([0-9]+)", output_text, flags=re.IGNORECASE)
    if m_flag:
        flag = int(m_flag.group(1))
    return score, flag


def parse_identifier(identifier: str) -> Dict[str, Optional[object]]:
    """Parse identifier into components for reporting."""
    parts = identifier.split("_")
    if len(parts) < 4:
        return {"random_key": None, "version": None, "group": None, "question": None}
    random_key = parts[0]
    version = parts[1]
    group_raw = parts[2]
    question_raw = parts[3]
    try:
        group = int(group_raw)
    except ValueError:
        group = None
    question_digit = question_raw[-1] if question_raw else ""
    try:
        question = int(question_digit)
    except ValueError:
        question = None
    return {"random_key": random_key, "version": version, "group": group, "question": question}


def load_random_key_mapping(path: Path) -> Dict[str, str]:
    """Load mapping from random key to student id."""
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            left, right = line.strip().split(":", 1)
            random_key = left.strip()
            student_id = right.strip()
            if random_key:
                mapping[random_key] = student_id
    return mapping


def load_student_id_mapping(path: Path) -> Dict[str, str]:
    """Load mapping from student id to random key."""
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            left, right = line.strip().split(":", 1)
            random_key = left.strip()
            student_id = right.strip()
            if student_id:
                mapping[student_id] = random_key
    return mapping


def natural_size_fit_width_mm(img_path: Path, target_w_mm: float) -> Tuple[float, float]:
    """Return width/height in mm preserving aspect ratio at target width."""
    with Image.open(img_path) as img:
        w_px, h_px = img.size
    aspect = w_px / h_px
    w_mm = target_w_mm
    h_mm = w_mm / aspect
    return w_mm, h_mm


def estimate_line_count(pdf, text: str, width: float) -> int:
    """Estimate how many wrapped lines the text will take for a given width."""
    lines = 0
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines += 1
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if pdf.get_string_width(test) <= width:
                current = test
            else:
                lines += 1
                current = word
        if current:
            lines += 1
    return max(1, lines)


def latin1_safe(s: str) -> str:
    """Normalize text into latin-1 safe representation with ASCII fallbacks."""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u00A0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2026": "...",
        "\u2022": "*",
        "\u221A": "sqrt",
        "\u2260": "!=",
        "\u00B1": "+/-",
        "\u00B7": "*",
        "\u2192": "->",
        "\u21D2": "=>",
        "\u2208": "in",
        "\u2228": "OR",
        "\u2248": "~",
        "\u211D": "R",
        "\u0394": "Delta",
        "\u03B1": "alpha",
        "\u00F7": "/",
        "\u2200": "forall",
        "\u2261": "===",
        "\u2265": ">=",
        "\u221E": "inf",
        "\u2264": "<=",
        "\u222A": "union",
        "\u2205": "emptyset",
        "\u21D4": "<=>",
        "\u2229": "intersect",
        "\u00B2": "^2",
        "\u00B0": "deg",
        "\u00D7": "*",
        "\u03B8": "theta",
        "\u03C0": "pi",
        "\u2032": "'",
        "\u2124": "Z",
        "\u22C5": "*",
        "\u00B9": "1",
        "\u207B": "^-",
        "\u00B3": "^3",
        "\u27F6": "->",
        "\u2713": " check mark ",
    }
    s = s.translate(str.maketrans(replacements))
    return s.encode("latin-1", "replace").decode("latin-1")


def clean_explanation(text: Optional[str]) -> Optional[str]:
    """Normalize explanation strings."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n\n", "\n")
    text = re.sub(r"\\\(", "", text)
    text = re.sub(r"\\\)", "", text)
    text = re.sub(r"\\\[", "", text)
    text = re.sub(r"\\\]", "", text)
    text = text.replace("\n\n", "\n")
    return text.strip() or None


def draw_student_header(
    pdf,
    random_key: str,
    student_id_to_random_key: Dict[str, str],
    margin: float,
    page_w: float,
    title_font_size: int,
    header_block_h: float,
    return_compact_y: bool = True,
    gap_after: float = 2.0,
) -> float:
    """Draw the header and return the bottom y position."""
    display_id = student_id_to_random_key.get(random_key, random_key)
    pdf.set_xy(margin, margin)
    pdf.set_font("Arial", "B", title_font_size)
    pdf.set_text_color(0, 0, 0)
    if return_compact_y:
        pdf.cell(
            w=page_w - 2 * margin,
            h=7,
            txt=latin1_safe(f"Student ID: {display_id}"),
            ln=1,
            align="L",
        )
        return pdf.get_y() + gap_after
    pdf.multi_cell(
        w=page_w - 2 * margin,
        h=8,
        txt=latin1_safe(f"Student ID: {display_id}"),
        align="L",
    )
    return margin + header_block_h


def draw_info_block(
    pdf,
    x: float,
    y: float,
    width: float,
    item: Dict[str, object],
    detail_font_size: int,
    score_font_inc: int,
    std_font_inc: int,
) -> None:
    """Draw the identifier, question text, and scoring summary."""
    pdf.set_xy(x, y)
    ident = item.get("identifier") if item else None
    ident_text = ident if ident else "missing"
    pdf.set_font("Arial", "B", detail_font_size)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(w=width, h=6, txt=latin1_safe(ident_text), align="L")
    pdf.set_xy(x, pdf.get_y())

    q_text = None
    meta = item.get("meta", {}) if item else {}
    q_text = meta.get("__lookup_question")
    q_str = f"Question: {q_text}" if (q_text is not None and str(q_text).strip()) else "Question: N/A"
    pdf.set_font("Arial", "", detail_font_size)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(w=width, h=6, txt=latin1_safe(q_str), align="L")
    pdf.set_xy(x, pdf.get_y())

    score = meta.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
    score_std = meta.get("score_std")
    score_std_str = str(score_std) if score_std is not None else "N/A"
    flag_review = meta.get("flag_review")
    flag_review_str = str(flag_review) if flag_review is not None else "N/A"

    line_y = pdf.get_y()
    pdf.set_xy(x, line_y)
    h = 6

    w1 = width * 0.40
    w2 = width * 0.30
    w3 = width - w1 - w2

    pdf.set_font("Arial", "B", detail_font_size + score_font_inc)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(w=w1, h=h, txt=latin1_safe(f"Score: {score_str}"), align="L", ln=0)

    if score_std == "low":
        pdf.set_text_color(0, 0, 0)
    elif score_std == "medium":
        pdf.set_text_color(0, 0, 200)
    elif score_std == "high":
        pdf.set_text_color(255, 140, 0)
    else:
        pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", "B", detail_font_size + std_font_inc)
    pdf.cell(w=w2, h=h, txt=latin1_safe(f"Std: {score_std_str}"), align="L", ln=0)

    pdf.set_font("Arial", "B", detail_font_size + std_font_inc)
    if flag_review == 1:
        pdf.set_text_color(128, 0, 128)
    elif flag_review == 0:
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.set_text_color(128, 128, 128)
    pdf.cell(w=w3, h=h, txt=latin1_safe(f"Flag: {flag_review_str}"), align="L", ln=1)


def combine_scores(df):
    """Combine multiple scores per identifier into score_1, score_2, ... columns."""
    if "identifier" not in df.columns or "score" not in df.columns:
        raise ValueError("Expected columns 'identifier' and 'score' not found in dataframe.")
    df = df.copy()
    df["score_no"] = df.groupby("identifier").cumcount() + 1
    df_wide = df.pivot(index="identifier", columns="score_no", values="score")
    df_wide.columns = [f"score_{i}" for i in df_wide.columns]
    return df_wide.reset_index()


def resolve_openai_api_key() -> Tuple[str, Optional[str]]:
    """Resolve API key from config or environment, and return (source, key)."""
    if config.OPENAI_API_KEY:
        return "config", config.OPENAI_API_KEY
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return "environment", env_key
    return "missing", None


def confirm_overwrite(path: Path, label: str) -> bool:
    """Confirm overwrite of an existing file."""
    if not path.exists():
        return True
    choice = input(f"{label} already exists at {path}. Overwrite? (y/n): ").strip().lower()
    if choice == "y":
        path.unlink()
        return True
    return False


def confirm_clear_dir(path: Path, label: str) -> bool:
    """Confirm clearing a directory before writing new outputs."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return True
    if not any(path.iterdir()):
        return True
    choice = input(f"{label} is not empty at {path}. Clear contents? (y/n): ").strip().lower()
    if choice == "y":
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    return False


def generate_run_id(prefix: str = "run") -> str:
    """Generate a time-based run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def generate_timestamp() -> str:
    """Generate a UTC timestamp string suitable for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def extract_timestamp_from_mapping(path: Path) -> Optional[str]:
    """Extract timestamp from anonymized IDs filename."""
    pattern = rf"^{re.escape(config.ANON_ID_PREFIX)}_(\d{{8}}T\d{{6}}Z)\.txt$"
    match = re.match(pattern, path.name)
    if match:
        return match.group(1)
    return None


def build_run_manifest(
    run_id: str,
    mapping_file: Path,
    processed_dir: Path,
    mapping_reused: bool,
) -> Dict[str, object]:
    """Create the initial run-manifest structure."""
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.NAME,
        "language": config.LANGUAGE,
        "image_path": str(config.IMAGE_PATH),
        "processed_dir": str(processed_dir),
        "anonymized_ids_file": str(mapping_file),
        "mapping_reused": mapping_reused,
        "batch_jsonl": None,
        "batch_ids_file": None,
        "aggregation_method": config.AGGREGATION_METHOD,
        "results_csv": None,
        "aggregated_results_csv": None,
        "pdf_output": None,
    }


def write_run_manifest(path: Path, data: Dict[str, object]) -> None:
    """Write a run-manifest JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_latest_manifest(manifest_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, object]]]:
    """Load the most recently modified run-manifest JSON."""
    if not manifest_dir.exists():
        return None, None
    manifests = list(manifest_dir.glob("*.json"))
    if not manifests:
        return None, None
    latest = max(manifests, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest, "r", encoding="utf-8") as handle:
            return latest, json.load(handle)
    except Exception:
        return latest, None


def update_manifest_fields(manifest_dir: Path, updates: Dict[str, object]) -> None:
    """Update the latest manifest with new fields."""
    path, data = load_latest_manifest(manifest_dir)
    if not path or data is None:
        print("Warning: no valid run manifest found to update.")
        return
    data.update(updates)
    write_run_manifest(path, data)


def _valid(arr):
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]


def _pairwise_mean_abs_diff(arr):
    a = _valid(arr)
    if a.size < 2:
        return 0.0
    return float(np.mean([abs(x - y) for x, y in combinations(a, 2)]))


def _bimodality_score_1d(arr):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    a = _valid(arr)
    if a.size < 3 or np.nanstd(a) < 1e-8 or np.unique(np.round(a, 8)).size < 2:
        return 0.0
    a = a.reshape(-1, 1)
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(a)
        if len(np.unique(km.labels_)) < 2:
            return 0.0
        sil = silhouette_score(a, km.labels_)
        return float(np.clip(sil, 0.0, 1.0))
    except Exception:
        return 0.0


def _hist_entropy(arr, bins=5):
    from scipy.stats import entropy

    a = _valid(arr)
    if a.size < 2 or np.all(a == a[0]):
        return 0.0
    hist, _ = np.histogram(a, bins=min(bins, max(2, a.size)), density=True)
    p = hist / (hist.sum() + 1e-12)
    h = entropy(p)
    max_h = np.log(len(p)) if len(p) > 0 else 1.0
    return float(h / (max_h + 1e-12))


def _leave_one_out_instability(arr):
    a = _valid(arr)
    k = a.size
    if k <= 1:
        return 0.0
    full_mean = a.mean()
    diffs = [abs(full_mean - np.mean(np.delete(a, i))) for i in range(k)]
    return float(np.max(diffs))


UNSUP_FEATURES = [
    "gpt_std",
    "gpt_iqr",
    "gpt_range",
    "pairwise_madiff",
    "bimodality",
    "hist_entropy",
    "loo_instability",
]


def _apply_frozen_z(col, med, scale, clip=6.0):
    z = (np.nan_to_num(col, nan=med) - med) / (scale if scale != 0 else 1.0)
    return np.clip(z, -clip, clip)


def build_unsup_features(df, score_cols=None, id_col="identifier"):
    """Compute unsupervised features and risk inputs."""
    import pandas as pd

    if score_cols is None:
        score_cols = [c for c in df.columns if c.startswith("score_")]
    X = df.copy()
    S = X[score_cols].astype(float)

    feats = pd.DataFrame(index=X.index)
    if id_col in X.columns:
        feats[id_col] = X[id_col]

    feats["k_valid"] = S.notna().sum(axis=1)
    feats["gpt_mean"] = S.mean(axis=1, skipna=True)
    feats["gpt_std"] = S.std(axis=1, ddof=1, skipna=True).fillna(0.0)
    feats["gpt_min"] = S.min(axis=1, skipna=True)
    feats["gpt_max"] = S.max(axis=1, skipna=True)
    feats["gpt_range"] = (feats["gpt_max"] - feats["gpt_min"]).fillna(0.0)
    q75 = S.quantile(0.75, axis=1, numeric_only=True)
    q25 = S.quantile(0.25, axis=1, numeric_only=True)
    feats["gpt_iqr"] = (q75 - q25).fillna(0.0)

    feats["pairwise_madiff"] = S.apply(lambda r: _pairwise_mean_abs_diff(r.values), axis=1)
    feats["bimodality"] = S.apply(lambda r: _bimodality_score_1d(r.values), axis=1)
    feats["hist_entropy"] = S.apply(
        lambda r: _hist_entropy(r.values, bins=min(5, len(score_cols))), axis=1
    )
    feats["loo_instability"] = S.apply(lambda r: _leave_one_out_instability(r.values), axis=1)

    return feats, score_cols


def build_unsup_features_with_scaler(df, score_cols=None, id_col="identifier", scaler=None):
    """Apply frozen scaler to compute unsupervised risk."""
    feats, score_cols = build_unsup_features(df, score_cols=score_cols, id_col=id_col)
    if scaler is not None:
        for feat in UNSUP_FEATURES:
            med = scaler[feat]["med"]
            scale = scaler[feat]["scale"]
            feats[f"{feat}_rz"] = _apply_frozen_z(feats[feat].astype(float).values, med, scale, clip=6.0)

        feats["unsup_risk"] = (
            0.28 * feats["gpt_std_rz"]
            + 0.18 * feats["gpt_iqr_rz"]
            + 0.18 * feats["gpt_range_rz"]
            + 0.18 * feats["pairwise_madiff_rz"]
            + 0.10 * feats["bimodality_rz"]
            + 0.08 * feats["loo_instability_rz"]
        )
    return feats, score_cols


def apply_unsupervised_policy(new_df, scaler, global_threshold, id_col="identifier"):
    """Apply unsupervised review flag based on frozen scaler and threshold."""
    score_cols = [c for c in new_df.columns if c.startswith("score_")]
    feats_new, _ = build_unsup_features_with_scaler(
        new_df, score_cols=score_cols, id_col=id_col, scaler=scaler
    )
    out = new_df.copy()
    out["unsup_risk"] = feats_new["unsup_risk"]
    out["flag_review"] = (out["unsup_risk"] >= global_threshold).astype(int)
    return out
