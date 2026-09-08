"""Create a manual-check PDF with questions, scores, and explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fpdf import FPDF

try:
    from . import config
    from . import utils
except ImportError:  # Allows running as a script: python src/04_create_manual_check_pdf.py
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent))
    import config  # type: ignore
    import utils  # type: ignore


def ensure_dirs() -> None:
    """Ensure results output directory exists."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    """Load the max-score results CSV."""
    if not config.BATCH_RESULTS_AGG_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {config.BATCH_RESULTS_AGG_CSV}")
    return pd.read_csv(config.BATCH_RESULTS_AGG_CSV)


def load_scores_for_flags() -> pd.DataFrame:
    """Load the raw batch results to compute review flags."""
    if not config.BATCH_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing batch results file: {config.BATCH_RESULTS_CSV}")
    return pd.read_csv(config.BATCH_RESULTS_CSV)


def build_question_lookup() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load the question metadata and build a lookup dictionary."""
    df = utils.load_questions_table(config.QUESTIONS_XLSX)
    df["Group"] = df["Group"].astype(str)
    return utils.build_bonus_test_dictionary(df)


def add_flag_review(df: pd.DataFrame) -> pd.DataFrame:
    """Compute unsupervised review flags and merge into df."""
    df_scores = load_scores_for_flags()
    df_wide = utils.combine_scores(df_scores)
    df_scored = utils.apply_unsupervised_policy(df_wide, config.scaler, config.global_thr)
    df_flags = df_scored[["identifier", "flag_review"]]
    return df.merge(df_flags, on="identifier", how="left")


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort results for PDF output."""
    df = df.copy()
    df["group"] = df["group"].astype(str)
    df["question"] = df["question"].astype(int)
    df["version"] = df["version"].astype(str)

    df["group"] = pd.Categorical(df["group"], categories=["1", "2", "3", "4", "5"], ordered=True)
    df["question"] = pd.Categorical(df["question"], categories=[1, 2], ordered=True)
    df["version"] = pd.Categorical(df["version"], categories=["A", "B"], ordered=True)

    df = df.sort_values(by=["question", "group", "version"])
    mask = df["question"] == 2
    df_question2 = df[mask].sort_values(by=["version", "group"])
    df_question1 = df[~mask]
    df = pd.concat([df_question1, df_question2], ignore_index=True)

    df["explanation"] = df["explanation"].apply(utils.clean_explanation)
    return df.reset_index(drop=True)


def build_id_to_meta(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Create a lookup dict for identifier metadata."""
    id_to_meta: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        identifier = str(row.get("identifier"))
        score = row.get("score")
        flag = row.get("flag")
        score_std = row.get("score_std")
        score_std_cat = row.get("score_std_cat")
        explanation = row.get("explanation")
        version = row.get("version")
        group = row.get("group")
        question = row.get("question")
        flag_review = row.get("flag_review")

        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None

        version = str(version).strip().upper() if version is not None else None
        if version not in ("A", "B"):
            version = None

        group = str(group).strip() if group is not None else None

        if isinstance(explanation, str):
            explanation = explanation.strip() or None

        id_to_meta[identifier] = {
            "score": score,
            "flag": flag,
            "score_std": score_std_cat if score_std_cat is not None else score_std,
            "explanation": explanation,
            "version": version,
            "group": group,
            "question": question,
            "flag_review": flag_review,
        }

    return id_to_meta


def lookup_question_text(item: dict, bonus_test_dictionary: dict) -> Optional[str]:
    """Lookup question text based on group, version and question number."""
    if not item:
        return None
    meta = item.get("meta", {})
    group = meta.get("group")
    version = meta.get("version")
    qnum = item.get("q")

    if not group:
        return None

    if qnum == 1:
        ver_key = config.QUESTION1_VERSION
    elif qnum == 2:
        ver_key = version if version in ("A", "B") else None
    else:
        ver_key = None

    if not ver_key:
        return None

    try:
        return str(bonus_test_dictionary[str(group)][ver_key]["Question"])
    except Exception:
        return None


def group_images_with_meta(
    src_dir: Path,
    id_to_meta: Dict[str, Dict[str, object]],
    bonus_test_dictionary: dict,
) -> Dict[str, dict]:
    """Group images by random key and attach metadata."""
    images = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")))
    grouped: Dict[str, dict] = {}

    for img_path in images:
        stem = img_path.stem
        random_key, qnum = utils.parse_student_and_question(stem)
        if random_key is None or qnum not in (1, 2):
            continue
        grouped.setdefault(random_key, {})
        if qnum not in grouped[random_key]:
            item = {
                "path": img_path,
                "identifier": stem,
                "q": qnum,
                "meta": id_to_meta.get(stem, {}),
            }
            item["meta"]["__lookup_question"] = lookup_question_text(item, bonus_test_dictionary)
            grouped[random_key][qnum] = item

    return grouped


def build_identifier_map(grouped: Dict[str, dict]) -> Dict[str, dict]:
    """Flatten grouped images into an identifier map without reordering."""
    by_identifier: Dict[str, dict] = {}
    for _, entry in grouped.items():
        for qnum in (1, 2):
            item = entry.get(qnum)
            if not item:
                continue
            ident = str(item["identifier"]).strip()
            if ident not in by_identifier:
                by_identifier[ident] = item
    return by_identifier


def render_pdf(
    ordered_identifiers: List[str],
    by_identifier: Dict[str, dict],
    student_map: Dict[str, str],
) -> None:
    """Render the manual-check PDF."""
    if not utils.confirm_overwrite(config.RESULTS_PDF_EXPLANATION, "Results PDF"):
        raise SystemExit("Aborted: existing PDF not overwritten.")

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)

    for ident in ordered_identifiers:
        item = by_identifier.get(ident)
        if not item:
            continue

        sid, _ = utils.parse_student_and_question(ident)
        pdf.add_page()
        header_bottom_y = utils.draw_student_header(
            pdf,
            sid or "",
            student_map,
            margin=config.PDF_MARGIN,
            page_w=config.PDF_PAGE_W,
            title_font_size=config.PDF_TITLE_FONT_SIZE,
            header_block_h=config.PDF_HEADER_BLOCK_H,
            return_compact_y=True,
            gap_after=2.0,
        )

        max_w = config.PDF_PAGE_W - 2 * config.PDF_MARGIN
        draw_area_top = header_bottom_y
        max_h = config.PDF_PAGE_H - draw_area_top - config.PDF_MARGIN

        img_w, img_h = utils.natural_size_fit_width_mm(item["path"], max_w)

        needed_h = (
            config.PDF_INFO_BLOCK_H
            + config.PDF_GAP_INFO_TO_IMG
            + img_h
            + config.PDF_GAP_IMG_TO_EXPL
            + config.PDF_EXPL_RESERVED_H
        )
        if needed_h > max_h:
            avail_for_img = max_h - (
                config.PDF_INFO_BLOCK_H
                + config.PDF_GAP_INFO_TO_IMG
                + config.PDF_GAP_IMG_TO_EXPL
                + config.PDF_EXPL_RESERVED_H
            )
            avail_for_img = max(avail_for_img, 15)
            scale = avail_for_img / img_h
            img_w *= scale
            img_h *= scale

        block_w = img_w if config.PDF_CENTER_BLOCK else max_w
        block_x = config.PDF_MARGIN + (max_w - block_w) / 2
        info_y = draw_area_top

        utils.draw_info_block(
            pdf,
            block_x,
            info_y,
            block_w,
            item,
            detail_font_size=config.PDF_DETAIL_FONT_SIZE,
            score_font_inc=config.PDF_SCORE_FONT_INC,
            std_font_inc=config.PDF_STD_FONT_INC,
        )

        img_x = config.PDF_MARGIN + (max_w - img_w) / 2
        img_y = info_y + config.PDF_INFO_BLOCK_H + config.PDF_GAP_INFO_TO_IMG
        pdf.image(str(item["path"]), x=img_x, y=img_y, w=img_w, h=img_h)

        expl = item["meta"].get("explanation")
        expl = utils.latin1_safe(expl) if (expl is not None and str(expl).strip()) else None
        expl_text = f"Explanation: {expl}" if expl else "Explanation: N/A"
        expl_text_safe = utils.latin1_safe(expl_text)

        expl_y = img_y + img_h + config.PDF_GAP_IMG_TO_EXPL
        pdf.set_font("Arial", "", config.PDF_DETAIL_FONT_SIZE)
        pdf.set_text_color(0, 0, 0)

        available_h = config.PDF_PAGE_H - config.PDF_MARGIN - expl_y
        lines = utils.estimate_line_count(pdf, expl_text_safe, max_w)
        needed_h = max(1, lines) * 6

        if needed_h > available_h:
            pdf.add_page()
            expl_y = config.PDF_MARGIN

        pdf.set_xy(config.PDF_MARGIN, expl_y)
        pdf.multi_cell(w=max_w, h=6, txt=expl_text_safe, align="L")

    pdf.output(str(config.RESULTS_PDF_EXPLANATION))
    utils.update_manifest_fields(
        config.RUN_MANIFEST_DIR, {"pdf_output": str(config.RESULTS_PDF_EXPLANATION)}
    )


def process() -> None:
    """Main entrypoint for manual-check PDF generation."""
    ensure_dirs()

    df = load_results()
    df = add_flag_review(df)
    df = normalize_results(df)

    id_to_meta = build_id_to_meta(df)
    question_lookup = build_question_lookup()
    manifest_path, manifest = utils.load_latest_manifest(config.RUN_MANIFEST_DIR)
    mapping_path = None
    if manifest and manifest.get("anonymized_ids_file"):
        mapping_path = Path(str(manifest.get("anonymized_ids_file")))
    if mapping_path is None:
        mapping_path = Path(config.ANON_ID_FILENAME)
    student_map = utils.load_random_key_mapping(mapping_path)

    grouped = group_images_with_meta(config.OUT_PROCESSED_DIR, id_to_meta, question_lookup)
    by_identifier = build_identifier_map(grouped)

    if "identifier" in df.columns:
        ordered_identifiers = [str(x).strip() for x in df["identifier"].tolist()]
    else:
        ordered_identifiers = list(by_identifier.keys())

    render_pdf(ordered_identifiers, by_identifier, student_map)


if __name__ == "__main__":
    process()
