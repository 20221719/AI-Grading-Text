"""Improved bonus test processing with manual corner selection and skip logging."""

from __future__ import annotations

from pathlib import Path
import random
import string
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from . import utils
except ImportError:  # Allows running as a script: python src/01_process_bonus_test.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    import config  # type: ignore
    import utils  # type: ignore

SKIP_LOG_PATH = Path(f"skipped_images_{config.NAME}_{config.LANGUAGE}.txt")


def ensure_dirs() -> None:
    """Ensure output and debug directories exist."""
    config.OUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.OMR_DEBUG_STUDENT_ID_DIR.mkdir(parents=True, exist_ok=True)
    config.OMR_DEBUG_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    config.OMR_DEBUG_VERSION_DIR.mkdir(parents=True, exist_ok=True)


def manual_select_markers(image, fname: str) -> Optional[Tuple[Tuple[int, int], ...]]:
    """
    Allow manual selection of corner markers.

    This is helpful when students draw near the corner markers and automatic
    detection fails. Click the center of each black corner marker circle.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(11, 15))
    ax.imshow(rgb)
    ax.set_title(
        "Auto-detection failed. Click the CENTER of the 4 black corner markers in order:\n"
        "1) top-left  2) top-right  3) bottom-left  4) bottom-right\n"
        "Controls: left click = add point, U = undo last, R = reset, Enter = accept, Esc/Close = cancel.",
        fontsize=9,
    )
    ax.axis("off")

    points: List[Tuple[int, int]] = []
    scatter = ax.scatter([], [], c="red", s=30)

    def redraw() -> None:
        if points:
            xs, ys = zip(*points)
            scatter.set_offsets(list(zip(xs, ys)))
        else:
            scatter.set_offsets([])
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes != ax:
            return
        if len(points) >= 4:
            return
        points.append((int(event.xdata), int(event.ydata)))
        redraw()

    def on_key(event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        if key == "u":
            if points:
                points.pop()
                redraw()
        elif key == "r":
            points.clear()
            redraw()
        elif key == "enter":
            plt.close(fig)
        elif key == "escape":
            points.clear()
            plt.close(fig)

    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
    except Exception:
        pass

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if len(points) != 4:
        return None

    return tuple(points)


def align_image_with_manual(
    image, fname: str, skip_log: List[str]
) -> Optional[np.ndarray]:
    """Align image using auto detection, falling back to manual selection."""
    aligned = utils.align_image(image)
    if aligned is not None:
        return aligned

    choice = input(
        f"Auto-detection failed for {fname}. Press Enter to select markers manually,"
        " or type 'skip' to skip this image: "
    ).strip().lower()
    if choice == "skip":
        skip_log.append(f"SKIP {fname}: user skipped manual corner selection")
        return None

    points = manual_select_markers(image, fname)
    if points is None:
        skip_log.append(f"SKIP {fname}: manual corner selection canceled or incomplete")
        return None

    src_pts = np.array(points, dtype=np.float32)
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


def extract_student_id(aligned) -> Tuple[Optional[str], Tuple]:
    """Extract student ID from the aligned image and return (id_str, debug_bundle)."""
    y1, y2, x1, x2 = config.ROIS["student_id"]
    roi = aligned[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bin_img = utils.binarize(gray)

    scores = utils.grid_fill_scores(bin_img, rows=config.ID_ROWS, cols=config.ID_COLS)
    picks = utils.pick_per_column(scores, config.ID_OFFSET_FROM_MEDIAN, config.ID_MIN_MARGIN)

    if any(p is None for p in picks):
        student_id_str = None
    else:
        digits = [str(p) for p in picks]
        student_id_str = "".join(digits)

    return student_id_str, (gray, scores, picks)


def extract_group(aligned) -> Tuple[Optional[str], Tuple]:
    """Extract group value from the aligned image."""
    y1, y2, x1, x2 = config.ROIS["group"]
    roi = aligned[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bin_img = utils.binarize(gray)

    scores = utils.grid_fill_scores(bin_img, rows=1, cols=len(config.GROUP_LABELS))
    idx = utils.pick_one_from_1xN(scores, config.GROUP_OFFSET_FROM_MEDIAN, config.GROUP_MIN_MARGIN)
    value = config.GROUP_LABELS[idx] if idx is not None else None

    return value, (gray, scores, idx)


def extract_version(aligned) -> Tuple[Optional[str], Tuple]:
    """Extract version value from the aligned image."""
    y1, y2, x1, x2 = config.ROIS["version"]
    roi = aligned[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bin_img = utils.binarize(gray)

    scores = utils.grid_fill_scores(bin_img, rows=1, cols=len(config.VERSION_LABELS))
    idx = utils.pick_one_from_1xN(scores, config.VERSION_OFFSET_FROM_MEDIAN, config.VERSION_MIN_MARGIN)
    value = config.VERSION_LABELS[idx] if idx is not None else None

    return value, (gray, scores, idx)


def assign_random_keys(
    results: Dict[str, dict],
    skip_log: List[str],
    existing_student_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Assign random keys to each valid student id and return mapping."""
    random.seed(config.RANDOM_SEED)
    random_keys = set()
    while len(random_keys) < config.RANDOM_KEY_COUNT:
        key = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=config.RANDOM_KEY_LENGTH)
        )
        random_keys.add(key)
    random_keys = list(random_keys)

    existing_student_map = existing_student_map or {}
    used_keys = set(existing_student_map.values())
    random_iter = (k for k in random_keys if k not in used_keys)

    anonymized_student_id: Dict[str, str] = {}
    for fname, entry in results.items():
        student_id = entry.get("student_id")
        if student_id is None:
            skip_log.append(f"SKIP {fname}: student_id is None")
            continue
        if student_id in existing_student_map:
            random_key = existing_student_map[student_id]
        else:
            random_key = next(random_iter, None)
            if random_key is None:
                skip_log.append(f"SKIP {fname}: ran out of random keys")
                continue
        anonymized_student_id[random_key] = student_id
        entry["random_key"] = random_key

    return anonymized_student_id


def write_anonymized_ids(mapping: Dict[str, str], path: Path) -> None:
    """Write anonymized student ids to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        for key, student_id in mapping.items():
            handle.write(f"{key}: {student_id}\n")


def crop_questions(fname: str, aligned, results: Dict[str, dict], skip_log: List[str]) -> None:
    """Crop and save question ROIs using the assigned random key."""
    entry = results.get(fname, {})
    student_id = entry.get("student_id")
    group_val = entry.get("group")
    version_val = entry.get("version")

    if None in (student_id, group_val, version_val):
        skip_log.append(f"SKIP {fname}: incomplete info {entry}")
        return

    random_key = entry.get("random_key")

    if (int(student_id[-1]) % 2 == 0 and version_val != "A") or (
        int(student_id[-1]) % 2 == 1 and version_val != "B"
    ):
        skip_log.append(
            f"SKIP {fname}: version {version_val} does not match student ID {student_id}"
        )
        return

    for qname in ("question1", "question2"):
        y1, y2, x1, x2 = config.ROIS[qname]
        roi = aligned[y1:y2, x1:x2]
        out_name = f"{random_key}_{version_val}_{group_val}_{qname.upper()}_{config.FILE_END}.jpg"
        out_path = config.OUT_PROCESSED_DIR / out_name
        cv2.imwrite(str(out_path), roi)


def check_duplicate_ids(mapping: Dict[str, str], skip_log: List[str]) -> None:
    """Report if any duplicate student IDs were assigned."""
    student_ids = list(mapping.values())
    if len(student_ids) != len(set(student_ids)):
        skip_log.append("WARNING: duplicate entries found in student_id")


def write_skip_log(skip_log: List[str]) -> None:
    """Persist skip and warning messages to a text file."""
    if not skip_log:
        return
    with open(SKIP_LOG_PATH, "w", encoding="utf-8") as handle:
        for line in skip_log:
            handle.write(f"{line}\n")


def process() -> None:
    """Run the full processing pipeline with manual-corner fallback."""
    ensure_dirs()
    skip_log: List[str] = []
    existing_student_map: Dict[str, str] = {}
    mapping_reused = False

    if not utils.confirm_clear_dir(config.OUT_PROCESSED_DIR, "Processed output folder"):
        raise SystemExit("Aborted: processed output folder not cleared.")

    stamp = utils.generate_timestamp()
    run_id = stamp
    mapping_file = Path(f"{config.ANON_ID_PREFIX}_{stamp}.txt")

    legacy_map = Path(config.ANON_ID_FILENAME)
    existing_maps = sorted(
        Path(".").glob(f"{config.ANON_ID_PREFIX}_*.txt"),
        key=lambda p: p.stat().st_mtime,
    )

    if legacy_map.exists():
        print(
            "Found legacy anonymized IDs file. Reusing it preserves linkage to any "
            "previously launched batches. Creating new keys breaks that linkage."
        )
        choice = input(
            f"Reuse legacy anonymized IDs file ({legacy_map.name})? (y/n): "
        ).strip().lower()
        if choice == "y":
            print(
                "Reusing legacy map: known student IDs keep their previous random key; "
                "new student IDs receive new keys."
            )
            existing_student_map = utils.load_student_id_mapping(legacy_map)
            mapping_reused = True
    elif existing_maps:
        latest_map = existing_maps[-1]
        print(
            "Warning: If you generate new random keys (without reusing an existing map), "
            "any previously launched batches that used old keys will no longer match."
        )
        choice = input(
            f"Reuse latest anonymized IDs file ({latest_map.name})? (y/n): "
        ).strip().lower()
        if choice == "y":
            print(
                "Reusing existing map: known student IDs keep their previous random key; "
                "new student IDs receive new keys."
            )
            existing_student_map = utils.load_student_id_mapping(latest_map)
            mapping_reused = True

    if mapping_file.exists():
        if not utils.confirm_overwrite(mapping_file, "Anonymized IDs file"):
            raise SystemExit("Aborted: anonymized IDs file not overwritten.")

    manifest_path = config.RUN_MANIFEST_DIR / f"{run_id}.json"
    manifest = utils.build_run_manifest(
        run_id=run_id,
        mapping_file=mapping_file,
        processed_dir=config.OUT_PROCESSED_DIR,
        mapping_reused=mapping_reused,
    )
    utils.write_run_manifest(manifest_path, manifest)
    images = utils.load_images(config.IMAGE_PATH)

    aligned_images: Dict[str, Optional[np.ndarray]] = {}

    for fname, image in images:
        aligned = align_image_with_manual(image, fname, skip_log)
        aligned_images[fname] = aligned
        if aligned is None:
            continue

        student_id, id_debug = extract_student_id(aligned)
        group_val, group_debug = extract_group(aligned)
        version_val, version_debug = extract_version(aligned)

        results_entry = RESULTS.setdefault(fname, {})
        results_entry["student_id"] = student_id
        results_entry["group"] = group_val
        results_entry["version"] = version_val

        utils.debug_grid(
            id_debug[0],
            id_debug[1],
            picks=id_debug[2],
            title=f"{fname} - Student ID: {student_id}",
            path=config.OMR_DEBUG_STUDENT_ID_DIR / f"{Path(fname).stem}_student_id_debug.png",
        )
        utils.debug_grid(
            group_debug[0],
            group_debug[1],
            picks=group_debug[2],
            title=f"{fname} - Group: {group_val}",
            path=config.OMR_DEBUG_GROUP_DIR / f"{Path(fname).stem}_group_debug.png",
        )
        utils.debug_grid(
            version_debug[0],
            version_debug[1],
            picks=version_debug[2],
            title=f"{fname} - Version: {version_val}",
            path=config.OMR_DEBUG_VERSION_DIR / f"{Path(fname).stem}_version_debug.png",
        )

    anonymized = assign_random_keys(RESULTS, skip_log, existing_student_map)
    write_anonymized_ids(anonymized, mapping_file)

    if legacy_map.exists() and mapping_reused:
        legacy_random_map = utils.load_random_key_mapping(legacy_map)
        merged_random_map = dict(legacy_random_map)
        merged_random_map.update(anonymized)
        write_anonymized_ids(merged_random_map, legacy_map)
    elif legacy_map.exists() and not mapping_reused:
        choice = input(
            f"Overwrite legacy anonymized IDs file ({legacy_map.name}) with new keys? (y/n): "
        ).strip().lower()
        if choice == "y":
            write_anonymized_ids(anonymized, legacy_map)

    for fname, _image in images:
        aligned = aligned_images.get(fname)
        if aligned is None:
            continue
        crop_questions(fname, aligned, RESULTS, skip_log)

    check_duplicate_ids(anonymized, skip_log)
    write_skip_log(skip_log)


RESULTS: Dict[str, dict] = {}

if __name__ == "__main__":
    process()
