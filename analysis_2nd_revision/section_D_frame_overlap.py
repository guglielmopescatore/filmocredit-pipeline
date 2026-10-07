#!/usr/bin/env python3
"""
Section D - are the frames of the hand-cut clips also selected on the full films?

The clips and the films are different encodes (different frame numbering, often
different resolution or letterbox), so frames cannot be matched by number. Two
complementary measures, per clip frame and per title:

1. Visual match (perceptual hash). Black borders are cropped, then the 64-bit
   pHash of each clip frame is compared with every film frame of the same title;
   the closest one and its Hamming distance are kept. A clip frame is "matched"
   if the distance is <= MATCH_DISTANCE (also reported at the looser LOOSE_DISTANCE).
   On static cards this is exact; on credit rolls the film may sample the same
   roll a few frames earlier or later, so the text sits higher or lower and the
   hash differs although the credits are captured.

2. Text coverage (Stage II OCR). Stage II stores the OCR text of every saved frame
   in analysis_manifest.json. For each clip frame: share of its OCR words (lower
   case, accents removed, >= 3 characters) that occur among the OCR words of the
   film's frames. Robust to the roll offset above; it measures whether the
   credits' text reached the film's frames, not pixel identity.

Manual check: the likely missing frames go to results/D_likely_missing_review.csv
(";"-separated) with a side-by-side image each in results/D_likely_missing/ (local only, git-ignored):
clip frame | closest film frame by pHash | film frame sharing most OCR words.
Mark the "controllo" column with
  v  credit present in the film frames (frame_film_verificato: the film frame that has it),
  b  present but badly extracted in the film (cross-fade, blur, cut),
  x  credit absent from the film frames (tipo: persona / titolo / logo / edizione),
  n  the clip frame itself shows no credit (watermark, props, transition): nothing to miss.
Rerunning the script keeps the marks, writes three short lists
(D_review_present_in_film.csv, D_review_missing_in_film.csv, D_review_no_credit_in_clip.csv)
and reports the counts after the check.

Inputs: data/episodes/<clip>/analysis (reference Stage II, 2025-12-04, the frames
Stage III used) and runs/FULL_pipeline/data/episodes/<title>_FULL/analysis.
Outputs: results/D_frame_overlap.csv (per title), results/D_frame_overlap_frames.csv
(per clip frame), results/D_frame_overlap.md, the review CSV and images.

Usage: python analysis_2nd_revision/section_D_frame_overlap.py
"""

import csv
import json
import re
import unicodedata
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CUT = ROOT / "data" / "episodes"
FULL = ROOT / "runs" / "FULL_pipeline" / "data" / "episodes"
RESULTS = HERE / "results"
REVIEW_CSV = RESULTS / "D_likely_missing_review.csv"
REVIEW_IMAGES = RESULTS / "D_likely_missing"
PANEL_HEIGHT = 360

MATCH_DISTANCE = 10   # pHash Hamming distance (of 64 bits) for "same frame"
LOOSE_DISTANCE = 16
MISSING_COVERAGE = 0.5  # likely missing: no loose visual match and < 50% of its OCR words found
BORDER_LEVEL = 16     # rows/columns darker than this (0-255 mean) count as letterbox
WORD = re.compile(r"[a-z0-9]{3,}")


def crop_borders(img: Image.Image) -> Image.Image:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    rows = np.where(gray.mean(axis=1) > BORDER_LEVEL)[0]
    cols = np.where(gray.mean(axis=0) > BORDER_LEVEL)[0]
    if len(rows) < 8 or len(cols) < 8:  # (almost) black frame: keep it whole
        return img
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(crop_borders(img))


def words(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return set(WORD.findall(text))


def frames_with_text(analysis: Path) -> dict[str, str]:
    """Frame file -> OCR text, for the frames Stage III receives (manifest entries present in frames/)."""
    manifest = json.loads((analysis / "analysis_manifest.json").read_text(encoding="utf-8"))
    out = {}
    for scene in manifest.get("scenes", {}).values():
        for f in scene.get("output_files", []):
            name = Path(f.get("path") or "").name
            if name and (analysis / "frames" / name).is_file():
                out[name] = f.get("ocr_text") or ""
    return out


def side_by_side(paths_labels: list[tuple[Path, str]], out: Path) -> None:
    panels = []
    for path, label in paths_labels:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((int(img.width * PANEL_HEIGHT / img.height), PANEL_HEIGHT))
        canvas = Image.new("RGB", (img.width, PANEL_HEIGHT + 24), "white")
        canvas.paste(img, (0, 24))
        ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
        panels.append(canvas)
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1), PANEL_HEIGHT + 24), "white")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + 8
    sheet.save(out, quality=88)


MANUAL_FIELDS = ("controllo", "tipo", "nota", "frame_film_verificato")


def load_marks() -> dict[tuple[str, str], dict]:
    """(clip, clip_frame) -> manual fields already written in the review CSV."""
    if not REVIEW_CSV.exists():
        return {}
    with open(REVIEW_CSV, encoding="utf-8-sig", newline="") as f:
        return {(r["clip"], r["clip_frame"]): {k: (r.get(k) or "").strip() for k in MANUAL_FIELDS}
                for r in csv.DictReader(f, delimiter=";")}


def write_short_lists(rows: list[dict]) -> None:
    """Three short ';'-separated lists of the checked frames, with only the files to look at."""
    film_dir = lambda r: FULL / f"{r['title']}_FULL" / "analysis" / "frames"
    lists = {
        "D_review_present_in_film.csv": [r for r in rows if r["controllo"] in ("v", "b")],
        "D_review_missing_in_film.csv": [r for r in rows if r["controllo"] == "x"],
        "D_review_no_credit_in_clip.csv": [r for r in rows if r["controllo"] == "n"],
    }
    for name, sub in lists.items():
        out = []
        for r in sub:
            film_frame = r["frame_film_verificato"] or r["best_film_frame_text"]
            out.append({"n": r["n"], "esito": r["controllo"], "tipo": r["tipo"], "titolo": r["title"], "nota": r["nota"],
                        "frame_clip": r["clip_frame_path"],
                        "frame_film": str(film_dir(r) / film_frame) if r["controllo"] != "n" else ""})
        with open(RESULTS / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n", "esito", "tipo", "titolo", "nota", "frame_clip", "frame_film"], delimiter=";")
            w.writeheader()
            w.writerows(out)


def write_review(frame_rows: list[dict], marks: dict) -> None:
    REVIEW_IMAGES.mkdir(parents=True, exist_ok=True)
    for old in REVIEW_IMAGES.glob("*.jpg"):  # numbering changes when the flagged set changes
        old.unlink()
    rows = []
    for i, r in enumerate((r for r in frame_rows if r["likely_missing"]), 1):
        clip_dir = CUT / r["clip"] / "analysis" / "frames"
        film_dir = FULL / f"{r['title']}_FULL" / "analysis" / "frames"
        image = REVIEW_IMAGES / f"{i:03d}_{r['clip']}_{Path(r['clip_frame']).stem}.jpg"
        side_by_side([(clip_dir / r["clip_frame"], f"CLIP {r['clip']} / {r['clip_frame']}"),
                      (film_dir / r["best_film_frame"], f"FILM closest pHash ({r['phash_distance']}) {r['best_film_frame']}"),
                      (film_dir / r["best_text_film_frame"], f"FILM most shared words ({r['best_text_share']}) {r['best_text_film_frame']}")],
                     image)
        manual = marks.get((r["clip"], r["clip_frame"]), {k: "" for k in MANUAL_FIELDS})
        rows.append({"n": i, **manual, "title": r["title"], "clip": r["clip"],
                     "clip_frame": r["clip_frame"], "image": str(image.relative_to(HERE)),
                     "phash_distance": r["phash_distance"], "ocr_word_coverage": r["ocr_word_coverage"],
                     "best_film_frame_phash": r["best_film_frame"], "best_film_frame_text": r["best_text_film_frame"],
                     "best_text_share": r["best_text_share"], "clip_ocr_text": r["clip_ocr_text"],
                     "clip_frame_path": str(clip_dir / r["clip_frame"]),
                     "film_frame_phash_path": str(film_dir / r["best_film_frame"]),
                     "film_frame_text_path": str(film_dir / r["best_text_film_frame"])})
    with open(REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    write_short_lists(rows)


def main() -> None:
    title_rows, frame_rows = [], []
    marks = load_marks()
    for film_dir in sorted(FULL.iterdir()):
        title = film_dir.name[: -len("_FULL")]
        clips = [title] if (CUT / title).is_dir() else sorted(
            p.name for p in CUT.iterdir() if p.name.startswith(title + "_") and (p / "analysis" / "frames").is_dir())
        film_frames = frames_with_text(film_dir / "analysis")
        film_hashes = [(name, phash(film_dir / "analysis" / "frames" / name)) for name in film_frames]
        film_frame_words = {n: words(t) for n, t in film_frames.items()}
        film_words = set().union(*film_frame_words.values())

        distances, coverages, clip_words, missing, checked_missing, checked_present, checked_nocredit = (
            [], [], set(), [], [], [], [])
        for clip in clips:
            analysis = CUT / clip / "analysis"
            for name, text in frames_with_text(analysis).items():
                h = phash(analysis / "frames" / name)
                best_name, best = min(((n, h - fh) for n, fh in film_hashes), key=lambda x: x[1])
                w = words(text)
                clip_words |= w
                cov = len(w & film_words) / len(w) if w else None
                text_name, text_share = max(((n, len(w & fw) / len(w) if w else 0.0) for n, fw in film_frame_words.items()),
                                            key=lambda x: x[1])
                is_missing = best > LOOSE_DISTANCE and (cov is None or cov < MISSING_COVERAGE)
                distances.append(best)
                missing.append(is_missing)
                if cov is not None:
                    coverages.append(cov)
                mark = marks.get((clip, name), {}).get("controllo", "").lower() if is_missing else ""
                checked_missing.append(mark == "x")
                checked_present.append(mark in ("v", "b"))
                checked_nocredit.append(mark == "n")
                frame_rows.append({"title": title, "clip": clip, "clip_frame": name, "best_film_frame": best_name,
                                   "phash_distance": best, "matched": best <= MATCH_DISTANCE,
                                   "ocr_words": len(w), "ocr_word_coverage": None if cov is None else round(cov, 3),
                                   "best_text_film_frame": text_name, "best_text_share": round(text_share, 3),
                                   "likely_missing": is_missing, "manual_check": mark, "clip_ocr_text": text})
        n = len(distances)
        title_rows.append({
            "title": title, "clip_frames": n, "film_frames": len(film_frames),
            "matched_pct": round(100 * sum(d <= MATCH_DISTANCE for d in distances) / n, 1),
            "matched_loose_pct": round(100 * sum(d <= LOOSE_DISTANCE for d in distances) / n, 1),
            "median_distance": float(np.median(distances)),
            "frames_with_text": len(coverages),
            "mean_word_coverage_pct": round(100 * sum(coverages) / len(coverages), 1) if coverages else None,
            "frames_full_text_pct": round(100 * sum(c == 1.0 for c in coverages) / len(coverages), 1) if coverages else None,
            "unique_clip_words": len(clip_words),
            "unique_words_found_pct": round(100 * len(clip_words & film_words) / len(clip_words), 1) if clip_words else None,
            "likely_missing": sum(missing),
            "checked_missing_x": sum(checked_missing),
            "checked_present_v": sum(checked_present),
            "checked_no_credit_n": sum(checked_nocredit),
            "unchecked": sum(missing) - sum(checked_missing) - sum(checked_present) - sum(checked_nocredit),
            "present_after_check_pct": round(100 * (n - sum(missing) + sum(checked_present) + sum(checked_nocredit)) / n, 1),
        })
        print(f"{title}: {title_rows[-1]}")

    all_d = [r["phash_distance"] for r in frame_rows]
    all_c = [r["ocr_word_coverage"] for r in frame_rows if r["ocr_word_coverage"] is not None]
    total = {"title": "**Total**", "clip_frames": len(all_d), "film_frames": sum(r["film_frames"] for r in title_rows),
             "matched_pct": round(100 * sum(d <= MATCH_DISTANCE for d in all_d) / len(all_d), 1),
             "matched_loose_pct": round(100 * sum(d <= LOOSE_DISTANCE for d in all_d) / len(all_d), 1),
             "median_distance": float(np.median(all_d)), "frames_with_text": len(all_c),
             "mean_word_coverage_pct": round(100 * sum(all_c) / len(all_c), 1),
             "frames_full_text_pct": round(100 * sum(c == 1.0 for c in all_c) / len(all_c), 1),
             "unique_clip_words": sum(r["unique_clip_words"] for r in title_rows), "unique_words_found_pct": "",
             "likely_missing": sum(r["likely_missing"] for r in title_rows),
             "checked_missing_x": sum(r["checked_missing_x"] for r in title_rows),
             "checked_present_v": sum(r["checked_present_v"] for r in title_rows),
             "checked_no_credit_n": sum(r["checked_no_credit_n"] for r in title_rows),
             "unchecked": sum(r["unchecked"] for r in title_rows)}
    total["present_after_check_pct"] = round(
        100 * (total["clip_frames"] - total["likely_missing"] + total["checked_present_v"] + total["checked_no_credit_n"])
        / total["clip_frames"], 1)

    RESULTS.mkdir(exist_ok=True)
    write_review(frame_rows, marks)
    for rows, name in ((title_rows, "D_frame_overlap.csv"), (frame_rows, "D_frame_overlap_frames.csv")):
        with open(RESULTS / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    cols = [("title", "Title"), ("clip_frames", "Clip frames"), ("film_frames", "Film frames"),
            ("matched_pct", f"Visual match % (pHash <= {MATCH_DISTANCE})"),
            ("matched_loose_pct", f"Visual match % (<= {LOOSE_DISTANCE})"), ("median_distance", "Median distance"),
            ("mean_word_coverage_pct", "Mean OCR-word coverage %"), ("frames_full_text_pct", "Frames with all words found %"),
            ("unique_words_found_pct", "Unique clip words found %"), ("likely_missing", "Likely missing frames"),
            ("checked_present_v", "Checked: present (v/b)"), ("checked_no_credit_n", "Checked: no credit in clip frame (n)"),
            ("checked_missing_x", "Checked: missing (x)"),
            ("unchecked", "Not checked"), ("present_after_check_pct", "Clip frames present % (after check)")]
    table = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"]
    table += ["| " + " | ".join(str(r[k]) for k, _ in cols) + " |" for r in title_rows + [total]]
    md = ["# Section D - clip frames found among the full-film frames", "",
          "For every frame the paper's Stage III received from a hand-cut clip, the closest frame Stage II selected "
          "on the full film of the same title. **Visual match**: perceptual hash (pHash, 64 bits) after cropping "
          f"black borders; matched if the Hamming distance is <= {MATCH_DISTANCE} ({LOOSE_DISTANCE} = loose). "
          "**OCR-word coverage**: share of the clip frame's OCR words (Stage II OCR, >= 3 characters) found among "
          "the OCR words of the film's frames; it also counts credit rolls sampled a few frames apart, where the "
          "image differs but the text is the same. **Likely missing**: clip frames with no loose visual match "
          f"(distance > {LOOSE_DISTANCE}) and less than {int(MISSING_COVERAGE * 100)}% of their OCR words found; "
          "OCR noise differs between encodes, so this is an upper bound to check by eye.", "", "\n".join(table), "",
          "Per-frame details (closest film frame, distance, coverage) in `results/D_frame_overlap_frames.csv`.", ""]
    (RESULTS / "D_frame_overlap.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
