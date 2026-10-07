#!/usr/bin/env python3
"""
Section D - full-length films, Stage I and II only (RIEPILOGO_run_v2.md, block D).

Compares, title by title, the Stage I/II of the hand-cut credit clips (reference
run of 2025-12-04, commit 00f0cbc5, the frames the paper's Stage III used) with
the Stage I ("Run on the whole episode", no manual deselection) and Stage II
(cap 150) run on the full films (runs/FULL_pipeline, commit f9c9bb6, OCR en).
Stage III/IV were not run on the full films.

Writes analysis_2nd_revision/D_full_films_stage1_stage2.md and results/D_full_films.csv.
Usage: python analysis_2nd_revision/section_D_full_films.py
"""

import csv
import json
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CUT = ROOT / "data" / "episodes"
FULL = ROOT / "runs" / "FULL_pipeline"


def stage_stats(analysis: Path) -> dict:
    raw = json.loads((analysis / "raw_scenes_cache.json").read_text(encoding="utf-8"))
    cand = json.loads((analysis / "initial_scene_analysis.json").read_text(encoding="utf-8"))["candidate_scenes"]
    frames_dir = analysis / "frames"
    cand_frames = sum(s["original_end_frame"] - s["original_start_frame"] for s in cand)
    return {
        "minutes": raw["total_frames"] / raw["fps"] / 60,
        "shots": len(raw["scenes"]),
        "candidates": len(cand),
        "candidate_minutes": cand_frames / raw["fps"] / 60,
        "frames": len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")]),
    }


KIND_EN = {"persona": "person credits", "logo": "company logos", "titolo": "titles (series / episode)",
           "edizione": "edition-specific card (Italian dubbing)"}


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def frame_overlap_section() -> list[str]:
    """Clip frames vs film frames, from section_D_frame_overlap.py and its manual review (if run)."""
    results = HERE / "results"
    if not (results / "D_frame_overlap.csv").exists():
        return ["## Clip frames found among the film frames", "",
                "Run `section_D_frame_overlap.py` first, then this script again.", ""]
    per_title = read_csv(results / "D_frame_overlap.csv")
    review = read_csv(results / "D_likely_missing_review.csv", ";") if (results / "D_likely_missing_review.csv").exists() else []
    total_clip = sum(int(r["clip_frames"]) for r in per_title)
    likely = sum(int(r["likely_missing"]) for r in per_title)
    by_mark = {m: [r for r in review if r["controllo"].lower() == m] for m in "vbxn"}
    unchecked = likely - sum(len(v) for v in by_mark.values())
    present = total_clip - likely + len(by_mark["v"]) + len(by_mark["b"]) + len(by_mark["n"])
    w = lambda r, k: float(r[k]) * int(r["clip_frames"])
    visual = sum(w(r, "matched_pct") for r in per_title) / total_clip
    visual_loose = sum(w(r, "matched_loose_pct") for r in per_title) / total_clip

    cols = [("title", "Title"), ("clip_frames", "Clip frames"), ("film_frames", "Film frames"),
            ("matched_pct", "Visual match % (pHash <= 10)"), ("matched_loose_pct", "Visual match % (<= 16)"),
            ("mean_word_coverage_pct", "Mean OCR-word coverage %"), ("likely_missing", "Likely missing"),
            ("checked_missing_x", "Missing after check"), ("present_after_check_pct", "Present % after check")]
    table = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"]
    table += ["| " + " | ".join(str(r[k]) for k, _ in cols) + " |" for r in per_title]

    missing = by_mark["x"]
    kinds = {}
    for r in missing:
        kinds.setdefault(r["tipo"] or "-", []).append(r)
    miss_table = ["| n | Title | Type | Why |", "|---:|---|---|---|"]
    miss_table += [f"| {r['n']} | {r['title']} | {r['tipo']} | {r['nota']} |" for r in missing]

    return [
        "## Clip frames found among the film frames", "",
        "Are the frames the paper's Stage III received from the hand-cut clips also selected by Stage II on the full "
        "films? Clips and films are different encodes (frame numbers, resolution, letterbox), so frames are matched by "
        "content (`section_D_frame_overlap.py`): (1) perceptual hash (pHash, 64 bits) after cropping black borders, "
        "closest film frame of the same title; (2) share of the clip frame's Stage II OCR words found among the film "
        "frames' OCR words, which also covers credit rolls sampled a few frames apart. A clip frame is *likely missing* "
        "when it has no visual match within distance 16 and less than 50% of its OCR words are found.", "",
        f"Of {total_clip:,} clip frames, {visual:.1f}% have a visual match (pHash <= 10; {visual_loose:.1f}% within 16) "
        f"and {likely} ({100 * likely / total_clip:.1f}%) were flagged as likely missing.", "",
        "**Manual check of the flagged frames** (each looked at side by side with the closest film frame by pHash and "
        "by shared OCR words, and the credit's names searched, also fuzzily, in the OCR text of every film frame; "
        "images in `results/D_likely_missing/`, kept locally only, not in the repository):", "",
        f"- **{len(by_mark['v'])} present** in the film frames (v) and **{len(by_mark['b'])} present but badly extracted** "
        "(b: the film frame was taken during a cross-fade). Most were false alarms of the clip's own OCR (handwriting, "
        "noisy reading), e.g. *Brad Pitt*, the cast and screenplay cards of *Romanzo criminale*, *Narrated by Laurence Olivier*.",
        f"- **{len(by_mark['n'])} clip frames show no credit at all** (n): the NAIJAPREY watermark of the Eternal Sunshine "
        "clip encode, props in Se7en, the Psycho title animation mid-transition, the Sky Atlantic ident. They are "
        "Stage III calls the clip pipeline spent on non-credit frames.",
        f"- **{len(missing)} missing** (x): "
        + "; ".join(f"{len(v)} {KIND_EN.get(k, k)}" for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])))
        + (f"; {unchecked} not checked yet." if unchecked else "."), "",
        f"**After the check, {present:,} of {total_clip:,} clip frames ({100 * present / total_clip:.1f}%) are present "
        "among the full-film frames.** The person credits truly lost on the films are concentrated in *Prime Suspect* "
        "(part of the end roll, Director of Photography, Producer) plus two single cards (*Amelie*: editor; *Fight Club*: "
        "the opening *Edward Norton* card, whose name is still in the end-credit cast list).", "",
        "### Missing clip frames", "", "\n".join(miss_table), "",
        "### Per title", "", "\n".join(table), "",
        "Lists: `results/D_review_present_in_film.csv`, `results/D_review_missing_in_film.csv`, "
        "`results/D_review_no_credit_in_clip.csv`; full review with the verdicts in `results/D_likely_missing_review.csv` "
        "(edit `controllo` and rerun `section_D_frame_overlap.py` then this script to update the numbers). Limit: a credit "
        "marked missing could still sit in a film frame whose OCR is completely unreadable. This compares frames; the "
        "credits themselves are compared with the gold set in the next section.", "",
    ]


def credits_section() -> list[str]:
    """Stage III/IV on the full films vs the clips, against the human gold set (20 products)."""
    from metrics_lib import DATASETS, GOLD_MAIN, METRIC_COLUMNS, MODES, evaluate, md_table, per_product, write_csv

    runs = [("ORIG_SOL_20products", "Clips, first revision (July)"), ("B_CAP150", "Clips, rerun (f9c9bb6)"),
            ("D_FULL", "Full films (f9c9bb6)")]
    rows = [r for name, label in runs for r in evaluate(name, label, GOLD_MAIN)]
    write_csv(rows, HERE / "results" / "D_credits_vs_gold.csv")
    per = {label: per_product(name, GOLD_MAIN) for name, label in runs}
    clip_label, film_label = runs[1][1], runs[2][1]
    per_rows = [{"product": ep,
                 "clips recall": per[clip_label][ep]["recall"], "films recall": per[film_label][ep]["recall"],
                 "clips FP": per[clip_label][ep]["FP"], "films FP": per[film_label][ep]["FP"],
                 "clips F1": per[clip_label][ep]["F1"], "films F1": per[film_label][ep]["F1"]}
                for ep in sorted(per[film_label])]
    write_csv(per_rows, HERE / "results" / "D_credits_per_product.csv")
    md = ["## Credits extracted from the full films vs the gold set", "",
          f"Stage III (GPT Sol Standard) and Stage IV (fuzzy 88) on the {DATASETS['D_FULL']['calls']:,} frames Stage II selected on the full films "
          f"({DATASETS['D_FULL']['cost_usd']:.2f} USD), "
          "exact match against the human gold set of the 20 products, compared with the hand-cut clips: the "
          "first-revision run and the rerun of the clips with the same code as the films (f9c9bb6, OCR en, cap 150). "
          "`cost_usd_uncached` prices every input token at full rate (July read most input from the cache).", ""]
    for mode in MODES:
        md += [f"### {mode}", "", md_table([r for r in rows if r["mode"] == mode], METRIC_COLUMNS), ""]
    md += ["### Per product (No Role): clips rerun vs full films", "", md_table(per_rows, list(per_rows[0])), "",
           "False positives on the films are names the model read outside the credits (signs, documents, newspapers "
           "and other on-screen text in the scenes Stage I kept as candidates) or credits absent from the gold set.", ""]
    return md


def main() -> None:
    info = json.loads((FULL / "run_info.json").read_text(encoding="utf-8"))
    conn = sqlite3.connect(FULL / "db" / "tvcredits_v3.db")
    times = {(ep, step): s for ep, step, s in conn.execute(
        "SELECT episode_id, step, SUM(seconds) FROM Time WHERE step IN ('step1','step2') GROUP BY episode_id, step")}
    rows = []
    for full_ep in sorted(info["videos"]):
        title = full_ep[: -len("_FULL")]
        parts = [title] if (CUT / title).is_dir() else sorted(
            p.name for p in CUT.iterdir() if p.name.startswith(title + "_") and (p / "analysis" / "raw_scenes_cache.json").exists())
        cut = [stage_stats(CUT / p / "analysis") for p in parts]
        full = stage_stats(FULL / "data" / "episodes" / full_ep / "analysis")
        row = {"title": title, "cut_parts": len(parts)}
        for k in ("minutes", "shots", "candidates", "candidate_minutes", "frames"):
            row[f"cut_{k}"] = round(sum(c[k] for c in cut), 1)
            row[f"full_{k}"] = round(full[k], 1)
        row["frames_ratio"] = round(full["frames"] / row["cut_frames"], 2)
        row["full_stage1_h"] = round(times.get((full_ep, "step1"), 0) / 3600, 2)
        row["full_stage2_h"] = round(times.get((full_ep, "step2"), 0) / 3600, 2)
        rows.append(row)
    total = {"title": "**Total**", "cut_parts": sum(r["cut_parts"] for r in rows)}
    for k in rows[0]:
        if k not in total and k != "frames_ratio":
            total[k] = round(sum(r[k] for r in rows), 1)
    total["frames_ratio"] = round(total["full_frames"] / total["cut_frames"], 2)

    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "D_full_films.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:,.1f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v))

    cols = [("title", "Title"), ("cut_parts", "Clips"), ("cut_minutes", "Clip min"), ("full_minutes", "Film min"),
            ("cut_shots", "Clip shots"), ("full_shots", "Film shots"), ("cut_candidates", "Clip cand. scenes"),
            ("full_candidates", "Film cand. scenes"), ("cut_candidate_minutes", "Clip cand. min"),
            ("full_candidate_minutes", "Film cand. min"), ("cut_frames", "Clip frames"), ("full_frames", "Film frames"),
            ("frames_ratio", "Frames film/clip"), ("full_stage1_h", "Film Stage I h"), ("full_stage2_h", "Film Stage II h")]
    table = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"]
    for r in rows + [total]:
        table.append("| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |")

    redo = info.get("redone_episodes", [])
    md = [
        "# Section D - full-length films: Stage I and Stage II", "",
        "**Clips**: hand-cut credit clips of the paper (Opening + End summed where there are two), Stage I/II "
        "of 2025-12-04 (commit 00f0cbc5, PaddleOCR `lang=en`); their frames are the ones Stage III used in "
        "July 2026. **Films**: the 20 full-length files (`runs/FULL_pipeline`, commit f9c9bb6, `lang=en`), "
        "Stage I with \"Run on the whole episode\" and no manual deselection of candidate scenes, Stage II with "
        "cap 150. Stage III/IV were run on the films afterwards ("
        f"{info['submitted_frames_total']:,} calls).", "",
        "Shots = scenes found by scene detection; candidate scenes = scenes kept by Stage I (text found by OCR); "
        "frames = frames selected by Stage II (= Stage III calls). Film Stage I/II hours are machine time "
        "(Stage I partly ran 3 films in parallel on one GPU).", "",
        "\n".join(table), "",
        "## Notes", "",
        f"- Film Stage I/II logs were checked for OCR failures (`PaddleOCR failed`): 0 in the final logs of all 20 films.",
    ]
    for r in redo:
        md.append(f"- {', '.join(r['episodes'])}: Stage I and II redone on {r['redone_at'][:10]} - {r['reason']}. "
                  "The failed attempt's logs are kept as `*.failed_<timestamp>.log`.")
    for r in info.get("replaced_inputs", []):
        names = ", ".join(v["episode"] for v in r["videos"])
        md.append(f"- {names}: {r['reason']} (old and new MD5 in `run_info.json`, `replaced_inputs`); everything Stage I/II "
                  "had produced from the wrong files was deleted before the rerun.")
    md += ["- Stage II keeps every frame of a candidate scene in RAM; the longest film candidate scenes needed up to "
           "~38 GB (Apocalypse Now) and ran through the page file.",
           "- Segment boundaries vs the annotated credit boundaries (precision/recall/IoU) need the annotations; "
           "the film candidate scenes are in `runs/FULL_pipeline/data/episodes/<film>/analysis/initial_scene_analysis.json` "
           "and the selected frames with timecodes in `runs/FULL_pipeline/run_output/stage2_frames_timecodes.csv`.", ""]
    md += frame_overlap_section()
    md += credits_section()
    (HERE / "D_full_films_stage1_stage2.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
