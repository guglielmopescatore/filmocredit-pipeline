#!/usr/bin/env python3
"""
Section C - repetitions on the hold-out (RIEPILOGO_run_v2.md, block C).

How much does the result change when the same run is repeated on the same
frames? Five repetitions per model on the 207 July frames of the 5 hold-out
products, against the hold-out gold set:
  - GPT Sol Standard: the first-revision run (ORIG_SOL_holdout5) + C_SOL_r1..r5;
  - Gemma 4 12B: C_GEMMA_r1..r5 (there is no first-revision Gemma run on the
    hold-out: the July Gemma run covers only the 20 main products).
Dispersion = mean, standard deviation, min, max, range of P/R/F1 over the five
repetitions, plus the mean pairwise Jaccard similarity of the predicted
credits (1.0 = every repetition returns exactly the same set).

Outputs results/C_repetitions.csv, results/C_dispersion.csv, results/C_repetitions.md.
Usage: python analysis_2nd_revision/section_C_repetitions.py
"""

from itertools import combinations

from metrics_lib import (GOLD_HOLDOUT, METRIC_COLUMNS, MODES, RESULTS, describe, evaluate, md_table, pred_keys,
                         write_csv)

MODELS = {
    "GPT Sol Standard": [f"C_SOL_r{n}" for n in range(1, 6)],
    "Gemma 4 12B": [f"C_GEMMA_r{n}" for n in range(1, 6)],
}


def main() -> None:
    rows = evaluate("ORIG_SOL_holdout5", "GPT Sol Standard - first revision", GOLD_HOLDOUT)
    for model, names in MODELS.items():
        for i, name in enumerate(names, 1):
            rows += evaluate(name, f"{model} - r{i}", GOLD_HOLDOUT)
    write_csv(rows, RESULTS / "C_repetitions.csv")

    disp = []
    for model, names in MODELS.items():
        for mode in MODES:
            reps = [r for r in rows if r["dataset"] in names and r["mode"] == mode]
            keys = [pred_keys(n, mode) for n in names]
            jaccard = [len(a & b) / len(a | b) for a, b in combinations(keys, 2)]
            row = {"model": model, "mode": mode, "repetitions": len(reps)}
            for metric in ("precision", "recall", "F1", "TP", "FP", "FN"):
                for stat, value in describe([r[metric] for r in reps]).items():
                    row[f"{metric}_{stat}"] = value
            row["pairwise_jaccard_mean"] = round(sum(jaccard) / len(jaccard), 5)
            row["pairwise_jaccard_min"] = round(min(jaccard), 5)
            disp.append(row)
    write_csv(disp, RESULTS / "C_dispersion.csv")

    md = ["# Section C - repetitions on the hold-out", "",
          "Same 207 frames (July hold-out Stage II) in every run, exact match against the hold-out gold set. "
          "Gemma runs locally with temperature 0; there is no first-revision Gemma run on the hold-out.", ""]
    for mode in MODES:
        md += [f"## Runs - {mode}", "", md_table([r for r in rows if r["mode"] == mode], METRIC_COLUMNS), ""]
    short = ["model", "mode", "repetitions", "precision_mean", "precision_sd", "recall_mean", "recall_sd",
             "F1_mean", "F1_sd", "F1_min", "F1_max", "F1_range", "pairwise_jaccard_mean", "pairwise_jaccard_min"]
    md += ["## Dispersion over the 5 repetitions", "", md_table(disp, short), "",
           "Full statistics (TP/FP/FN included) in results/C_dispersion.csv.", ""]
    (RESULTS / "C_repetitions.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
