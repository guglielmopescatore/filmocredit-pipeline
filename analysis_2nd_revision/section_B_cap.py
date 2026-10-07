#!/usr/bin/env python3
"""
Section B - scroll cap (RIEPILOGO_run_v2.md, block B).

Which cap on the frame interval of credit rolls works best against the gold set?
Compares the first-revision pipeline (ORIG_SOL_20products, Stage II of 2025-12-04)
with the rerun at cap 150 / 100 / 75 (commit f9c9bb6, OCR en), on:
  - all 20 main-corpus products;
  - the 15 products the cap can change: the 14 roll titles + Yes, Prime Minister,
    whose roll is detected since 5fec349.
B_CAP150 vs ORIG also shows what changed in Stage II itself between the two
versions (5fec349) at the same cap.

Outputs results/B_cap.csv, results/B_cap_per_product.csv, results/B_cap.md.
Usage: python analysis_2nd_revision/section_B_cap.py
"""

from metrics_lib import (DATASETS, GOLD_MAIN, METRIC_COLUMNS, MODES, RESULTS, cm, evaluate, md_table, per_product,
                         write_csv)

RUNS = [
    ("ORIG_SOL_20products", "Pipeline first revision (cap 150)"),
    ("B_CAP150", "Rerun cap 150"),
    ("B_CAP100", "Rerun cap 100"),
    ("B_CAP75", "Rerun cap 75"),
]
CAP_DEPENDENT = {cm.canon_episode(p) for p in DATASETS["B_CAP100"]["parts"][0]["products"]} | {
    cm.canon_episode("Yes,_Prime_Minister_1x8")}


def main() -> None:
    rows = []
    for scope, products in (("20 products", None), ("15 cap-dependent products", CAP_DEPENDENT)):
        for name, label in RUNS:
            for r in evaluate(name, label, GOLD_MAIN, products):
                rows.append({"scope": scope, **r})
    write_csv(rows, RESULTS / "B_cap.csv")

    per = {label: per_product(name, GOLD_MAIN) for name, label in RUNS}
    per_rows = [{"product": ep, **{f"{label} recall": per[label][ep]["recall"] for _, label in RUNS}}
                for ep in sorted(next(iter(per.values())))]
    write_csv(per_rows, RESULTS / "B_cap_per_product.csv")

    md = ["# Section B - scroll cap", "",
          "Exact match against the human gold set. Calls and cost refer to the whole dataset (20 products); "
          "`cost_usd_uncached` prices every input token at full rate. The rerun is f9c9bb6 with PaddleOCR "
          "lang=en; it differs from the first revision in Stage II only by 5fec349 (scroll flow percentile).", ""]
    for scope in ("20 products", "15 cap-dependent products"):
        for mode in MODES:
            sub = [r for r in rows if r["scope"] == scope and r["mode"] == mode]
            md += [f"## {scope} - {mode}", "", md_table(sub, METRIC_COLUMNS), ""]
    md += ["## Recall per product (No Role)", "", md_table(per_rows, list(per_rows[0])), ""]
    (RESULTS / "B_cap.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
