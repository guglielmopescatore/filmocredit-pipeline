#!/usr/bin/env python3
"""
Section A - naive interval curve (RIEPILOGO_run_v2.md, block A).

Does a sparser uniform sampling cost less than the pipeline and perform as well?
Compares, on the 20 main-corpus products against the human gold set:
  the pipeline of the first revision (ORIG_SOL_20products) and the naive runs at
  0.8 s (first revision), 2.4 s and 4.8 s (k = 3 and 6 subsamples of the 0.8 s frames).

Outputs results/A_naive.csv, results/A_naive_per_product.csv, results/A_naive.md.
Usage: python analysis_2nd_revision/section_A_naive.py
"""

from metrics_lib import GOLD_MAIN, METRIC_COLUMNS, MODES, RESULTS, evaluate, md_table, per_product, write_csv

RUNS = [
    ("ORIG_SOL_20products", "Pipeline (first revision)"),
    ("A_NAIVE_0.8s", "Naive 0.8 s"),
    ("A_NAIVE_2.4s", "Naive 2.4 s"),
    ("A_NAIVE_4.8s", "Naive 4.8 s"),
]


def main() -> None:
    rows = [r for name, label in RUNS for r in evaluate(name, label, GOLD_MAIN)]
    for r in rows:
        r["cost_per_TP_uncached"] = round(r["cost_usd_uncached"] / r["TP"], 4) if r["TP"] else None
    write_csv(rows, RESULTS / "A_naive.csv")

    per = {label: per_product(name, GOLD_MAIN) for name, label in RUNS}
    episodes = sorted(next(iter(per.values())))
    per_rows = [{"product": ep, **{f"{label} recall": per[label][ep]["recall"] for _, label in RUNS}} for ep in episodes]
    write_csv(per_rows, RESULTS / "A_naive_per_product.csv")

    cols = METRIC_COLUMNS + ["cost_per_TP_uncached"]
    md = ["# Section A - naive interval curve", "",
          "Exact match against the human gold set, 20 main-corpus products. `cost_usd` is what each run "
          "cost with the Azure cache behaviour of its time (July: in-memory cache reads; September: "
          "almost none); `cost_usd_uncached` prices every input token at full rate, so the runs are "
          "comparable.", ""]
    for mode in MODES:
        md += [f"## {mode}", "", md_table([r for r in rows if r["mode"] == mode], cols), ""]
    md += ["## Recall per product (No Role)", "", md_table(per_rows, list(per_rows[0])), ""]
    (RESULTS / "A_naive.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
