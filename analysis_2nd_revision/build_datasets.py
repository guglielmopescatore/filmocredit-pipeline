#!/usr/bin/env python3
"""
Build the datasets of the second revision in analysis_2nd_revision/data/.

Every dataset is one credits CSV (same columns as the usual FUZZY88 exports)
assembled from one or more runs, plus an entry in data/datasets.json with its
provenance (source CSV and DB of every part, the products taken from each) and
the VLM calls it cost, counted from the raw responses of the source DBs:
  calls, input/cached/output tokens, cost_usd (actual, with the cache reads of
  that run) and cost_usd_uncached (every input token at full price, so runs
  made with different Azure cache behaviour stay comparable).

Datasets:
  ORIG_SOL_20products / ORIG_SOL_holdout5   first-revision GPT Sol Standard run
                                            (25-products DB) split main / hold-out
  A_NAIVE_0.8s / 2.4s / 4.0s / 4.8s              naive runs, 20 products
  B_CAP150 / CAP100 / CAP75                 pipeline with scroll cap, 20 products:
                                            14 roll titles from PIPE_cap<N>, 6 titles
                                            from PIPE_noscroll_cap150, except
                                            Yes,_Prime_Minister_1x8 at cap 100/75,
                                            taken from PIPE_noscroll_cap<N>
  C_SOL_r1..r5 / C_GEMMA_r1..r5             repetitions on the 5 hold-out products

Usage (from the repository root):
    python analysis_2nd_revision/build_datasets.py
"""

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prepare_naive_run import PRICING_USD_PER_1M  # noqa: E402
from run_headless import call_metrics  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
RUNS = ROOT / "runs"
EXPORTS = ROOT / "exports"
DB = ROOT / "db"

HOLDOUT = {"3_Percent_S01E06", "Blue_Eye_Samurai_S01E01", "Honeyland", "Persepolis", "Wild_Strawberries"}
YPM = "Yes,_Prime_Minister_1x8"
NOSCROLL = {"8_e_mezzo", "Apocalypse_Now", "Chernobyl_S01E01", "The_World_At_War_S01E03", "Twin_Peaks_1x3", YPM}

ORIG_CSV = EXPORTS / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv"
ORIG_DB = DB / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.db"


def run_part(run_id: str, products: set[str] | None = None) -> dict:
    run = RUNS / run_id
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    return {"csv": run / "run_output" / info["expected_export_csv"], "db": run / "db" / "tvcredits_v3.db",
            "products": products, "run_id": run_id}


def file_part(csv_path: Path, db_path: Path, products: set[str] | None = None) -> dict:
    return {"csv": csv_path, "db": db_path, "products": products, "run_id": None}


# Full-film episode ids are <product>_FULL; El desorden's full file was named without the episode code.
FULL_RENAME = {"El_desorden_que_dejas": "El_desorden_que_dejas_S01E03"}


def product_of(episode_id: str) -> str:
    # DBs keep the _End/_Opening folders; the CSV exports merge them into one product.
    # Full-film runs (block D) use <product>_FULL: map them back to the product name.
    ep = re.sub(r"_(End|Opening)$", "", episode_id or "")
    if ep.endswith("_FULL"):
        ep = ep[: -len("_FULL")]
        ep = FULL_RENAME.get(ep, ep)
    return ep


def read_rows(part: dict) -> tuple[list[str], list[dict]]:
    with open(part["csv"], encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            r["episode_id"] = product_of(r["episode_id"]) if r["episode_id"].endswith("_FULL") else r["episode_id"]
            if part["products"] is None or r["episode_id"] in part["products"]:
                rows.append(r)
        return reader.fieldnames, rows


def call_stats(part: dict, products: set[str]) -> dict:
    totals = {"calls": 0, "input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0,
              "cost_usd": 0.0, "cost_usd_uncached": 0.0}
    conn = sqlite3.connect(part["db"])
    for episode_id, raw in conn.execute("SELECT episode_id, raw_response FROM raw_response_llm_call"):
        if product_of(episode_id) not in products:
            continue
        response = json.loads(raw)
        usage = response.get("usage") or {}
        if "prompt_tokens" in usage:  # llama.cpp (Gemma, local): no cost
            m = {"input_tokens": usage.get("prompt_tokens", 0), "cache_read_tokens": 0,
                 "output_tokens": usage.get("completion_tokens", 0), "cost_usd": 0.0}
            uncached = 0.0
        else:
            m = call_metrics(response, PRICING_USD_PER_1M)
            uncached = (m["input_tokens"] * PRICING_USD_PER_1M["input"]
                        + m["output_tokens"] * PRICING_USD_PER_1M["output"]) / 1_000_000
        totals["calls"] += 1
        for k in ("input_tokens", "cache_read_tokens", "output_tokens", "cost_usd"):
            totals[k] += m[k]
        totals["cost_usd_uncached"] += uncached
    conn.close()
    totals["cost_usd"] = round(totals["cost_usd"], 4)
    totals["cost_usd_uncached"] = round(totals["cost_usd_uncached"], 4)
    return totals


def build(name: str, parts: list[dict], expected_products: int, note: str = "") -> dict:
    fieldnames, rows, provenance = None, [], []
    totals = {}
    for part in parts:
        fields, part_rows = read_rows(part)
        fieldnames = fieldnames or fields
        products = part["products"] or {r["episode_id"] for r in part_rows}
        rows += part_rows
        stats = call_stats(part, products)
        for k, v in stats.items():
            totals[k] = round(totals.get(k, 0) + v, 4)
        provenance.append({"run_id": part["run_id"], "csv": str(part["csv"].relative_to(ROOT)),
                           "db": str(part["db"].relative_to(ROOT)), "products": sorted(products), **stats})
    products = sorted({r["episode_id"] for r in rows})
    if len(products) != expected_products:
        sys.exit(f"{name}: {len(products)} products, expected {expected_products}: {products}")
    out_csv = OUT / f"{name}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(products)} products, {len(rows)} rows, {totals['calls']} calls, {totals['cost_usd']} USD")
    return {"csv": out_csv.name, "products": products, "rows": len(rows), "note": note,
            "parts": provenance, **totals}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    main20 = None  # products of the main corpus, from the original run
    with open(ORIG_CSV, encoding="utf-8") as f:
        main20 = {r["episode_id"] for r in csv.DictReader(f)} - HOLDOUT
    scroll14 = main20 - NOSCROLL

    datasets = {
        "ORIG_SOL_20products": build("ORIG_SOL_20products", [file_part(ORIG_CSV, ORIG_DB, main20)], 20,
                                     "first revision, GPT Sol Standard, pipeline (Stage II of 2025-12-04)"),
        "ORIG_SOL_holdout5": build("ORIG_SOL_holdout5", [file_part(ORIG_CSV, ORIG_DB, HOLDOUT)], 5,
                                   "first revision, GPT Sol Standard, hold-out (Stage II of July 2026)"),
        "A_NAIVE_0.8s": build("A_NAIVE_0.8s", [file_part(EXPORTS / "NAIVE_FUZZY88_GPT_SOL_STANDARD_20products_tvcredits_v3.csv",
                                                         DB / "NAIVE_FUZZY88_GPT_SOL_STANDARD_20products_tvcredits_v3.db")], 20,
                              "first revision naive run (0.8 s)"),
        "A_NAIVE_2.4s": build("A_NAIVE_2.4s", [run_part("NAIVE_2.4s")], 20, "k = 3 subsample of the 0.8 s frames"),
        "A_NAIVE_4.0s": build("A_NAIVE_4.0s", [run_part("NAIVE_4.0s")], 20, "k = 5 subsample of the 0.8 s frames"),
        "A_NAIVE_4.8s": build("A_NAIVE_4.8s", [run_part("NAIVE_4.8s")], 20, "k = 6 subsample of the 0.8 s frames"),
        "B_CAP150": build("B_CAP150", [run_part("PIPE_cap150"), run_part("PIPE_noscroll_cap150")], 20,
                          "f9c9bb6, OCR en: 14 roll titles + 6 titles without rolls"),
    }
    for cap in (100, 75):
        datasets[f"B_CAP{cap}"] = build(
            f"B_CAP{cap}",
            [run_part(f"PIPE_cap{cap}"), run_part("PIPE_noscroll_cap150", NOSCROLL - {YPM}),
             run_part(f"PIPE_noscroll_cap{cap}")],
            20, f"14 roll titles at cap {cap}; {YPM} (roll found today) at cap {cap}; "
                f"the other 5 titles without rolls from PIPE_noscroll_cap150 (cap-independent)")
    datasets["D_FULL"] = build("D_FULL", [run_part("FULL_pipeline")], 20,
                               "full-length films, Stage I (whole episode) + II (cap 150) + III + IV, f9c9bb6, OCR en")
    for model, prefix in (("SOL", "REP_SOL"), ("GEMMA", "REP_GEMMA")):
        for n in range(1, 6):
            datasets[f"C_{model}_r{n}"] = build(f"C_{model}_r{n}", [run_part(f"{prefix}_r{n}")], 5,
                                               f"repetition {n} on the July hold-out frames (207)")
    assert all(set(d["products"]) == scroll14 | NOSCROLL for k, d in datasets.items() if k.startswith(("A_", "B_", "D_")))
    (OUT / "datasets.json").write_text(json.dumps(datasets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(datasets)} datasets in {OUT}")


if __name__ == "__main__":
    main()
