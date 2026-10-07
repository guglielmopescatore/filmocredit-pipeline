"""
Shared helpers for the second-revision analyses.

The matching itself is the one of compare_llm_human_metrics.py (exact match on
the normalized name, "No Role" and "With Role", persons only, deduplicated per
product, each model evaluated on the products it covers): its functions are
imported, not copied, so these analyses cannot drift from the paper's method.
What is added here: datasets from data/datasets.json (with their VLM calls and
cost), per-product metrics, and CSV/Markdown output.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import compare_llm_human_metrics as cm  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
DATASETS = json.loads((DATA / "datasets.json").read_text(encoding="utf-8"))

GOLD_MAIN = cm.load_gold_triples(cm.HUMAN_PATH)
GOLD_HOLDOUT = cm.load_gold_triples(cm.VALIDATION5_GOLD_PATH)
MODES = ("No Role", "With Role")


def predictions(name: str) -> tuple[set, set]:
    return cm.load_pred(DATA / DATASETS[name]["csv"])


def evaluate(name: str, label: str, gold: set, products: set | None = None) -> list[dict]:
    """Both modes for one dataset; `products` (canonical names) restricts the evaluation."""
    triples, episodes = predictions(name)
    gold_episodes = {t[0] for t in gold}
    if products is not None:
        episodes &= products
        gold_episodes &= products
    rows = []
    for (lab, n, _, mode, tp, fp, fn, p, r, f1, _) in cm.evaluate_model(label, triples, episodes, gold, gold_episodes):
        rows.append({"dataset": name, "label": lab, "mode": mode, "products": n, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(p, 5), "recall": round(r, 5), "F1": round(f1, 5)})
    meta = DATASETS[name]
    for row in rows:
        row.update(calls=meta["calls"], cost_usd=meta["cost_usd"], cost_usd_uncached=meta["cost_usd_uncached"])
    return rows


def per_product(name: str, gold: set, mode: str = "No Role") -> dict[str, dict]:
    """TP/FP/FN/recall/F1 per product, in one mode."""
    triples, episodes = predictions(name)
    out = {}
    for ep in sorted(episodes & {t[0] for t in gold}):
        g_nr, g_wr = cm.counters_from_triples({t for t in gold if t[0] == ep})
        p_nr, p_wr = cm.counters_from_triples({t for t in triples if t[0] == ep})
        g, p = (g_nr, p_nr) if mode == "No Role" else (g_wr, p_wr)
        tp, fp, fn, prec, rec, f1 = cm.compute_metrics(g, p)
        out[ep] = {"TP": tp, "FP": fp, "FN": fn, "recall": round(rec, 4), "F1": round(f1, 4)}
    return out


def pred_keys(name: str, mode: str) -> set:
    triples, _ = predictions(name)
    return {(ep, n) for ep, n, _ in triples} if mode == "No Role" else set(triples)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:,.2f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def describe(values: list[float]) -> dict:
    return {"mean": round(statistics.mean(values), 5), "sd": round(statistics.stdev(values), 5) if len(values) > 1 else 0.0,
            "min": round(min(values), 5), "max": round(max(values), 5), "range": round(max(values) - min(values), 5)}


METRIC_COLUMNS = ["label", "mode", "products", "TP", "FP", "FN", "precision", "recall", "F1", "calls",
                  "cost_usd", "cost_usd_uncached"]
