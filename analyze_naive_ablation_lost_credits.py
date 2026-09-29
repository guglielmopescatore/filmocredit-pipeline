"""
Distribuzione per titolo dei crediti persi dalla pipeline completa rispetto
all'ablation naive (estrazione a intervallo fisso, senza scene detection ne'
screening/deduplica OCR), sui 20 titoli del gold standard.

CONFRONTO
  Pipeline completa : exports/FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv
  Ablation naive    : exports/NAIVE_FUZZY88_GPT_SOL_STANDARD_20products_tvcredits_v3.csv
  Riferimento       : gold umano 20 prodotti (compare_llm_human_metrics.HUMAN_PATH)

MATCHING
  Exact match sulle stesse chiavi normalizzate usate da compare_llm_human_metrics.py:
  nessuna soglia fuzzy e' coinvolta (il "FUZZY88" nei nomi file indica solo la
  soglia con cui e' stato prodotto l'export, non il criterio di confronto qui).
  Due granularita': No Role (episodio, nome) e With Role (episodio, nome, role_group).
  Deduplica per prodotto identica a quella del confronto principale.

QUANTITA' RIPORTATE (per titolo)
  gold                  voci di riferimento del titolo
  tp_full / tp_naive    voci del gold recuperate da ciascuna configurazione
  lost_by_full          TP_naive \\ TP_full: crediti che il naive trova e la pipeline
                        completa NO. E' il costo in recall della selezione dei frame
                        (la quantita' aggregata a 248 in modalita' No Role).
  only_full             TP_full \\ TP_naive: crediti che solo la pipeline completa trova
  missed_by_both        voci del gold che nessuna delle due configurazioni recupera
  fp_full / fp_naive    crediti emessi che non trovano riscontro nel gold
  recall_*, delta_recall, share_lost

OUTPUT
  exports/naive_ablation_lost_by_title[_with_role].csv   aggregato per titolo
  exports/naive_ablation_lost_detail[_with_role].csv     un record per credito perso

Usage:
    python analyze_naive_ablation_lost_credits.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import compare_llm_human_metrics as cllm

ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = ROOT / "exports"
FULL_PATH = EXPORTS_DIR / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv"
NAIVE_PATH = EXPORTS_DIR / "NAIVE_FUZZY88_GPT_SOL_STANDARD_20products_tvcredits_v3.csv"


def keys_by_mode(triples, with_role: bool) -> set:
    """Chiavi distinte con la stessa semantica di deduplica del confronto principale:
    (episodio, nome) oppure (episodio, nome, role_group), ogni chiave vale 1."""
    no_role, wr = cllm.counters_from_triples(triples)
    return set(wr if with_role else no_role)


def scope_episodes(gold_triples, full_episodes, naive_episodes) -> set:
    """I 20 titoli del gold: presenti nel gold e in entrambi gli export, esclusi i
    5 prodotti di validazione che non fanno parte del gold set principale."""
    gold_episodes = {t[0] for t in gold_triples}
    eps = gold_episodes & full_episodes & naive_episodes
    return eps - cllm.VALIDATION5_EPISODES


def analyse(gold_keys, full_keys, naive_keys, episodes):
    tp_full = gold_keys & full_keys
    tp_naive = gold_keys & naive_keys
    lost_by_full = tp_naive - tp_full
    only_full = tp_full - tp_naive
    missed_both = gold_keys - full_keys - naive_keys
    fp_full = full_keys - gold_keys
    fp_naive = naive_keys - gold_keys

    def per_ep(keys) -> Counter:
        return Counter(k[0] for k in keys)

    c_gold, c_tpf, c_tpn = per_ep(gold_keys), per_ep(tp_full), per_ep(tp_naive)
    c_lost, c_only, c_both = per_ep(lost_by_full), per_ep(only_full), per_ep(missed_both)
    c_fpf, c_fpn = per_ep(fp_full), per_ep(fp_naive)

    total_lost = len(lost_by_full)
    rows = []
    for ep in sorted(episodes):
        gold_n = c_gold[ep]
        lost = c_lost[ep]
        r_full = c_tpf[ep] / gold_n if gold_n else 0.0
        r_naive = c_tpn[ep] / gold_n if gold_n else 0.0
        rows.append({
            "episode": ep,
            "gold": gold_n,
            "tp_full": c_tpf[ep],
            "tp_naive": c_tpn[ep],
            "lost_by_full": lost,
            "only_full": c_only[ep],
            "missed_by_both": c_both[ep],
            "fp_full": c_fpf[ep],
            "fp_naive": c_fpn[ep],
            "recall_full": r_full,
            "recall_naive": r_naive,
            "delta_recall": r_naive - r_full,
            "lost_rate_on_gold": lost / gold_n if gold_n else 0.0,
            "share_of_lost": lost / total_lost if total_lost else 0.0,
        })
    rows.sort(key=lambda r: (-r["lost_by_full"], r["episode"]))
    return rows, lost_by_full, {
        "gold": len(gold_keys), "tp_full": len(tp_full), "tp_naive": len(tp_naive),
        "lost_by_full": total_lost, "only_full": len(only_full),
        "missed_by_both": len(missed_both), "fp_full": len(fp_full), "fp_naive": len(fp_naive),
    }


def write_by_title(rows, totals, out_path: Path):
    header = ["episode", "gold", "tp_full", "tp_naive", "lost_by_full", "only_full",
              "missed_by_both", "fp_full", "fp_naive", "recall_full", "recall_naive",
              "delta_recall", "lost_rate_on_gold", "share_of_lost"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["episode"], r["gold"], r["tp_full"], r["tp_naive"], r["lost_by_full"],
                        r["only_full"], r["missed_by_both"], r["fp_full"], r["fp_naive"],
                        f"{r['recall_full']:.5f}", f"{r['recall_naive']:.5f}",
                        f"{r['delta_recall']:+.5f}", f"{r['lost_rate_on_gold']:.5f}",
                        f"{r['share_of_lost']:.5f}"])
        w.writerow(["TOTAL", totals["gold"], totals["tp_full"], totals["tp_naive"],
                    totals["lost_by_full"], totals["only_full"], totals["missed_by_both"],
                    totals["fp_full"], totals["fp_naive"],
                    f"{totals['tp_full']/totals['gold']:.5f}",
                    f"{totals['tp_naive']/totals['gold']:.5f}",
                    f"{(totals['tp_naive']-totals['tp_full'])/totals['gold']:+.5f}", "", "1.00000"])


def write_detail(lost_keys, with_role: bool, out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "name", "role_group"] if with_role else ["episode", "name"])
        for key in sorted(lost_keys):
            w.writerow(list(key))


def print_table(rows, totals, title: str):
    header = ["Title", "Gold", "TP full", "TP naive", "Lost", "%gold", "%lost", "R full", "R naive", "FP naive"]
    body = [[r["episode"][:38], str(r["gold"]), str(r["tp_full"]), str(r["tp_naive"]),
             str(r["lost_by_full"]), f"{100*r['lost_rate_on_gold']:.1f}",
             f"{100*r['share_of_lost']:.1f}", f"{r['recall_full']:.3f}",
             f"{r['recall_naive']:.3f}", str(r["fp_naive"])] for r in rows]
    body.append(["TOTAL", str(totals["gold"]), str(totals["tp_full"]), str(totals["tp_naive"]),
                 str(totals["lost_by_full"]),
                 f"{100*totals['lost_by_full']/totals['gold']:.1f}", "100.0",
                 f"{totals['tp_full']/totals['gold']:.3f}",
                 f"{totals['tp_naive']/totals['gold']:.3f}", str(totals["fp_naive"])])
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(header)]

    def fmt(cells):
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for r in body[:-1]:
        print(fmt(r))
    print("  ".join("-" * w for w in widths))
    print(fmt(body[-1]))


def main():
    for path in (FULL_PATH, NAIVE_PATH):
        if not path.exists():
            print(f"[ERRORE] File non trovato: {path}", file=sys.stderr)
            sys.exit(1)

    gold_triples = cllm.load_gold_triples(cllm.HUMAN_PATH)
    full_triples, full_eps = cllm.load_pred(FULL_PATH)
    naive_triples, naive_eps = cllm.load_pred(NAIVE_PATH)

    episodes = scope_episodes(gold_triples, full_eps, naive_eps)
    print(f"Titoli valutati: {len(episodes)}")
    dropped = ({t[0] for t in gold_triples} - cllm.VALIDATION5_EPISODES) - episodes
    if dropped:
        print(f"[AVVISO] titoli del gold assenti da almeno un export, esclusi: {sorted(dropped)}")

    g = {t for t in gold_triples if t[0] in episodes}
    fu = {t for t in full_triples if t[0] in episodes}
    na = {t for t in naive_triples if t[0] in episodes}

    for with_role in (False, True):
        mode = "With Role" if with_role else "No Role"
        suffix = "_with_role" if with_role else ""
        rows, lost_keys, totals = analyse(
            keys_by_mode(g, with_role), keys_by_mode(fu, with_role),
            keys_by_mode(na, with_role), episodes,
        )
        print_table(rows, totals,
                    f"CREDITI PERSI DALLA PIPELINE COMPLETA vs ABLATION NAIVE - {mode} (exact match)")
        by_title = EXPORTS_DIR / f"naive_ablation_lost_by_title{suffix}.csv"
        detail = EXPORTS_DIR / f"naive_ablation_lost_detail{suffix}.csv"
        write_by_title(rows, totals, by_title)
        write_detail(lost_keys, with_role, detail)
        print(f"\nAggregato per titolo : {by_title}")
        print(f"Dettaglio crediti    : {detail}")

        nonzero = [r for r in rows if r["lost_by_full"]]
        if nonzero:
            top3 = sum(r["lost_by_full"] for r in nonzero[:3])
            print(f"Concentrazione: {len(nonzero)}/{len(rows)} titoli con almeno un credito perso; "
                  f"i primi 3 titoli assorbono {top3}/{totals['lost_by_full']} "
                  f"({100*top3/totals['lost_by_full']:.1f}%) delle perdite.")


if __name__ == "__main__":
    main()
