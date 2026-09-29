"""
Appendice B: breakdown per prodotto dei risultati di estrazione dei crediti
per GPT Sol Standard, in exact match (nessuna soglia fuzzy coinvolta).

Quattro tabelle separate:
  B1. Gold standard 20 prodotti  - No Role   (chiave: prodotto, nome)
  B2. Gold standard 20 prodotti  - With Role (chiave: prodotto, nome, role_group)
  B3. Hold-out corpus 5 prodotti - No Role
  B4. Hold-out corpus 5 prodotti - With Role

METRICHE
  Per ciascun prodotto, TP/FP/FN sono calcolati sulle sole chiavi di quel prodotto,
  con la stessa normalizzazione e la stessa deduplica del confronto principale
  (compare_llm_human_metrics.py), cosi' che la media dei prodotti sia leggibile
  accanto alle Tabelle 2 e 3 del paper.

  delta_f1_vs_mean = f1 del prodotto - media macro degli f1 della tabella.
  In coda a ogni tabella sono riportate la media macro (media degli f1 per prodotto)
  e la micro (metriche aggregate su tutti i crediti, cioe' il valore del paper).

Usage:
    python analyze_per_product_breakdown.py
"""

import csv
import sys
from pathlib import Path

import compare_llm_human_metrics as cllm

ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = ROOT / "exports"
PRED_PATH = EXPORTS_DIR / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv"
MD_OUT = ROOT / "file_per_analisi" / "appendice_B_per_prodotto_gpt_sol_standard.md"

# Nomi dei prodotti come compaiono nel paper (la chiave canonica e' minuscola e
# normalizzata: qui si ripristina la forma editoriale del titolo).
DISPLAY_NAMES = {
    "la piovra s01e02": "La piovra S01E02",
    "twin peaks s01e03": "Twin Peaks S01E03",
    "chernobyl s01e01": "Chernobyl S01E01",
    "planet earth s01e10": "Planet Earth S01E10",
    "psycho": "Psycho",
    "maigret s03e01": "Maigret S03E01",
    "eternal sunshine of the spotless mind": "Eternal Sunshine of the Spotless Mind",
    "the world at war s01e03": "The World At War S01E03",
    "8 e mezzo": "8 e mezzo",
    "el desorden que dejas s01e03": "El desorden que dejas S01E03",
    "se7en": "Se7en",
    "hill street blues s01e13": "Hill Street Blues S01E13",
    "romanzo criminale s01e01": "Romanzo criminale S01E01",
    "dark s01e09": "Dark S01E09",
    "amelie": "Amelie",
    "fight club": "Fight Club",
    "apocalypse now": "Apocalypse Now",
    "prime suspect s01e01": "Prime Suspect S01E01",
    "la grande bellezza": "La grande bellezza",
    "yes prime minister s01e08": "Yes, Prime Minister S01E08",
    "3 percent s01e06": "3% S01E06",
    "blue eye samurai s01e01": "Blue Eye Samurai S01E01",
    "honeyland": "Honeyland",
    "persepolis": "Persepolis",
    "wild strawberries": "Wild Strawberries",
}


def display(ep: str) -> str:
    return DISPLAY_NAMES.get(ep, ep)


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def keys_by_mode(triples, with_role: bool) -> set:
    """Stessa semantica di deduplica del confronto principale: ogni chiave vale 1."""
    no_role, wr = cllm.counters_from_triples(triples)
    return set(wr if with_role else no_role)


def build_rows(gold_triples, pred_triples, episodes, with_role: bool):
    gold_keys = keys_by_mode({t for t in gold_triples if t[0] in episodes}, with_role)
    pred_keys = keys_by_mode({t for t in pred_triples if t[0] in episodes}, with_role)

    rows = []
    for ep in episodes:
        g = {k for k in gold_keys if k[0] == ep}
        p = {k for k in pred_keys if k[0] == ep}
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        prec, rec, f1 = prf(tp, fp, fn)
        rows.append({"episode": ep, "product": display(ep), "gold": len(g),
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})

    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    for r in rows:
        r["delta_f1_vs_mean"] = r["f1"] - macro_f1
    rows.sort(key=lambda r: (-r["delta_f1_vs_mean"], r["product"]))

    tot_tp = sum(r["tp"] for r in rows)
    tot_fp = sum(r["fp"] for r in rows)
    tot_fn = sum(r["fn"] for r in rows)
    micro_p, micro_r, micro_f1 = prf(tot_tp, tot_fp, tot_fn)
    summary = {
        "macro_precision": sum(r["precision"] for r in rows) / len(rows) if rows else 0.0,
        "macro_recall": sum(r["recall"] for r in rows) / len(rows) if rows else 0.0,
        "macro_f1": macro_f1,
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "gold": sum(r["gold"] for r in rows), "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
    }
    return rows, summary


def write_csv(rows, summary, out_path: Path):
    header = ["product", "gold", "tp", "fp", "fn", "precision", "recall", "f1", "delta_f1_vs_mean"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["product"], r["gold"], r["tp"], r["fp"], r["fn"],
                        f"{r['precision']:.5f}", f"{r['recall']:.5f}", f"{r['f1']:.5f}",
                        f"{r['delta_f1_vs_mean']:+.5f}"])
        w.writerow(["MACRO AVERAGE", summary["gold"], "", "", "",
                    f"{summary['macro_precision']:.5f}", f"{summary['macro_recall']:.5f}",
                    f"{summary['macro_f1']:.5f}", "+0.00000"])
        w.writerow(["MICRO (aggregate)", summary["gold"], summary["tp"], summary["fp"], summary["fn"],
                    f"{summary['micro_precision']:.5f}", f"{summary['micro_recall']:.5f}",
                    f"{summary['micro_f1']:.5f}", ""])


def markdown_table(rows, summary) -> str:
    out = ["| product | gold | precision | recall | f1 | delta_f1_vs_mean |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['product']} | {r['gold']} | {r['precision']:.5f} | {r['recall']:.5f} | "
                   f"{r['f1']:.5f} | {r['delta_f1_vs_mean']:+.5f} |")
    out.append(f"| **Macro average** | **{summary['gold']}** | **{summary['macro_precision']:.5f}** | "
               f"**{summary['macro_recall']:.5f}** | **{summary['macro_f1']:.5f}** | — |")
    out.append(f"| **Micro (aggregate)** | **{summary['gold']}** | **{summary['micro_precision']:.5f}** | "
               f"**{summary['micro_recall']:.5f}** | **{summary['micro_f1']:.5f}** | — |")
    return "\n".join(out)


def print_table(title: str, rows, summary):
    print(f"\n{'=' * 85}\n{title}\n{'=' * 85}")
    print(f"{'product':38} {'gold':>5} {'prec':>8} {'recall':>8} {'f1':>8} {'delta':>9}")
    print("-" * 85)
    for r in rows:
        print(f"{r['product'][:38]:38} {r['gold']:5d} {r['precision']:8.5f} {r['recall']:8.5f} "
              f"{r['f1']:8.5f} {r['delta_f1_vs_mean']:+9.5f}")
    print("-" * 85)
    print(f"{'MACRO AVERAGE':38} {summary['gold']:5d} {summary['macro_precision']:8.5f} "
          f"{summary['macro_recall']:8.5f} {summary['macro_f1']:8.5f}")
    print(f"{'MICRO (aggregate)':38} {summary['gold']:5d} {summary['micro_precision']:8.5f} "
          f"{summary['micro_recall']:8.5f} {summary['micro_f1']:8.5f}")


def main():
    if not PRED_PATH.exists():
        print(f"[ERRORE] Export non trovato: {PRED_PATH}", file=sys.stderr)
        sys.exit(1)

    pred_triples, pred_episodes = cllm.load_pred(PRED_PATH)

    gold_20 = cllm.load_gold_triples(cllm.HUMAN_PATH)
    eps_20 = sorted(({t[0] for t in gold_20} & pred_episodes) - cllm.VALIDATION5_EPISODES)

    gold_5 = cllm.load_gold_triples(cllm.VALIDATION5_GOLD_PATH)
    eps_5 = sorted({t[0] for t in gold_5} & pred_episodes & cllm.VALIDATION5_EPISODES)

    print(f"Export: {PRED_PATH.name}")
    print(f"Gold 20 prodotti: {len(eps_20)} titoli | hold-out: {len(eps_5)} titoli")

    specs = [
        ("B1", "Gold standard 20 prodotti - No Role", gold_20, eps_20, False,
         "per_product_breakdown_20_products.csv"),
        ("B2", "Gold standard 20 prodotti - With Role", gold_20, eps_20, True,
         "per_product_breakdown_20_products_with_role.csv"),
        ("B3", "Hold-out corpus 5 prodotti - No Role", gold_5, eps_5, False,
         "per_product_breakdown_holdout5.csv"),
        ("B4", "Hold-out corpus 5 prodotti - With Role", gold_5, eps_5, True,
         "per_product_breakdown_holdout5_with_role.csv"),
    ]

    md = ["# APPENDIX B - Results for each product", "",
          "GPT Sol Standard, **exact match**: nessuna soglia fuzzy e' coinvolta nel confronto",
          "(il token `FUZZY88` nel nome dell'export indica solo la soglia con cui e' stato",
          "prodotto, non il criterio di confronto applicato qui).", "",
          f"Export: `{PRED_PATH.name}`  |  Generato da `analyze_per_product_breakdown.py`", "",
          "**Colonne.** `gold` = numero di voci di riferimento del prodotto (crediti persona",
          "distinti dopo normalizzazione e deduplica), cioe' il denominatore del richiamo;",
          "`delta_f1_vs_mean` = f1 del prodotto meno la media macro degli f1 della tabella.", "",
          "## Micro e macro average", "",
          "Le due righe di sintesi in coda a ogni tabella rispondono a domande diverse e in",
          "generale non coincidono.", "",
          "**Micro (aggregate)** - i crediti di tutti i prodotti confluiscono in un unico",
          "insieme: TP, FP e FN si contano una volta sola sull'intero corpus e da quei totali",
          "si calcolano precisione, richiamo e F1. Ogni *credito* pesa uguale, quindi i",
          "prodotti con titoli di coda lunghi dominano il risultato (in Table B1, `Chernobyl",
          "S01E01` con 551 voci pesa oltre quaranta volte `The World At War S01E03` che ne ha",
          "13). Risponde a: *dato un credito qualsiasi del corpus, con che accuratezza il",
          "sistema lo tratta?* E' il valore riportato nelle Tabelle 2 e 3 del corpo del paper.", "",
          "**Macro average** - precisione, richiamo e F1 sono calcolati separatamente su ogni",
          "prodotto e poi mediati aritmeticamente senza pesi. Ogni *prodotto* pesa uguale a",
          "prescindere dalla sua dimensione. Risponde a: *dato un titolo qualsiasi, con che",
          "accuratezza il sistema lo tratta?* - la domanda pertinente quando interessa la",
          "robustezza da un titolo all'altro piu' che il volume complessivo di crediti.", "",
          "L'F1 macro riportato qui e' la **media degli F1 per prodotto**, non l'F1 ricalcolato",
          "a partire dalla precisione e dal richiamo macro: le due quantita' non coincidono",
          "(in Table B1, 0.93626 contro 0.93813; in Table B2, 0.88865 contro 0.89057).", "",
          "**Come leggere lo scarto.** Macro sotto micro significa che i prodotti piccoli vanno",
          "peggio di quelli grandi: e' il caso di Table B1 (0.93626 contro 0.94806), dove i due",
          "titoli in coda alla tabella hanno 13 e 26 voci di gold e insieme valgono lo 0.8% dei",
          "crediti ma un decimo del peso nella media macro. Lo scarto misura quindi la",
          "disomogeneita' fra titoli, non un errore di calcolo.", ""]

    for tag, title, gold, eps, with_role, csv_name in specs:
        rows, summary = build_rows(gold, pred_triples, eps, with_role)
        print_table(f"Table {tag}. {title}", rows, summary)
        write_csv(rows, summary, EXPORTS_DIR / csv_name)
        caption = ("Per-product breakdown of credit-extraction results without role assignment"
                   if not with_role else
                   "Per-product breakdown of name-and-role extraction results")
        scope = ("hold-out corpus, 5 products" if gold is gold_5
                 else "gold standard corpus, 20 products")
        md += [f"## Table {tag}", "",
               f"*Table {tag}. {caption} ({scope}).*", "",
               markdown_table(rows, summary), ""]

    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nTabelle in markdown: {MD_OUT}")
    print(f"CSV per tabella     : {EXPORTS_DIR}")


if __name__ == "__main__":
    main()
