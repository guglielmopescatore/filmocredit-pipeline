"""
Valuta la CORRETTEZZA dell'assegnazione IMDB fatta da GPT_SOL_STANDARD, per ogni
soglia fuzzy disponibile (84/86/88/90/92/94), rispetto all'INTERO gold standard
(tutte le righe is_person=True), con metriche Precision/Recall/F1 in stile
compare_llm_human_metrics.py. Due tabelle separate: gold a 20 prodotti e gold
di validazione a 5 prodotti.

NOTA sulle soglie 84/86: il gold ha colonne nome_corretto_imdb_{N} solo per
N in [88, 90, 92, 94, 96, 98, 99] (le uniche soglie con cui e' stato costruito).
Per le righe FUZZY84/FUZZY86 (create da scripts_v3/create_lower_threshold_dbs.py)
non esiste quindi una colonna dedicata: si usa come riferimento la soglia
disponibile piu' bassa (88), la piu' vicina/permissiva a 84 e 86.

Per ogni riga del gold (episodio, nome normalizzato) si definisce:
  - gold_value: nome_corretto_imdb_{soglia} normalizzato ("" se il gold stesso
    non ha un match valido a quella soglia).
  - pred_value: imdb_name normalizzato assegnato dalla pipeline LLM per quella
    riga (assente/"" se non e' stato trovato un match reale).

E si contano TP/FP/FN cosi':
  - HUMAN override presente in gold (colonna "HUMAN nome/codice imdb corretto"
    valorizzata): un umano ha dovuto correggere l'automatismo del gold, quindi
    nome_corretto_imdb_{soglia} non e' affidabile li' e la riga conta sempre
    come un errore -> FN (se la pipeline non ha assegnato nulla) oppure FP+FN
    (se ha assegnato qualcosa, comunque sbagliato per definizione).
  - gold_value non vuoto (un match corretto e' atteso):
      pred_value == gold_value          -> TP
      pred_value non vuoto ma diverso   -> FP + FN (match sbagliato)
      pred_value vuoto, ma il nome normalizzato del credito == gold_value
                                         -> TP (match testuale "di fallback",
                                            corretto anche senza collegamento IMDB)
      pred_value vuoto e nome diverso   -> FN (mancato)
  - gold_value vuoto (il gold stesso non ha un match valido a quella soglia):
      pred_value non vuoto              -> FP (assegnazione spuria)
      pred_value vuoto                  -> vero negativo, non conteggiato
                                            (non entra in P/R/F1)

Usage:
    python compare_imdb_assignment_vs_gold.py
"""

import csv
import re
import sys
from pathlib import Path

from scripts_v3.utils import normalize_name, strip_parentheticals
import compare_llm_human_metrics as cllm

ROOT = Path(__file__).resolve().parent
GOLD_20_PATH = ROOT / "human_data_to_be_imdbized" / "IMDBIZED_credits_human_corrected_20_products.csv"
GOLD_5_PATH = ROOT / "human_data_to_be_imdbized" / "IMDBIZED_credits_human_corrected_validation_5.csv"
EXPORTS_DIR = ROOT / "exports"
FUZZY_GLOB = "FUZZY*_GPT_SOL_STANDARD_*.csv"
FUZZY_PREFIX_RE = re.compile(r"^FUZZY(\d+)_(.+?)_\d+products", re.IGNORECASE)

# Stesso alias noto usato in compare_llm_human_metrics.py: il gold usa
# "8 1-2"/"3 Percent" (senza suffisso episodio) per due prodotti il cui
# episode_id lato LLM e' invece "8_e_mezzo"/"3_Percent_S01E06_*".
EPISODE_ALIASES = {
    "8 1 2": "8 e mezzo",
    "3 percent": "3 percent s01e06",
}


def canon_episode(raw) -> str:
    s = (raw or "").strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"[_\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\d+)x(\d+)", lambda m: f"s{int(m.group(1)):02d}e{int(m.group(2)):02d}", s)
    s = re.sub(r"s(\d+)e(\d+)", lambda m: f"s{int(m.group(1)):02d}e{int(m.group(2)):02d}", s)
    return EPISODE_ALIASES.get(s, s)


def is_truthy(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "t", "yes")


def norm_imdb_name(value) -> str:
    """Normalizza un nome IMDB (o un nome grezzo) per il confronto, modalita' persona."""
    if not value:
        return ""
    return normalize_name(str(value), is_person=True)


def model_label_from_filename(fname: str) -> str:
    m = FUZZY_PREFIX_RE.match(fname)
    if not m:
        return fname
    label = m.group(2)
    if re.search(r"old[_ ]prompt", fname, re.IGNORECASE):
        label = f"{label} OLD PROMPT"
    return label


def load_gold_index(gold_path: Path):
    """Ritorna (index, thresholds):
    index: dict (episodio_canonico, nome_normalizzato) -> {
        'has_human_override': bool,
        <soglia_int>: nome_corretto_imdb_<soglia> normalizzato, per ogni soglia trovata
    }
    Su chiave duplicata tiene la prima riga e conta la collisione."""
    index = {}
    dupes = 0
    with gold_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        threshold_cols = sorted(
            {int(m.group(1)) for c in (reader.fieldnames or [])
             for m in [re.match(r"nome_corretto_imdb_(\d+)", c)] if m}
        )
        for row in reader:
            if not is_truthy(row.get("is_person")):
                continue
            ep = canon_episode(row.get("numero_episodio"))
            name = norm_imdb_name(row.get("nome"))
            if not name:
                continue
            key = (ep, name)
            if key in index:
                dupes += 1
                continue
            human_override = bool((row.get("HUMAN nome imdb corretto") or "").strip()) or \
                bool((row.get("HUMAN codice imdb corretto") or "").strip())
            entry = {"has_human_override": human_override}
            for t in threshold_cols:
                entry[t] = norm_imdb_name(row.get(f"nome_corretto_imdb_{t}"))
            index[key] = entry
    if dupes:
        print(f"[AVVISO] {dupes} righe gold con chiave (episodio, nome) duplicata in {gold_path.name} - tenuta solo la prima")
    return index, threshold_cols


def load_llm_index(path: Path) -> dict:
    """dict (episodio_canonico, nome_normalizzato) -> imdb_name (stringa, puo' essere vuota)."""
    index = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_truthy(row.get("is_person")):
                continue
            ep = canon_episode(row.get("episode_id"))
            name = strip_parentheticals((row.get("normalized_name") or "").strip().lower())
            if not name:
                continue
            key = (ep, name)
            if key not in index:
                index[key] = (row.get("imdb_name") or "").strip()
    return index


def closest_gold_threshold(gold_thresholds, threshold: int) -> int:
    """Soglia del gold da usare come riferimento per `threshold`: quella esatta
    se esiste, altrimenti la piu' alta disponibile <= threshold, altrimenti
    (soglie richieste piu' basse di qualunque colonna nel gold, es. 84/86) la
    piu' bassa disponibile in assoluto."""
    if threshold in gold_thresholds:
        return threshold
    lower_or_equal = [t for t in gold_thresholds if t <= threshold]
    if lower_or_equal:
        return max(lower_or_equal)
    return min(gold_thresholds)


def evaluate_file(path: Path, threshold: int, gold_index: dict, gold_lookup_threshold: int = None) -> dict:
    """`threshold` e' la soglia fuzzy usata per generare `path` (per i log);
    `gold_lookup_threshold` e' la colonna nome_corretto_imdb_{N} del gold da
    usare come riferimento (vedi closest_gold_threshold) - di norma coincide
    con `threshold`, tranne per le soglie assenti dal gold (84/86)."""
    if gold_lookup_threshold is None:
        gold_lookup_threshold = threshold
    tp = fp = fn = 0
    no_gold_threshold = 0
    fp_examples = []
    fn_examples = []

    llm_index = load_llm_index(path)

    for (ep, name), gold_entry in gold_index.items():
        gold_value = gold_entry.get(gold_lookup_threshold)
        if gold_value is None:
            no_gold_threshold += 1
            continue

        imdb_name = llm_index.get((ep, name), "")
        pred_value = norm_imdb_name(imdb_name) if imdb_name else ""

        if gold_entry.get("has_human_override"):
            fn += 1
            fn_examples.append((name, imdb_name or "(nessuno)", "HUMAN override present in gold"))
            if pred_value:
                fp += 1
                fp_examples.append((name, imdb_name, "HUMAN override present in gold"))
            continue

        if gold_value:
            if pred_value == gold_value:
                tp += 1
            elif pred_value:
                fp += 1
                fn += 1
                fp_examples.append((name, imdb_name, gold_value))
                fn_examples.append((name, imdb_name, gold_value))
            elif name == gold_value:
                tp += 1
            else:
                fn += 1
                fn_examples.append((name, "(nessun match IMDB assegnato)", gold_value))
        else:
            if pred_value:
                fp += 1
                fp_examples.append((name, imdb_name, "(gold: nessun match atteso)"))
            # pred_value vuoto e gold_value vuoto -> vero negativo, non conteggiato

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "no_gold_threshold": no_gold_threshold,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }


def exact_match_row_for_model(label: str, fuzzy88_path: Path, gold_path: Path, episode_scope: str):
    """Riusa compare_llm_human_metrics.py per calcolare l'Exact Match "No Role"
    (nome normalizzato, senza collegamento IMDB) esattamente come fa quello
    script: stessa normalizzazione, stesso file FUZZY88 (l'estrazione dei nomi
    non cambia al variare della soglia fuzzy usata solo per il linking IMDB),
    stesso gold set "human" grezzo (non imdbizzato).
    episode_scope: 'main' (esclude i 5 prodotti di validazione) oppure
    'validation5' (solo i 5 prodotti di validazione)."""
    pred_triples, pred_episodes = cllm.load_pred(fuzzy88_path)
    if episode_scope == "main":
        scoped_pred_episodes = pred_episodes - cllm.VALIDATION5_EPISODES
    else:
        scoped_pred_episodes = pred_episodes & cllm.VALIDATION5_EPISODES

    gold_triples = cllm.load_gold_triples(gold_path)
    gold_episodes = {t[0] for t in gold_triples}

    rows = cllm.evaluate_model(label, pred_triples, scoped_pred_episodes, gold_triples, gold_episodes)
    # rows[0] = "No Role". Slice invece di unpack completo: evaluate_model puo'
    # aggiungere colonne in coda (es. raw_calls) senza rompere questo chiamante.
    tp, fp, fn, p, r, f1 = rows[0][4:10]
    return tp, fp, fn, p, r, f1


def compute_prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def print_and_save_table(rows, out_path: Path):
    """Stampa tabella con colonne allineate, migliore valore per colonna in
    grassetto (ANSI). Stesso stile di compare_llm_human_metrics.py. Salva CSV."""
    header = ["Model", "Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    numeric_cols = {"TP", "FP", "FN", "Precision", "Recall", "F1"}
    lower_is_better = {"FP", "FN"}
    BOLD, RESET = "\033[1m", "\033[0m"

    raw_values = [
        {"Model": label, "Threshold": threshold, "TP": tp, "FP": fp, "FN": fn,
         "Precision": p, "Recall": r, "F1": f1}
        for label, threshold, tp, fp, fn, p, r, f1 in rows
    ]
    str_rows = [
        [str(v["Model"]), str(v["Threshold"]), str(v["TP"]), str(v["FP"]), str(v["FN"]),
         f"{v['Precision']:.5f}", f"{v['Recall']:.5f}", f"{v['F1']:.5f}"]
        for v in raw_values
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in str_rows)) if str_rows else len(h)
        for i, h in enumerate(header)
    ]

    best_idx_per_col = {col: set() for col in numeric_cols}
    for col in numeric_cols:
        if not raw_values:
            continue
        pick = min if col in lower_is_better else max
        best_i = pick(range(len(raw_values)), key=lambda i: raw_values[i][col])
        best_idx_per_col[col].add(best_i)

    def fmt_row(cells, row_idx=None):
        parts = []
        for i, (cell, h) in enumerate(zip(cells, header)):
            padded = cell.rjust(widths[i]) if h in numeric_cols else cell.ljust(widths[i])
            if row_idx is not None and h in numeric_cols and row_idx in best_idx_per_col[h]:
                padded = f"{BOLD}{padded}{RESET}"
            parts.append(padded)
        return "  ".join(parts)

    print()
    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for i, r in enumerate(str_rows):
        print(fmt_row(r, row_idx=i))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for label, threshold, tp, fp, fn, p, r, f1 in rows:
            w.writerow([label, threshold, tp, fp, fn, f"{p:.5f}", f"{r:.5f}", f"{f1:.5f}"])
    print(f"\nMetriche salvate in: {out_path}")


def run_evaluation(gold_path: Path, table_title: str, out_csv_name: str,
                    exact_match_gold_path: Path, exact_match_scope: str):
    if not gold_path.exists():
        print(f"[ERRORE] Gold file non trovato: {gold_path}", file=sys.stderr)
        return

    gold_index, gold_thresholds = load_gold_index(gold_path)
    print(f"\n\n{'#' * 90}")
    print(f"# {table_title}")
    print(f"{'#' * 90}")
    print(f"Gold index: {len(gold_index)} (episodio, nome) unici [is_person], soglie disponibili nel gold: {gold_thresholds}")

    fuzzy_files = sorted(EXPORTS_DIR.glob(FUZZY_GLOB))
    if not fuzzy_files:
        print(f"[ERRORE] Nessun file {FUZZY_GLOB} in {EXPORTS_DIR}", file=sys.stderr)
        return

    sortable_rows = []

    # Riga "Exact Match": stesso calcolo No-Role di compare_llm_human_metrics.py
    # (nome normalizzato senza collegamento IMDB), indipendente dalla soglia
    # fuzzy - usa il file FUZZY88 come riferimento per l'estrazione dei nomi,
    # esattamente come fa quello script.
    fuzzy88_files = [p for p in fuzzy_files if p.name.startswith("FUZZY88_")]
    if fuzzy88_files:
        path = fuzzy88_files[0]
        label = model_label_from_filename(path.name)
        tp, fp, fn, precision, recall, f1 = exact_match_row_for_model(
            label, path, exact_match_gold_path, exact_match_scope
        )
        sortable_rows.append((label, -1, "Exact Match", tp, fp, fn, precision, recall, f1))
        print(f"\nExact Match (No Role, no IMDB - via compare_llm_human_metrics.py, {path.name})")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

    for path in fuzzy_files:
        m = FUZZY_PREFIX_RE.match(path.name)
        if not m:
            print(f"[SKIP] {path.name}: nome file non nel formato FUZZY<soglia>_<modello>_<N>products")
            continue
        threshold = int(m.group(1))
        label = model_label_from_filename(path.name)

        gold_lookup_threshold = closest_gold_threshold(gold_thresholds, threshold)
        result = evaluate_file(path, threshold, gold_index, gold_lookup_threshold)
        precision, recall, f1 = compute_prf(result["tp"], result["fp"], result["fn"])
        sortable_rows.append((label, threshold, str(threshold), result["tp"], result["fp"], result["fn"],
                               precision, recall, f1))

        print(f"\n{path.name}")
        if gold_lookup_threshold != threshold:
            print(f"  [NOTA] soglia {threshold} assente nel gold - uso nome_corretto_imdb_{gold_lookup_threshold} come riferimento")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, "
              f"soglia assente nel gold: {result['no_gold_threshold']}, "
              f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        if result["fp_examples"]:
            print(f"  Esempi FP (fino a 5):")
            for name, llm_val, gold_val in result["fp_examples"][:5]:
                print(f"    '{name}': LLM='{llm_val}' vs gold='{gold_val}'")
        if result["fn_examples"]:
            print(f"  Esempi FN (fino a 5):")
            for name, llm_val, gold_val in result["fn_examples"][:5]:
                print(f"    '{name}': LLM='{llm_val}' vs gold='{gold_val}'")

    sortable_rows.sort(key=lambda r: (r[0], r[1]))
    rows = [(label, threshold_display, tp, fp, fn, p, r, f1)
            for label, _sort_key, threshold_display, tp, fp, fn, p, r, f1 in sortable_rows]
    print_and_save_table(rows, EXPORTS_DIR / out_csv_name)


def main():
    run_evaluation(GOLD_20_PATH, "GOLD STANDARD 20 PRODOTTI",
                    "imdb_assignment_prf_vs_gold_20_products.csv",
                    exact_match_gold_path=cllm.HUMAN_PATH, exact_match_scope="main")
    run_evaluation(GOLD_5_PATH, "GOLD STANDARD 5 PRODOTTI (validazione)",
                    "imdb_assignment_prf_vs_gold_validation5.csv",
                    exact_match_gold_path=cllm.VALIDATION5_GOLD_PATH, exact_match_scope="validation5")


if __name__ == "__main__":
    main()
