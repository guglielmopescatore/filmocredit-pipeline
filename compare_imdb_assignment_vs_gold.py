"""
Valuta la CORRETTEZZA dell'estrazione e dell'assegnazione IMDB (entity linking),
per ogni soglia fuzzy disponibile nel gold (es. 84/86/88/90/92/94), rispetto
all'intero gold standard (tutte le righe is_person=True), con metriche
Precision/Recall/F1 allineate a compare_llm_human_metrics.py.

Quattro tabelle separate:
  1. Gold standard 20 prodotti - No Role (chiave: episodio, nome)
  2. Gold standard 20 prodotti - With Role (chiave: episodio, nome, role_group)
  3. Gold standard di validazione a 5 prodotti - No Role
  4. Gold standard di validazione a 5 prodotti - With Role

La prima riga di ciascuna tabella riporta il benchmark "Exact Match" calcolato
tramite compare_llm_human_metrics.py (pura estrazione testuale prima del linking).

SOGLIE ASSENTI:
  Se un export LLM fa riferimento a una soglia non presente nelle colonne del
  gold standard (nome_corretto_imdb_{N}), la soglia viene saltata (nessun fallback
  artificiale alla 88).

VALORE CANDIDATO LLM:
  Per ogni credito estratto dal modello:
    - se imdb_name e' presente ed ha un valore, si usa imdb_name normalizzato;
    - altrimenti si ricade su normalized_name del modello (mai su 'name' grezzo).

GERARCHIA DI MATCHING PER LA SOGLIA S:
  1. Ramo Standard (imdb_nconst_S vuoto nel Gold):
     L'automatismo non ha trovato codici IMDb nel gold.
     Target = normalized_name del Gold.
     - candidate == gold_normalized_name -> TP
     - candidate != gold_normalized_name -> FP + FN
     - credito non estratto da LLM       -> FN

  2. Ramo con Match IMDb nel Gold (imdb_nconst_S presente nel Gold):
     2.1 Sottocaso "Assente" (HUMAN nome imdb corretto == "Assente"):
         L'umano ha certificato che la persona reale non e' presente su IMDb e che
         il codice trovato dall'automatismo del gold era errato.
         - se l'export LLM contiene imdb_name valorizzato -> FP + FN (match errato)
         - se l'export LLM NON contiene imdb_name (vuoto):
           - llm.normalized_name == gold.normalized_name -> TP
           - llm.normalized_name != gold.normalized_name -> FP + FN
           - credito non estratto da LLM                 -> FN
     2.2 Sottocaso Correzione Umana con Codice Diverso (HUMAN codice != imdb_nconst_S):
         Target = HUMAN nome imdb corretto.
         - candidate == human_target -> TP
         - candidate != human_target -> FP + FN
         - credito non estratto       -> FN
     2.3 Sottocaso Codice Umano Uguale o Campi HUMAN Vuoti (HUMAN codice == imdb_nconst_S o vuoto):
         Target = nome_corretto_imdb_S.
         - candidate == nome_corretto_imdb_S -> TP
         - candidate != nome_corretto_imdb_S -> FP + FN
         - credito non estratto              -> FN

FALSI POSITIVI PURI:
  Tutti i crediti estratti dal modello per gli episodi valutati che non trovano
  alcuna corrispondenza nel Gold Standard (crediti spuri o allucinati) vengono
  conteggiati come FP puri (FP += 1), garantendo che nessun credito venga contato
  due volte.

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
    """Normalizza un nome per il confronto usando la pipeline standard del progetto."""
    if not value:
        return ""
    clean = strip_parentheticals(str(value).strip())
    return normalize_name(clean, is_person=True)


def model_label_from_filename(fname: str) -> str:
    m = FUZZY_PREFIX_RE.match(fname)
    if not m:
        return fname
    key = m.group(2)
    label = cllm.MODEL_LABELS.get(key, cllm.MODEL_LABELS.get(key.lower(), key))
    if re.search(r"old[_ ]prompt", fname, re.IGNORECASE):
        label = f"{label} OLD PROMPT"
    if fname.upper().startswith("NAIVE_"):
        label = f"{label} NAIVE"
    return label


def load_gold_index(gold_path: Path, with_role: bool = False):
    """Ritorna (index, thresholds):
    index: dict con chiave (episodio_canonico, nome_normalizzato) se
    with_role=False, altrimenti (episodio_canonico, nome_normalizzato,
    role_group_normalizzato) -> {
        'gold_normalized_name': str,
        'human_nome': str,
        'human_code': str,
        'raw_role_group': str,
        'nconst_<soglia>': str,
        'nome_<soglia>': str,
    }
    """
    index = {}
    dupes = 0
    key_desc = "(episodio, nome, role_group)" if with_role else "(episodio, nome)"
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
            if not ep or not cllm.episode_allowed(ep):
                continue
            gold_norm = norm_imdb_name(row.get("normalized_name")) or norm_imdb_name(row.get("nome"))
            if not gold_norm:
                continue
            role_norm = cllm.norm_role_group(row.get("role_group"))
            key = (ep, gold_norm, role_norm) if with_role else (ep, gold_norm)
            if key in index:
                dupes += 1
                continue
            entry = {
                "gold_normalized_name": gold_norm,
                "human_nome": (row.get("HUMAN nome imdb corretto") or "").strip(),
                "human_code": (row.get("HUMAN codice imdb corretto") or "").strip(),
            }
            if with_role:
                entry["raw_role_group"] = (row.get("role_group") or "").strip()
            for t in threshold_cols:
                entry[f"nconst_{t}"] = (row.get(f"imdb_nconst_{t}") or "").strip()
                entry[f"nome_{t}"] = norm_imdb_name(row.get(f"nome_corretto_imdb_{t}"))
            index[key] = entry
    if dupes:
        print(f"[AVVISO] {dupes} righe gold con chiave {key_desc} duplicata in {gold_path.name} - tenuta solo la prima")
    return index, threshold_cols


def load_llm_index(path: Path, with_role: bool = False) -> dict:
    """Carica le predizioni LLM indicizzate per (ep, norm_name) o (ep, norm_name, role_norm).
    Ritorna dict: key -> {
        'candidate': str (imdb_name se valorizzato, altrimenti normalized_name),
        'imdb_name': str (norm_imdb_name del campo imdb_name),
        'normalized_name': str (norm_imdb_name del campo normalized_name),
        'raw_imdb_name': str,
        'role_group': str,
        'episode': str
    }
    """
    index = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_truthy(row.get("is_person")):
                continue
            ep = canon_episode(row.get("episode_id"))
            if not ep or not cllm.episode_allowed(ep):
                continue
            norm_name = norm_imdb_name(row.get("normalized_name")) or norm_imdb_name(row.get("name"))
            if not norm_name:
                continue
            role_norm = cllm.norm_role_group(row.get("role_group_normalized"))
            key = (ep, norm_name, role_norm) if with_role else (ep, norm_name)

            imdb_raw = (row.get("imdb_name") or "").strip()
            imdb_norm = norm_imdb_name(imdb_raw) if imdb_raw else ""
            candidate = imdb_norm if imdb_norm else norm_name

            if key not in index or (not index[key]["imdb_name"] and imdb_norm):
                index[key] = {
                    "candidate": candidate,
                    "imdb_name": imdb_norm,
                    "normalized_name": norm_name,
                    "raw_imdb_name": imdb_raw,
                    "role_group": role_norm,
                    "episode": ep,
                }
    return index


def evaluate_file(path: Path, threshold: int, gold_path: Path, with_role: bool = False) -> dict | None:
    """Valuta un file di export LLM alla soglia specificata usando la formulazione insiemistica.
    Ritorna None se la soglia non e' presente tra le colonne del gold (skip).
    """
    gold_targets = set()
    gold_episodes = set()
    gold_details = {}  # key -> dict di dettagli per gli esempi

    with gold_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        threshold_cols = {
            int(m.group(1)) for c in (reader.fieldnames or [])
            for m in [re.match(r"nome_corretto_imdb_(\d+)", c)] if m
        }
        if threshold not in threshold_cols:
            return None

        for row in reader:
            if not is_truthy(row.get("is_person")):
                continue
            ep = canon_episode(row.get("numero_episodio"))
            if not ep or not cllm.episode_allowed(ep):
                continue
            gold_episodes.add(ep)

            gold_norm = norm_imdb_name(row.get("normalized_name")) or norm_imdb_name(row.get("nome"))
            if not gold_norm:
                continue
            role_norm = cllm.norm_role_group(row.get("role_group"))
            raw_role = (row.get("role_group") or "").strip()

            nconst = (row.get(f"imdb_nconst_{threshold}") or "").strip()
            nome_imdb = norm_imdb_name(row.get(f"nome_corretto_imdb_{threshold}"))
            human_nome = (row.get("HUMAN nome imdb corretto") or "").strip()
            human_code = (row.get("HUMAN codice imdb corretto") or "").strip()

            # Gerarchia di determinazione del target
            if not nconst or human_nome.strip().lower() == "assente":
                target = gold_norm
            elif human_code and human_code != nconst:
                target = norm_imdb_name(human_nome)
            else:
                target = nome_imdb

            key = (ep, target, role_norm) if with_role else (ep, target)
            gold_targets.add(key)
            gold_details[key] = {
                "gold_norm": gold_norm,
                "target": target,
                "role_group": raw_role,
                "episode": ep,
            }

    llm_candidates = set()
    llm_details = {}  # key -> dict di dettagli per gli esempi

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_truthy(row.get("is_person")):
                continue
            ep = canon_episode(row.get("episode_id"))
            if not ep or not cllm.episode_allowed(ep) or ep not in gold_episodes:
                continue

            norm_name = norm_imdb_name(row.get("normalized_name")) or norm_imdb_name(row.get("name"))
            if not norm_name:
                continue
            role_norm = cllm.norm_role_group(row.get("role_group_normalized"))
            raw_role = (row.get("role_group_normalized") or "").strip()

            imdb_raw = (row.get("imdb_name") or "").strip()
            imdb_norm = norm_imdb_name(imdb_raw) if imdb_raw else ""
            candidate = imdb_norm if imdb_norm else norm_name

            key = (ep, candidate, role_norm) if with_role else (ep, candidate)
            llm_candidates.add(key)
            llm_details[key] = {
                "candidate": candidate,
                "norm_name": norm_name,
                "imdb_name": imdb_norm,
                "role_group": raw_role,
                "episode": ep,
            }

    tp_set = gold_targets & llm_candidates
    fp_set = llm_candidates - gold_targets
    fn_set = gold_targets - llm_candidates

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    fp_examples = []
    for key in list(fp_set)[:10]:
        det = llm_details.get(key, {})
        name = det.get("norm_name", key[1])
        rg = det.get("role_group", "")
        cand = det.get("candidate", key[1])
        fp_examples.append((name, rg, cand, "(non corrisponde a nessun target del gold)"))

    fn_examples = []
    for key in list(fn_set)[:10]:
        det = gold_details.get(key, {})
        name = det.get("gold_norm", key[1])
        rg = det.get("role_group", "")
        target = det.get("target", key[1])
        fn_examples.append((name, rg, "(non estratto/non collegato)", target))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }


def print_examples(title: str, examples: list, with_role: bool):
    if not examples:
        return
    print(f"  {title} (fino a 5):")
    for ex in examples[:5]:
        name, rg, llm_val, gold_val = ex
        if with_role:
            print(f"    '{name}' [{rg}]: LLM='{llm_val}' vs gold='{gold_val}'")
        else:
            print(f"    '{name}': LLM='{llm_val}' vs gold='{gold_val}'")


def exact_match_row_for_model(label: str, fuzzy88_path: Path, gold_path: Path, episode_scope: str,
                              with_role: bool = False):
    pred_triples, pred_episodes = cllm.load_pred(fuzzy88_path)
    if episode_scope == "main":
        scoped_pred_episodes = pred_episodes - cllm.VALIDATION5_EPISODES
    else:
        scoped_pred_episodes = pred_episodes & cllm.VALIDATION5_EPISODES

    gold_triples = cllm.load_gold_triples(gold_path)
    gold_episodes = {t[0] for t in gold_triples}

    rows = cllm.evaluate_model(label, pred_triples, scoped_pred_episodes, gold_triples, gold_episodes)
    row = rows[1] if with_role else rows[0]
    tp, fp, fn, p, r, f1 = row[4:10]
    return tp, fp, fn, p, r, f1


def compute_prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def print_and_save_table(rows, out_path: Path):
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
    row_modes = ["Exact Match" if v["Threshold"] == "Exact Match" else "Fuzzy" for v in raw_values]
    modes = set(row_modes)
    for mode in modes:
        idxs = [i for i, m in enumerate(row_modes) if m == mode]
        for col in numeric_cols:
            if not idxs:
                continue
            pick = min if col in lower_is_better else max
            best_i = pick(idxs, key=lambda i: raw_values[i][col])
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
                   exact_match_gold_path: Path, exact_match_scope: str,
                   with_role: bool = False):
    if not gold_path.exists():
        print(f"[ERRORE] Gold file non trovato: {gold_path}", file=sys.stderr)
        return

    key_desc = "(episodio, target, role_group)" if with_role else "(episodio, target)"
    mode_name = "With Role" if with_role else "No Role"
    gold_index, gold_thresholds = load_gold_index(gold_path, with_role)
    print(f"\n\n{'#' * 90}")
    print(f"# {table_title}  [chiave {key_desc}]")
    print(f"{'#' * 90}")
    print(f"Gold standard: {len(gold_index)} voci uniche [is_person], soglie disponibili nel gold: {gold_thresholds}")

    fuzzy_files = sorted(EXPORTS_DIR.glob(FUZZY_GLOB))
    if not fuzzy_files:
        print(f"[ERRORE] Nessun file {FUZZY_GLOB} in {EXPORTS_DIR}", file=sys.stderr)
        return

    sortable_rows = []

    # Riga "Exact Match"
    fuzzy88_files = [p for p in fuzzy_files if p.name.startswith("FUZZY88_")]
    if fuzzy88_files:
        path = fuzzy88_files[0]
        label = model_label_from_filename(path.name)
        tp, fp, fn, precision, recall, f1 = exact_match_row_for_model(
            label, path, exact_match_gold_path, exact_match_scope, with_role
        )
        sortable_rows.append((label, -1, "Exact Match", tp, fp, fn, precision, recall, f1))
        print(f"\nExact Match ({mode_name}, no IMDB - via compare_llm_human_metrics.py, {path.name})")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

    for path in fuzzy_files:
        m = FUZZY_PREFIX_RE.match(path.name)
        if not m:
            print(f"[SKIP] {path.name}: formato nome file non riconosciuto")
            continue
        threshold = int(m.group(1))
        label = model_label_from_filename(path.name)

        result = evaluate_file(path, threshold, gold_path, with_role)
        if result is None:
            print(f"\n[SKIP] {path.name}: soglia {threshold} assente nel gold {gold_path.name}")
            continue

        precision, recall, f1 = compute_prf(result["tp"], result["fp"], result["fn"])
        sortable_rows.append((label, threshold, str(threshold), result["tp"], result["fp"], result["fn"],
                               precision, recall, f1))

        print(f"\n{path.name}")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, "
              f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        print_examples("Esempi FP", result["fp_examples"], with_role)
        print_examples("Esempi FN", result["fn_examples"], with_role)

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
    run_evaluation(GOLD_20_PATH, "GOLD STANDARD 20 PRODOTTI",
                   "imdb_assignment_prf_vs_gold_with_role_20_products.csv",
                   exact_match_gold_path=cllm.HUMAN_PATH, exact_match_scope="main",
                   with_role=True)
    run_evaluation(GOLD_5_PATH, "GOLD STANDARD 5 PRODOTTI (validazione)",
                   "imdb_assignment_prf_vs_gold_with_role_validation5.csv",
                   exact_match_gold_path=cllm.VALIDATION5_GOLD_PATH, exact_match_scope="validation5",
                   with_role=True)


if __name__ == "__main__":
    main()
