"""
Confronta i risultati grezzi dei vari LLM (file FUZZY88_*.csv in exports/)
contro il gold set HUMAN (HUMAN_credits_final.csv).

Produce, per ogni LLM trovato, le metrice di Exact Match in due modalita':
  - "No Role":  match solo sul nome normalizzato (per episodio)
  - "With Role": match su nome normalizzato + role_group

La normalizzazione del nome umano riusa utils.normalize_name(is_person=True),
identica a quella usata per costruire la colonna `normalized_name` dei file LLM.
I file LLM contribuiscono con le colonne `normalized_name` e `role_group_normalized` cosi' come sono.

Filtraggio: da entrambi i lati si scartano le righe con is_person non veritiero (1 / TRUE).
Deduplicazione (a livello di SINGOLO PRODOTTO, su gold e LLM):
  - No Role:   ogni coppia (prodotto, nome) DISTINTA vale 1 voce. Un nome ripetuto
               nello stesso prodotto (anche sotto piu' role_group) collassa in una
               sola voce; lo stesso nome in prodotti diversi resta due voci distinte.
  - With Role: ogni tripla (prodotto, nome, role_group) DISTINTA vale 1 voce.

Copertura: ogni modello e' valutato SOLO sui prodotti che ha effettivamente
processato. La colonna `num_products` riporta quanti prodotti quel modello copre e
il gold viene ristretto a quegli stessi prodotti: un modello che ne ha 5 si confronta
con i 5 gold corrispondenti, uno che li ha tutti e 20 con tutti e 20. Cosi' TP/FP/FN
(e quindi P/R/F1) sono confrontabili tra modelli con coperture diverse.

I file FUZZY88_* vengono scoperti automaticamente. Se uno dei tre (es. gpt_terra)
non e' ancora presente, lo script lo ignora ed esegue l'analisi sui file disponibili.
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

from scripts_v3.utils import normalize_name, strip_parentheticals

ROOT = Path(__file__).resolve().parent
HUMAN_PATH = ROOT / "human_data_to_be_imdbized" / "credits_human_corrected_merged_to_be_imdbized_20_products.csv"
EXPORTS_DIR = ROOT / "exports"

# Gold umano separato per i 5 prodotti esclusivi di GPT Sol Standard (nessun
# altro modello li ha ancora processati) - non fanno parte del gold set
# principale, quindi vengono valutati a parte in una tabella dedicata.
VALIDATION5_GOLD_PATH = ROOT / "human_data_to_be_imdbized" / "credits_human_corrected_to_be_imdbized_validation_5.csv"
VALIDATION5_EPISODES = {
    "3 percent s01e06",
    "blue eye samurai s01e01",
    "honeyland",
    "persepolis",
    "wild strawberries",
}

# Auto-discovery: tutti i file FUZZY88_*.csv dentro exports/
FUZZY_GLOB = "FUZZY88_*.csv"

# Mappa il token estratto dal nome file al label leggibile del modello.
MODEL_LABELS = {
    "sonnet4.5": "Claude Sonnet 4.5",
    "claude": "Claude Sonnet 4.5",
    "gpt5.2": "GPT 5.2",
    "gpt_terra": "GPT Terra",
    "gpt_terra_standard": "GPT Terra Standard",
    "gpt_sol_reasoning": "GPT Sol Reasoning",
    "gpt_sol_standard": "GPT Sol Standard",
    "gemma_4_12b": "Gemma 4 12b",
}

# Alias noti tra la forma canonica del gold umano e quella usata dagli export LLM:
#   "8 1-2" (umano) <-> "8_e_mezzo" (LLM)
#   "3 Percent" (umano, senza "S01E06") <-> "3_Percent_S01E06_*" (LLM)
EPISODE_ALIASES = {
    "8 1 2": "8 e mezzo",
    "3 percent": "3 percent s01e06",
}

# CONSENT: analysis is limited to episodes set to True. Set an episode to False
# to exclude it from both the gold set and the LLM predictions. Keys MUST be in
# canonical form (lowercase, "s01e01" style). All default to True.
EPISODE_CONSENT = {
    "8 e mezzo": True,
    "amelie": True,
    "apocalypse now": True,
    "chernobyl s01e01": True,
    "dark s01e09": True,
    "el desorden que dejas s01e03": True,
    "eternal sunshine of the spotless mind": True,
    "fight club": True,
    "hill street blues s01e13": True,
    "la grande bellezza": True,
    "la piovra s01e02": True,
    "maigret s03e01": True,
    "planet earth s01e10": True,
    "prime suspect s01e01": True,
    "psycho": True,
    "romanzo criminale s01e01": True,
    "se7en": True,
    "the world at war s01e03": True,
    "twin peaks s01e03": True,
    "yes prime minister s01e08": True,
}


def episode_allowed(ep: str) -> bool:
    """True if the (canonical) episode is consented for analysis."""
    return EPISODE_CONSENT.get(ep, True)


def canon_episode(raw: str) -> str:
    """Canonicalizza un identificatore di episodio (entrambi i lati) in una
    chiave unica, gestendo spazi<->underscore, S01E09<->1x9 e l'alias '8 1-2'."""
    s = (raw or "").strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"[_\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # "1x9" -> "s01e09"
    s = re.sub(
        r"(\d+)x(\d+)",
        lambda m: f"s{int(m.group(1)):02d}e{int(m.group(2)):02d}",
        s,
    )
    # "S01E09" -> "s01e09"
    s = re.sub(
        r"s(\d+)e(\d+)",
        lambda m: f"s{int(m.group(1)):02d}e{int(m.group(2)):02d}",
        s,
    )
    return EPISODE_ALIASES.get(s, s)


def is_truthy(value) -> bool:
    """is_person veritiero = '1' o 'TRUE' (case-insensitive)."""
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("1", "true", "t", "yes")


def norm_role_group(value) -> str:
    """Normalizzazione leggera del role_group per il confronto With Role."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def human_normalized_name(raw_name) -> str:
    """Riusa la pipeline unica del progetto in modalita' persona (solo persone
    arrivano qui, vedi filtro is_person in load_gold_triples): nickname tra
    virgolette e testo tra parentesi tonde vengono rimossi dal nome."""
    if raw_name is None:
        return ""
    return normalize_name(str(raw_name), is_person=True)


def model_label_from_filename(fname: str) -> str:
    """Estrae il nome del modello dal nome del CSV FUZZY: il token tra 'FUZZY<n>_'
    e '_<n>products'. Es. 'FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv'
    -> 'GPT_SOL_STANDARD'. Se il token e' in MODEL_LABELS (confronto case-insensitive)
    usa l'etichetta leggibile, altrimenti restituisce il token grezzo.
    Se nel nome file compare 'OLD_PROMPT' (dopo '_<n>products'), lo mantiene
    aggiungendo ' OLD PROMPT' all'etichetta (cosi' due varianti dello stesso
    modello restano distinte)."""
    m = re.search(r"FUZZY\d+_(.+?)_\d+products", fname)
    if not m:
        return fname
    key = m.group(1)
    label = MODEL_LABELS.get(key, MODEL_LABELS.get(key.lower(), key))
    if re.search(r"old[_ ]prompt", fname, re.IGNORECASE):
        label = f"{label} OLD PROMPT"
    return label


def counters_from_triples(triples):
    """Costruisce i due Counter a partire dall'insieme di triple distinte
    (episode, name, role_group). Deduplicazione a livello di SINGOLO PRODOTTO su
    entrambe le modalita':
      - no_role:   ogni coppia (episode, name) DISTINTA vale 1. Un nome ripetuto
                   nello stesso prodotto (anche sotto piu' role_group) collassa in
                   una sola voce; lo stesso nome in prodotti diversi resta due voci.
      - with_role: ogni tripla (episode, name, role_group) DISTINTA vale 1.
    """
    no_role = Counter()
    with_role = Counter()
    for ep, name, rg in triples:
        no_role[(ep, name)] = 1
        with_role[(ep, name, rg)] = 1
    return no_role, with_role


def load_gold_triples(path: Path = HUMAN_PATH):
    """Carica un gold set HUMAN (solo is_person veritiero) come insieme di triple
    distinte (episode, name, role_group). Il filtro per prodotto avviene dopo,
    per modello, in main(). `path` permette di riusare la stessa funzione anche
    per il gold set separato dei 5 prodotti di validazione (VALIDATION5_GOLD_PATH)."""
    if not path.exists():
        print(f"[ERRORE] File gold non trovato: {path}", file=sys.stderr)
        sys.exit(1)

    triples = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if not is_truthy(row.get("is_person")):
                continue
            ep = canon_episode(row.get("numero_episodio"))
            if not episode_allowed(ep):
                continue
            name = human_normalized_name(row.get("nome"))
            if not name:
                continue
            rg = norm_role_group(row.get("role_group"))
            triples.add((ep, name, rg))
    return triples


def load_pred(path: Path):
    """Carica un file LLM.

    Ritorna (triples, episodes):
      - triples: triple distinte (episode, name, role_group), solo is_person veritiero;
      - episodes: i prodotti presenti nel file, raccolti su TUTTE le righe (anche
        quelle non-persona), perche' un episodio processato che non ha prodotto
        nomi di persona resta comunque un prodotto "di cui abbiamo i dati".
    """
    triples = set()
    episodes = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = canon_episode(row.get("episode_id"))
            if not ep or not episode_allowed(ep):
                continue
            episodes.add(ep)
            if not is_truthy(row.get("is_person")):
                continue
            # normalized_name dell'export conserva le parentesi: rimuovile qui per
            # coerenza col gold (i FUZZY88 sono file pre-calcolati, non rigenerati).
            name = strip_parentheticals((row.get("normalized_name") or "").strip().lower())
            if not name:
                continue
            rg = norm_role_group(row.get("role_group_normalized"))
            triples.add((ep, name, rg))
    return triples, episodes


def compute_metrics(gold: Counter, pred: Counter):
    tp = sum(min(g, pred[k]) for k, g in gold.items())
    fp = sum(v for k, v in pred.items() if k not in gold) + sum(
        max(0, v - gold[k]) for k, v in pred.items() if k in gold
    )
    fn = sum(v for k, v in gold.items() if k not in pred) + sum(
        max(0, v - pred[k]) for k, v in gold.items() if k in pred
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return tp, fp, fn, precision, recall, f1


def evaluate_model(label, pred_triples, pred_episodes, gold_triples, gold_episodes):
    """Valuta un modello contro un gold set, ristretto all'intersezione tra i
    prodotti che il modello copre (pred_episodes) e quelli presenti nel gold
    (gold_episodes). Ritorna le due righe (No Role, With Role) per la tabella.
    Fattorizzata da main() cosi' la stessa logica serve sia per il confronto
    principale sia per la tabella separata dei 5 prodotti di validazione."""
    eval_episodes = pred_episodes & gold_episodes
    missing_in_gold = pred_episodes - gold_episodes
    if missing_in_gold:
        print(
            f"[AVVISO] {label}: {len(missing_in_gold)} prodotti senza gold umano, "
            f"esclusi dal confronto: {sorted(missing_in_gold)}"
        )

    gold_sub = {t for t in gold_triples if t[0] in eval_episodes}
    pred_sub = {t for t in pred_triples if t[0] in eval_episodes}
    gold_nr, gold_wr = counters_from_triples(gold_sub)
    pred_nr, pred_wr = counters_from_triples(pred_sub)
    num_products = len(eval_episodes)

    print(
        f"  {label}: {num_products} prodotti, "
        f"gold ristretto a {sum(gold_nr.values())} voci (No Role), "
        f"{sum(gold_wr.values())} voci (With Role)"
    )

    rows = []
    tp, fp, fn, p, r, f1 = compute_metrics(gold_nr, pred_nr)
    rows.append((label, num_products, "Exact Match", "No Role", tp, fp, fn, p, r, f1))

    tp, fp, fn, p, r, f1 = compute_metrics(gold_wr, pred_wr)
    rows.append((label, num_products, "Exact Match", "With Role", tp, fp, fn, p, r, f1))
    return rows


def print_and_save_table(rows, out_path: Path):
    """Stampa tabella con colonne allineate: testo a sinistra, numeri a destra.
    Il miglior valore di ogni colonna numerica viene evidenziato in grassetto
    (ANSI), confrontando solo righe con lo stesso Mode (No Role / With Role non
    sono confrontabili tra loro: scale diverse). Salva anche in CSV."""
    header = ["Model", "num_products", "Method", "Mode", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    numeric_cols = {"num_products", "TP", "FP", "FN", "Precision", "Recall", "F1"}
    lower_is_better = {"FP", "FN"}
    BOLD, RESET = "\033[1m", "\033[0m"

    raw_values = [
        {"Model": label, "num_products": num_products, "Method": method, "Mode": mode,
         "TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f1}
        for label, num_products, method, mode, tp, fp, fn, p, r, f1 in rows
    ]
    str_rows = [
        [str(v["Model"]), str(v["num_products"]), str(v["Method"]), str(v["Mode"]),
         str(v["TP"]), str(v["FP"]), str(v["FN"]),
         f"{v['Precision']:.5f}", f"{v['Recall']:.5f}", f"{v['F1']:.5f}"]
        for v in raw_values
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in str_rows)) if str_rows else len(h)
        for i, h in enumerate(header)
    ]

    # Per ogni colonna numerica, l'indice di riga migliore DENTRO ciascun gruppo di Mode.
    best_idx_per_col = {col: set() for col in numeric_cols}
    modes = {v["Mode"] for v in raw_values}
    for mode in modes:
        idxs = [i for i, v in enumerate(raw_values) if v["Mode"] == mode]
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
        for label, num_products, method, mode, tp, fp, fn, p, r, f1 in rows:
            w.writerow([label, num_products, method, mode, tp, fp, fn,
                        f"{p:.5f}", f"{r:.5f}", f"{f1:.5f}"])
    print(f"\nMetriche salvate in: {out_path}")


def main():
    fuzzy_files = sorted(EXPORTS_DIR.glob(FUZZY_GLOB))
    if not fuzzy_files:
        print(f"[ERRORE] Nessun file {FUZZY_GLOB} in {EXPORTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"File HUMAN gold: {HUMAN_PATH}")
    print(f"File LLM trovati ({len(fuzzy_files)}):")
    for p in fuzzy_files:
        print(f"  - {p.name}")

    gold_triples = load_gold_triples()
    gold_episodes = {t[0] for t in gold_triples}
    print(f"\nGold set: {len(gold_triples)} voci su {len(gold_episodes)} prodotti")

    # Tabella principale: i 5 prodotti esclusivi di GPT Sol Standard (vedi
    # VALIDATION5_EPISODES) vengono esclusi qui - non hanno gold in HUMAN_PATH
    # comunque, ma escluderli esplicitamente evita avvisi "senza gold umano"
    # fuorvianti ed e' piu' chiaro: quei 5 hanno la loro tabella dedicata sotto.
    rows = []
    for path in fuzzy_files:
        label = model_label_from_filename(path.name)
        pred_triples, pred_episodes = load_pred(path)
        main_pred_episodes = pred_episodes - VALIDATION5_EPISODES
        rows.extend(evaluate_model(label, pred_triples, main_pred_episodes, gold_triples, gold_episodes))

    print_and_save_table(rows, EXPORTS_DIR / "llm_human_exact_match_metrics.csv")

    # Tabella separata: i 5 prodotti che, ad oggi, SOLO GPT Sol Standard ha
    # processato (3 Percent S01E06, Blue Eye Samurai S01E01, Honeyland,
    # Persepolis, Wild Strawberries). Gold dedicato (VALIDATION5_GOLD_PATH),
    # non ancora unito al gold set principale. Stessa identica metodologia di
    # "true matching" (nessuna imdbizzazione) usata sopra - solo il gold e
    # l'universo di episodi cambiano. Solo i modelli che coprono almeno uno di
    # questi 5 prodotti compaiono in questa tabella.
    print("\n" + "=" * 60)
    print("VALIDATION SET: 5 prodotti esclusivi (solo modelli che li coprono)")
    print("=" * 60)

    gold_v5_triples = load_gold_triples(VALIDATION5_GOLD_PATH)
    gold_v5_episodes = {t[0] for t in gold_v5_triples}
    print(f"Gold validation-5: {len(gold_v5_triples)} voci su {len(gold_v5_episodes)} prodotti")

    v5_rows = []
    for path in fuzzy_files:
        label = model_label_from_filename(path.name)
        pred_triples, pred_episodes = load_pred(path)
        v5_pred_episodes = pred_episodes & VALIDATION5_EPISODES
        if not v5_pred_episodes:
            continue
        v5_rows.extend(evaluate_model(label, pred_triples, v5_pred_episodes, gold_v5_triples, gold_v5_episodes))

    if v5_rows:
        print_and_save_table(v5_rows, EXPORTS_DIR / "llm_human_exact_match_metrics_validation5.csv")
    else:
        print("Nessun modello copre i prodotti della validation-5.")


if __name__ == "__main__":
    main()
