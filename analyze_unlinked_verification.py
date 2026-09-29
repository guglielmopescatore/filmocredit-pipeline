"""
Rigenera dai file le cifre della VERIFICA MANUALE SUI CREDITI NON AGGANCIATI
(campione di 100 righe compilato da Gabriele), che finora esistevano solo come
numeri calcolati a mano in un foglio Excel.

Produce tre cose:

  1. COPERTURA IMDb DEI NON AGGANCIATI
     Quante delle persone che l'IMDbization NON ha collegato sono comunque
     presenti su IMDb (SI / NO / DUBBIO). Da qui il "76% dei non agganciati e'
     in realta' su IMDb": il collo di bottiglia e' il matching, non la
     copertura del database. I DUBBIO sono ESCLUSI dal rapporto (SI/(SI+NO));
     lo script riporta comunque i due estremi (dubbi tutti assenti / tutti
     presenti) perche' quella esclusione e' una scelta, non un dato.

  2. STRATIFICAZIONE PER CONFIDENZA
     Incrocio fra l'esito della verifica e lo stato che la pipeline aveva
     assegnato a quel credito (ambiguous / manual_required / internal_assigned).
     Serve a mostrare che la confidenza dichiarata dal sistema e' empiricamente
     calibrata. Riporta SEMPRE i denominatori e un intervallo di confidenza di
     Wilson al 95%: con ~30 osservazioni per cella le percentuali da sole non
     sono interpretabili.

  3. PROIEZIONE SUL CORPUS
     Applica il tasso osservato nel campione al tasso di non-agganciati
     misurato sull'export, ottenendo (a) la quota di crediti-persona realmente
     assenti da IMDb e (b) il recall dell'entity linking. Entrambi con IC
     propagati dalla proporzione campionaria.

NOTA SUL CAMPIONE. Le 100 righe sono estratte dal GOLD imdbizzato
(IMDBIZED_credits_human_corrected_*.csv), non dall'export della pipeline:
riguardano quindi crediti che l'IMDbization non ha collegato. Lo stato di
confidenza (punto 2) e' invece una proprieta' della pipeline, e va recuperato
con un join sull'export - join che non e' totale (alcuni crediti del gold il
sistema non li ha estratti affatto). La copertura del join e' riportata
esplicitamente nell'output: e' parte del risultato, non un dettaglio tecnico.

Usage:
    python analyze_unlinked_verification.py
"""

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts_v3.utils import normalize_name

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "file_per_analisi" / "verifica_IMDb_crediti_non_associati.xlsx"
SAMPLE_SHEET = "Campione"
GOLD_20_PATH = ROOT / "human_data_to_be_imdbized" / "IMDBIZED_credits_human_corrected_20_products.csv"
GOLD_5_PATH = ROOT / "human_data_to_be_imdbized" / "IMDBIZED_credits_human_corrected_validation_5.csv"
EXPORT_PATH = ROOT / "exports" / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv"
OUT_DIR = ROOT / "exports"

# Stessi alias di compare_llm_human_metrics.py: il gold usa "8 1-2"/"3 Percent",
# l'export "8_e_mezzo"/"3_Percent_S01E06_*".
EPISODE_ALIASES = {
    "8 1 2": "8 e mezzo",
    "8 1-2": "8 e mezzo",
    "3 percent": "3 percent s01e06",
}

# Ordine di presentazione degli stati, dal piu' al meno confidente.
STATUS_ORDER = ["ambiguous", "manual_required", "internal_assigned", "auto_assigned"]

# Quando lo stesso nome compare nell'export sotto piu' righe con stati diversi
# (stessa persona in due role_group), il join a due chiavi e' ambiguo. Si
# risolve con questa priorita', dichiarata invece che implicita: si tiene lo
# stato che descrive il trattamento PIU' conservativo riservato a quel nome
# (aver coniato un codice interno e' l'esito piu' "chiuso", quindi vince).
STATUS_PRIORITY = ["internal_assigned", "manual_required", "ambiguous", "auto_assigned"]


def wilson(k: int, n: int, z: float = 1.96):
    """Intervallo di confidenza di Wilson al 95% per una proporzione."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x) -> str:
    return "n.d." if x != x else f"{x * 100:.1f}%"


def ep_key(value: str) -> str:
    """Chiave episodio confrontabile fra gold ed export."""
    text = re.sub(r"_(End|Opening)$", "", str(value))
    text = re.sub(r"[_\s]+", " ", text).strip().lower()
    return EPISODE_ALIASES.get(text, text)


def name_key(value: str) -> str:
    return normalize_name(str(value), is_person=True)


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "vero", "si", "yes"}


def load_sample() -> pd.DataFrame:
    df = pd.read_excel(SAMPLE_PATH, sheet_name=SAMPLE_SHEET)
    df["esito"] = df["presente_su_imdb"].astype(str).str.strip().str.upper()
    df["_ep"] = df["numero_episodio"].map(ep_key)
    df["_nn"] = df["nome"].map(name_key)
    return df


def check_sample_against_gold(sample: pd.DataFrame) -> None:
    """Verifica che ogni riga del campione esista ancora nel gold corrente.

    Il campione porta con se' `riga_sorgente`, ma il gold e' stato rigenerato
    dopo l'estrazione, quindi la posizione non e' piu' affidabile: si ricongiunge
    su (episodio, nome normalizzato), che e' stabile. Se questa verifica fallisce
    il campione non descrive piu' il gold e i numeri a valle non valgono.
    """
    keys = set()
    for path in (GOLD_20_PATH, GOLD_5_PATH):
        gold = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
        keys.update(zip(gold["numero_episodio"].map(ep_key), gold["nome"].map(name_key)))
    missing = [
        (row["numero_episodio"], row["nome"])
        for _, row in sample.iterrows()
        if (row["_ep"], row["_nn"]) not in keys
    ]
    found = len(sample) - len(missing)
    print(f"Integrita' campione -> gold: {found}/{len(sample)} righe ritrovate")
    for episode, name in missing[:10]:
        print(f"   NON TROVATA: {episode} | {name}")


def load_export():
    """Indice (episodio, nome normalizzato) -> stati, piu' i totali di corpus."""
    export = pd.read_csv(EXPORT_PATH, dtype=str)
    export["_ep"] = export["episode_id"].map(ep_key)
    export["_nn"] = export["normalized_name"].fillna("").astype(str)

    status_by_key = defaultdict(set)
    for ep, nn, status in zip(export["_ep"], export["_nn"], export["code_assignment_status"].fillna("")):
        if nn:
            status_by_key[(ep, nn)].add(status)

    persons = export[export["is_person"].map(truthy)]
    linked = persons["assigned_code"].fillna("").str.startswith("nm").sum()
    return status_by_key, int(len(persons)), int(linked)


def resolve_status(statuses: set) -> str:
    for candidate in STATUS_PRIORITY:
        if candidate in statuses:
            return candidate
    return next(iter(statuses)) if statuses else ""


def main() -> None:
    sample = load_sample()
    print("=" * 78)
    print(f"VERIFICA CREDITI NON AGGANCIATI - campione di {len(sample)} righe")
    print("=" * 78)
    check_sample_against_gold(sample)

    # ------------------------------------------------------------------ 1.
    counts = Counter(sample["esito"])
    si, no, dubbio = counts.get("SI", 0), counts.get("NO", 0), counts.get("DUBBIO", 0)
    decidible = si + no
    lo, hi = wilson(si, decidible)
    print("\n1. COPERTURA IMDb DEI NON AGGANCIATI")
    print(f"   SI (presente su IMDb): {si}")
    print(f"   NO (assente da IMDb):  {no}")
    print(f"   DUBBIO:                {dubbio}")
    print(f"   -> presenti su IMDb, esclusi i dubbi: {si}/{decidible} = {pct(si / decidible)} "
          f"(IC95% {pct(lo)}-{pct(hi)})")
    print(f"      estremi sui dubbi: {pct(si / len(sample))} (dubbi=assenti) - "
          f"{pct((si + dubbio) / len(sample))} (dubbi=presenti)")
    for corpus, group in sample.groupby("corpus"):
        c = Counter(group["esito"])
        d = c.get("SI", 0) + c.get("NO", 0)
        print(f"      [{corpus}] n={len(group)}: SI={c.get('SI', 0)} NO={c.get('NO', 0)} "
              f"DUBBIO={c.get('DUBBIO', 0)} -> {pct(c.get('SI', 0) / d) if d else 'n.d.'}")

    # ------------------------------------------------------------------ 2.
    status_by_key, n_persons, n_linked = load_export()
    rows = []
    multi = 0
    for _, row in sample.iterrows():
        statuses = status_by_key.get((row["_ep"], row["_nn"]), set())
        if len(statuses) > 1:
            multi += 1
        rows.append({
            "id_campione": row["id_campione"],
            "corpus": row["corpus"],
            "numero_episodio": row["numero_episodio"],
            "nome": row["nome"],
            "role_group": row["role_group"],
            "presente_su_imdb": row["esito"],
            "codice_imdb": row.get("codice_imdb"),
            "code_assignment_status": resolve_status(statuses),
            "stati_multipli": "si" if len(statuses) > 1 else "",
            "note": row.get("note"),
        })
    joined = pd.DataFrame(rows)
    matched = joined[joined["code_assignment_status"] != ""]

    print("\n2. STRATIFICAZIONE PER CONFIDENZA")
    print(f"   Join campione -> export riuscito su {len(matched)}/{len(sample)} righe "
          f"({pct(len(matched) / len(sample))}); {len(sample) - len(matched)} crediti del gold "
          f"non risultano estratti dal sistema.")
    if multi:
        print(f"   {multi} righe avevano piu' stati (stesso nome in piu' role_group): "
              f"risolte con priorita' {' > '.join(STATUS_PRIORITY)}.")
    # Due convenzioni sui DUBBIO, entrambe riportate perche' la scelta sposta
    # il numero di qualche punto e non c'e' una risposta ovvia: SI/(SI+NO) e'
    # coerente con il 76% del punto 1, SI/n e' la lettura piu' prudente.
    print(f"\n   {'stato':<20} {'n':>4} {'SI':>4} {'NO':>4} {'DUB':>4} "
          f"{'SI/(SI+NO)':>12} {'IC95%':>16} {'SI/n':>8}")
    strat_rows = []
    for status in STATUS_ORDER:
        block = matched[matched["code_assignment_status"] == status]
        if block.empty:
            continue
        c = Counter(block["presente_su_imdb"])
        s, n_, d_ = c.get("SI", 0), c.get("NO", 0), c.get("DUBBIO", 0)
        dec = s + n_
        share = s / dec if dec else float("nan")
        lo_, hi_ = wilson(s, dec)
        print(f"   {status:<20} {len(block):>4} {s:>4} {n_:>4} {d_:>4} {pct(share):>12} "
              f"{pct(lo_) + '-' + pct(hi_):>16} {pct(s / len(block)):>8}")
        strat_rows.append({
            "code_assignment_status": status, "n": len(block), "SI": s, "NO": n_, "DUBBIO": d_,
            "quota_esclusi_dubbi": round(share, 4),
            "ic95_basso": round(lo_, 4), "ic95_alto": round(hi_, 4),
            "quota_su_n": round(s / len(block), 4),
        })

    # ------------------------------------------------------------------ 3.
    n_unlinked = n_persons - n_linked
    p_unlinked = n_unlinked / n_persons
    p_absent = no / decidible
    lo_a, hi_a = wilson(no, decidible)
    absent_share = p_unlinked * p_absent
    # Piu' persone sono davvero assenti da IMDb, piu' piccolo e' l'insieme di
    # quelle agganciabili e quindi PIU ALTO il recall: l'estremo basso dell'IC
    # sul recall corrisponde all'estremo basso sulla quota di assenti.
    recall = n_linked / (n_persons - n_persons * absent_share)
    recall_lo = n_linked / (n_persons - n_persons * p_unlinked * lo_a)
    recall_hi = n_linked / (n_persons - n_persons * p_unlinked * hi_a)

    print("\n3. PROIEZIONE SUL CORPUS (export FUZZY88 GPT Sol Standard, 25 prodotti)")
    print(f"   crediti-persona:            {n_persons}")
    print(f"   agganciati a IMDb (nm):     {n_linked} ({pct(n_linked / n_persons)})")
    print(f"   NON agganciati:             {n_unlinked} ({pct(p_unlinked)})")
    print(f"   -> realmente assenti da IMDb: {pct(p_unlinked)} x {no}/{decidible} = "
          f"{pct(absent_share)} (IC95% {pct(p_unlinked * lo_a)}-{pct(p_unlinked * hi_a)})")
    print(f"   -> recall dell'entity linking: {pct(recall)} "
          f"(IC95% {pct(recall_lo)}-{pct(recall_hi)})")

    # ------------------------------------------------------------------ output
    OUT_DIR.mkdir(exist_ok=True)
    joined.to_csv(OUT_DIR / "verifica_non_agganciati_campione.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strat_rows).to_csv(
        OUT_DIR / "verifica_non_agganciati_stratificazione.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "campione_n": len(sample), "SI": si, "NO": no, "DUBBIO": dubbio,
        "quota_presenti_su_imdb_esclusi_dubbi": round(si / decidible, 4),
        "ic95_basso": round(lo, 4), "ic95_alto": round(hi, 4),
        "crediti_persona_corpus": n_persons, "agganciati": n_linked, "non_agganciati": n_unlinked,
        "quota_non_agganciati": round(p_unlinked, 4),
        "quota_realmente_assenti_da_imdb": round(absent_share, 4),
        "recall_entity_linking": round(recall, 4),
        "recall_ic95_basso": round(recall_lo, 4), "recall_ic95_alto": round(recall_hi, 4),
        "join_export_riuscito": len(matched),
    }]).to_csv(OUT_DIR / "verifica_non_agganciati_sintesi.csv", index=False, encoding="utf-8-sig")
    print("\nScritti in exports/: verifica_non_agganciati_{campione,stratificazione,sintesi}.csv")


if __name__ == "__main__":
    main()
