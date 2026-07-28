"""
Tempi di elaborazione dei 5 prodotti del gold standard di validazione
(3 Percent, Blue Eye Samurai, Honeyland, Persepolis, Wild Strawberries).

Tabella: una riga per episodio, una colonna per step (step1..step4) piu' una
colonna Totale; l'ultima riga somma ogni step e chiude con il totale generale
nell'ultima cella.

I tempi vengono dalla tabella `Time` del db, popolata a ogni esecuzione della
pipeline. Nel db i prodotti spezzati in due parti compaiono come due
episode_id distinti (`Persepolis_End` e `Persepolis_Opening`): di default le
due righe vengono sommate nel prodotto, perche' il gold standard e' definito
sui 5 prodotti. Con --by-episode si vedono invece gli episode_id grezzi.

Usage:
    python compute_validation5_times.py
    python compute_validation5_times.py --by-episode
    python compute_validation5_times.py --db db/FUZZY94_GPT_SOL_STANDARD_25products_tvcredits_v3.db
    python compute_validation5_times.py --seconds        # secondi grezzi invece di h:mm:ss
"""

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "db" / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.db"
OUT_PATH = ROOT / "exports" / "validation5_times.csv"

# I 5 prodotti del gold di validazione, nella forma con cui iniziano gli
# episode_id del db (gli stessi di VALIDATION5_EPISODES in
# compare_llm_human_metrics.py, qui in forma grezza per il match sul db).
VALIDATION5_PREFIXES = (
    "3_Percent",
    "Blue_Eye_Samurai",
    "Honeyland",
    "Persepolis",
    "Wild_Strawberries",
)

TOTAL_LABEL = "TOTALE"


def product_of(episode_id: str) -> str:
    """Prodotto a cui appartiene un episode_id, togliendo il suffisso di parte."""
    return re.sub(r"_(End|Opening)$", "", episode_id)


def is_validation5(episode_id: str) -> bool:
    return any(episode_id.startswith(p) for p in VALIDATION5_PREFIXES)


def fmt_time(seconds, raw: bool) -> str:
    """h:mm:ss leggibile, oppure secondi grezzi con --seconds. '-' se assente."""
    if seconds is None:
        return "-"
    if raw:
        return f"{seconds:.2f}"
    total = int(round(seconds))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def load_times(db_path: Path, by_episode: bool):
    """Ritorna (times, steps):
    times: dict riga -> {step: secondi}
    steps: elenco ordinato degli step trovati
    La riga e' l'episode_id grezzo con --by-episode, altrimenti il prodotto.
    """
    conn = sqlite3.connect(db_path)
    try:
        has_time = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='Time'"
        ).fetchone()
        if not has_time:
            print(f"[ERRORE] {db_path.name} non ha una tabella 'Time'.", file=sys.stderr)
            sys.exit(1)
        rows = conn.execute("SELECT episode_id, step, seconds FROM Time").fetchall()
    finally:
        conn.close()

    times, steps = {}, set()
    for episode_id, step, seconds in rows:
        episode_id = str(episode_id or "")
        if not is_validation5(episode_id):
            continue
        key = episode_id if by_episode else product_of(episode_id)
        steps.add(step)
        # Somma: piu' episode_id confluiscono nella stessa riga quando il
        # prodotto e' diviso in End + Opening (solo senza --by-episode).
        times.setdefault(key, {})
        times[key][step] = times[key].get(step, 0.0) + float(seconds or 0.0)
    return times, sorted(steps)


def build_table(times: dict, steps: list, raw: bool, by_episode: bool):
    """Header + righe formattate, con la riga dei totali in fondo."""
    header = ["Episodio" if by_episode else "Prodotto"] + steps + ["Totale"]

    body, col_totals, grand_total = [], {s: 0.0 for s in steps}, 0.0
    for key in sorted(times):
        per_step = times[key]
        row_total = sum(per_step.values())
        grand_total += row_total
        cells = [key]
        for s in steps:
            value = per_step.get(s)
            if value is not None:
                col_totals[s] += value
            cells.append(fmt_time(value, raw))
        cells.append(fmt_time(row_total, raw))
        body.append(cells)

    totals_row = [TOTAL_LABEL] + [fmt_time(col_totals[s], raw) for s in steps]
    totals_row.append(fmt_time(grand_total, raw))
    return header, body, totals_row, col_totals, grand_total


def print_table(header, body, totals_row):
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body + [totals_row]))
        for i in range(len(header))
    ]

    def line(cells, bold=False):
        out = [
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        ]
        text = "  ".join(out)
        return f"\033[1m{text}\033[0m" if bold else text

    print()
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for row in body:
        print(line(row))
    print("  ".join("-" * w for w in widths))
    print(line(totals_row, bold=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"Database da cui leggere i tempi (default: {DEFAULT_DB.name})")
    parser.add_argument("--by-episode", action="store_true",
                        help="Una riga per episode_id grezzo (End/Opening separati) invece che per prodotto")
    parser.add_argument("--seconds", action="store_true",
                        help="Mostra i secondi grezzi invece del formato h:mm:ss")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"[ERRORE] Database non trovato: {args.db}", file=sys.stderr)
        sys.exit(1)

    times, steps = load_times(args.db, args.by_episode)
    if not times:
        print(f"[ERRORE] Nessun tempo trovato per i 5 prodotti di validazione in {args.db.name}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {args.db.name}")
    print(f"Righe: {'episode_id' if args.by_episode else 'prodotto'}  |  step trovati: {', '.join(steps)}")

    header, body, totals_row, col_totals, grand_total = build_table(times, steps, args.seconds, args.by_episode)
    print_table(header, body, totals_row)

    missing = [
        (row[0], step)
        for row in body
        for step, cell in zip(steps, row[1:1 + len(steps)])
        if cell == "-"
    ]
    if missing:
        print(f"\n[AVVISO] {len(missing)} combinazione/i (riga, step) senza tempo registrato: {missing}")

    # Il CSV conserva i secondi grezzi: e' il formato utile per rielaborazioni.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(times):
            per_step = times[key]
            w.writerow([key] + [f"{per_step.get(s, 0.0):.2f}" for s in steps]
                       + [f"{sum(per_step.values()):.2f}"])
        w.writerow([TOTAL_LABEL] + [f"{col_totals[s]:.2f}" for s in steps] + [f"{grand_total:.2f}"])
    print(f"\nTempi (in secondi) salvati in: {OUT_PATH}")


if __name__ == "__main__":
    main()
