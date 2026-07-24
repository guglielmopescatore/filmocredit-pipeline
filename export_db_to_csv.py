#!/usr/bin/env python3
"""
Export SQLite database tables to CSV files.

Converte in CSV TUTTI i file .db presenti nella cartella db/. Per ogni database
viene esportata la tabella `credits` (default) in un CSV il cui nome deriva dal
nome del .db, es.:
    FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.db
    -> FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.csv

Usage:
    python export_db_to_csv.py                  # tutti i .db -> exports/<db>.csv (tabella credits)
    python export_db_to_csv.py --output-dir out # cartella di output diversa
    python export_db_to_csv.py --table episodes # esporta un'altra tabella
    python export_db_to_csv.py --all-tables     # tutte le tabelle: <db>_<tabella>.csv
    python export_db_to_csv.py --timestamp      # aggiunge _YYYYMMDD_HHMMSS al nome
    python export_db_to_csv.py --list-tables    # elenca le tabelle di ogni .db
"""

import argparse
import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


# Colonne "pulite" esportate per la tabella credits (nell'ordine dato). Le colonne
# non presenti in un dato db vengono saltate automaticamente.
CREDITS_COLUMNS = [
    'id', 'episode_id', 'source_frame', 'role_group', 'secondary_role_group',
    'name', 'role_detail', 'role_group_normalized', 'role_group_corrected',
    'scene_position', 'original_frame_number', 'reviewed_status', 'is_person',
    'normalized_name', 'normalized_name_with_nickname', 'assigned_code', 'code_assignment_status', 'imdb_matches',
    'imdb_name', 'metadata',
]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def get_db_dir() -> Path:
    """Directory dei database SQLite."""
    return get_project_root() / 'db'


def find_db_files() -> list[Path]:
    """Tutti i file .db nella cartella db/ (ordinati per nome)."""
    return sorted(get_db_dir().glob('*.db'))


def _existing_columns(cursor, table_name: str) -> set[str]:
    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}


def export_table_to_csv(
    db_path: Path,
    table_name: str,
    output_dir: Path,
    include_timestamp: bool = False,
    columns: list[str] | None = None,
    output_basename: str | None = None,
) -> Path:
    """Esporta una tabella di un db in CSV.

    Args:
        db_path: percorso del file SQLite.
        table_name: tabella da esportare.
        output_dir: cartella di destinazione.
        include_timestamp: se True aggiunge _YYYYMMDD_HHMMSS al nome file.
        columns: se dato, esporta solo queste colonne (quelle mancanti nel db
            vengono ignorate); altrimenti SELECT *.
        output_basename: base del nome file (default: nome tabella). Nome finale:
            <basename>[_YYYYMMDD_HHMMSS].csv

    Returns:
        Path del CSV creato.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    base = output_basename or table_name
    if include_timestamp:
        base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{base}.csv"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        select_cols = None
        if columns:
            existing = _existing_columns(cursor, table_name)
            select_cols = [c for c in columns if c in existing]
        if select_cols:
            query = f"SELECT {', '.join(select_cols)} FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"

        cursor.execute(query)
        rows = cursor.fetchall()
        column_names = [d[0] for d in cursor.description]

        # Pulisci episode_id della tabella credits: togli i suffissi _End/_Opening.
        if table_name == 'credits' and 'episode_id' in column_names:
            idx = column_names.index('episode_id')
            cleaned = []
            for row in rows:
                row = list(row)
                ep = row[idx]
                if ep:
                    ep = str(ep)
                    if ep.endswith('_End'):
                        ep = ep[:-4]
                    elif ep.endswith('_Opening'):
                        ep = ep[:-8]
                    row[idx] = ep
                cleaned.append(tuple(row))
            rows = cleaned

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            writer.writerows(rows)

        print(f"✅ {db_path.name} :: {table_name} ({len(rows)} righe) -> {output_path.name}")
        return output_path
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error exporting '{table_name}' from {db_path.name}: {e}")
    finally:
        conn.close()


def list_tables(db_path: Path) -> list[str]:
    """List all tables in the database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_table_info(db_path: Path, table_name: str) -> dict:
    """Get information about a table."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = cur.fetchall()
        return {
            'name': table_name,
            'row_count': row_count,
            'columns': [{'name': c[1], 'type': c[2]} for c in columns],
        }
    finally:
        conn.close()


def main():
    """Main execution function."""
    # La console Windows di default e' cp1252: forza UTF-8 per le emoji nei print.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description='Converte in CSV tutti i .db della cartella db/ (nome CSV = nome db).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output-dir', default='exports',
                        help='Cartella di output per i CSV (default: exports/)')
    parser.add_argument('--table', default=None,
                        help='Esporta solo questa tabella (default: credits)')
    parser.add_argument('--all-tables', action='store_true',
                        help='Esporta tutte le tabelle di ogni db: <db>_<tabella>.csv')
    parser.add_argument('--timestamp', action='store_true',
                        help='Aggiunge _YYYYMMDD_HHMMSS al nome del CSV')
    parser.add_argument('--list-tables', action='store_true',
                        help='Elenca le tabelle di ogni .db ed esce')
    parser.add_argument('--info', action='store_true',
                        help='Mostra info dettagliate su tabelle/colonne ed esce')
    args = parser.parse_args()

    output_dir = get_project_root() / args.output_dir
    db_files = find_db_files()
    if not db_files:
        print(f"❌ Nessun file .db trovato in {get_db_dir()}")
        sys.exit(1)

    try:
        if args.list_tables:
            for db_path in db_files:
                tables = list_tables(db_path)
                print(f"\n📊 {db_path.name} ({len(tables)} tabelle):")
                for table in tables:
                    info = get_table_info(db_path, table)
                    print(f"   • {table} ({info['row_count']} righe)")
            sys.exit(0)

        if args.info:
            for db_path in db_files:
                print(f"\n📊 {db_path.name}")
                for table in list_tables(db_path):
                    info = get_table_info(db_path, table)
                    print(f"   📋 {info['name']} ({info['row_count']} righe, {len(info['columns'])} colonne)")
            sys.exit(0)

        print(f"\n🔄 Esporto {len(db_files)} database in CSV -> {output_dir}\n")
        exported = []
        for db_path in db_files:
            if args.table:
                tables_to_export = [args.table]
            elif args.all_tables:
                tables_to_export = list_tables(db_path)
            else:
                tables_to_export = ['credits']
            multi = len(tables_to_export) > 1
            for table in tables_to_export:
                columns = CREDITS_COLUMNS if table == 'credits' else None
                basename = f"{db_path.stem}_{table}" if multi else db_path.stem
                try:
                    out = export_table_to_csv(
                        db_path, table, output_dir,
                        include_timestamp=args.timestamp,
                        columns=columns,
                        output_basename=basename,
                    )
                    exported.append(out)
                except (sqlite3.Error, FileNotFoundError) as e:
                    print(f"⚠️  {db_path.name}: impossibile esportare '{table}': {e}")

        print(f"\n✨ Fatto! {len(exported)} file esportati in {output_dir.relative_to(get_project_root())}/")
    except KeyboardInterrupt:
        print("\n\n⚠️  Export annullato dall'utente.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
