#!/usr/bin/env python3
"""
build_imdb_sqlite.py: Download IMDb name.basics TSV and build SQLite FTS5 database

Usage:
    python build_imdb_sqlite.py \
        [--out-db db/imdb.sqlite] \
        [--tsv-url https://datasets.imdbws.com/name.basics.tsv.gz] \
        [--force] [--chunk 50000] [--quiet]

Examples:
    python build_imdb_sqlite.py
    python build_imdb_sqlite.py --force
"""
import argparse
import os
import sys
import sqlite3
import gzip
import csv
import urllib.request
from pathlib import Path

# tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' is not installed. Please install via 'pip install tqdm'.")
    sys.exit(1)


def download_tsv(url: str, dest: Path, quiet: bool):
    """Download TSV file with progress bar."""
    if not quiet:
        print(f"Downloading {url} to {dest}")
    with urllib.request.urlopen(url) as response:
        total = int(response.getheader('Content-Length', 0))
        with open(dest, 'wb') as out_f, tqdm(
            total=total, unit='B', unit_scale=True,
            desc='Downloading', leave=False, disable=quiet
        ) as pbar:
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                pbar.update(len(chunk))
    if not quiet:
        print("Download complete.")


def build_db(db_path: Path, tsv_path: Path, chunk_size: int, quiet: bool):
    """Stream-parse TSV and build SQLite DB with FTS5."""
    if not quiet:
        print(f"Building SQLite DB at {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Performance pragmas
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = WAL")

    # Create tables
    cur.execute("""
CREATE TABLE IF NOT EXISTS name(
    nm TEXT PRIMARY KEY,
    primary_name TEXT,
    jobs TEXT,
    birth_year TEXT,
    known_for_titles TEXT
)""")
    cur.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS name_fts
    USING fts5(primary_name, content='name')
""")

    # Stream-parse TSV
    with gzip.open(tsv_path, 'rt', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        batch = []
        count = 0
        for row in reader:
            nm, primary, birth, death, known, *rest = row
            batch.append((nm, primary, rest[0] if rest else '', birth, known))
            count += 1
            if count % chunk_size == 0:
                cur.executemany(
                    "INSERT OR REPLACE INTO name VALUES (?, ?, ?, ?, ?)", batch
                )
                cur.execute("INSERT INTO name_fts(name_fts) VALUES('rebuild')")
                conn.commit()
                if not quiet:
                    print(f"Inserted {count} rows...")
                batch.clear()
        # Final batch
        if batch:
            cur.executemany(
                "INSERT OR REPLACE INTO name VALUES (?, ?, ?, ?, ?)", batch
            )
            cur.execute("INSERT INTO name_fts(name_fts) VALUES('rebuild')")
            conn.commit()
            if not quiet:
                print(f"Inserted total {count} rows.")

    # Optimize
    if not quiet:
        print("Finalizing database (ANALYZE, VACUUM)...")
    cur.execute("ANALYZE")
    conn.commit()
    cur.execute("VACUUM")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download IMDb name.basics TSV and build SQLite database"
    )
    parser.add_argument('--out-db', default='db/imdb.sqlite',
                        help='Path to output SQLite database')
    parser.add_argument('--tsv-url',
                        default='https://datasets.imdbws.com/name.basics.tsv.gz',
                        help='URL to IMDb name.basics.tsv.gz')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--chunk', type=int, default=50000,
                        help='Rows per INSERT batch')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce console output')
    args = parser.parse_args()

    out_db = Path(args.out_db)
    out_tsv = out_db.with_name('name.basics.tsv.gz')
    out_db.parent.mkdir(parents=True, exist_ok=True)

    # Download TSV if needed
    if args.force or not out_tsv.is_file():
        download_tsv(args.tsv_url, out_tsv, args.quiet)
    else:
        if not args.quiet:
            print(f"TSV already exists at {out_tsv}")

    # Build DB
    if out_db.is_file() and not args.force:
        print(f"Database already built at {out_db}. Use --force to overwrite.")
        sys.exit(0)
    try:
        build_db(out_db, out_tsv, args.chunk, args.quiet)
    except KeyboardInterrupt:
        print("Interrupted! Cleaning up and exiting.")
        sys.exit(1)

    # Print summary
    db_size = out_db.stat().st_size / (1024*1024)
    tsv_size = out_tsv.stat().st_size / (1024*1024)
    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM name")
    row_count = cur.fetchone()[0]
    conn.close()
    print(f"Rows: {row_count}")
    print(f"DB size: {db_size:.1f} MB")
    print(f"TSV size: {tsv_size:.1f} MB")

if __name__ == '__main__':
    main()
