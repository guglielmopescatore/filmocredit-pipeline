#!/usr/bin/env python3
"""
Redoes Step 4 (IMDB code assignment) for every FUZZY##_..._tvcredits_v3.db
snapshot in db/ that already has a "Time" table, using the EXACT same code
path the app's GUI Step 4 button uses per episode
(scripts_v3.imdb_batch_validation.IMDBBatchValidatorWithCodeAssignment +
get_unprocessed_credits(episode_id=...)), so results are identical to what
re-running Step 4 from the interface would produce for each episode - just
automated across every snapshot at once.

DB files with no "Time" table are skipped entirely (not reprocessed at all) -
only files that already track per-step wall-clock timing are touched.

For each episode processed, the existing step4 row(s) in that file's Time
table are updated in place with the new elapsed seconds and a fresh
recorded_at timestamp (a step4 row is inserted if the episode doesn't have
one yet). step1/step2/step3 rows - and every other episode's rows - are
never touched.

The fuzzy threshold for each file is taken from its filename prefix
(FUZZY88_... -> 88, FUZZY90_... -> 90, etc.) instead of a command-line flag,
since that's how these snapshots are already named/organized.

Every credit is reprocessed (force_reprocess=True): these files are already
fully processed from a previous run, and the point of this script is to
refresh those results against the CURRENT matching logic (unified
normalize_name pipeline, with-nickname disambiguation, regenerated parquet) -
without force_reprocess, get_unprocessed_credits() would return nothing,
since its default WHERE clause skips rows that already have an
assigned_code/code_assignment_status.

Each .db file is backed up to <name>.db.pre_redo_imdbization.bak (only if
that backup doesn't already exist) before being modified.

Usage:
    python scripts_v3/redo_imdbization_all_dbs.py
    python scripts_v3/redo_imdbization_all_dbs.py db/FUZZY90_SOME_MODEL_tvcredits_v3.db [more_paths ...]
"""

import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts_v3 import config
from scripts_v3.imdb_batch_validation import IMDBBatchValidatorWithCodeAssignment

DB_DIR = Path(__file__).parent.parent / "db"
FUZZY_PREFIX_RE = re.compile(r"^FUZZY(\d+)_")
TIME_TABLE = "Time"
STEP4 = "step4"


def find_threshold(db_path: Path) -> Optional[int]:
    m = FUZZY_PREFIX_RE.match(db_path.name)
    return int(m.group(1)) if m else None


def _has_time_table(db_path: Path) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TIME_TABLE,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _update_step4_timing(db_path: Path, episode_id: str, seconds: float) -> None:
    """Updates this episode's existing step4 row(s) in the Time table with
    the new elapsed seconds and a fresh recorded_at timestamp - inserts one
    if the episode has none yet. Only ever touches step4 rows for this
    episode_id; step1/step2/step3 and every other episode are untouched."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT id FROM "{TIME_TABLE}" WHERE episode_id = ? AND step = ?',
            (episode_id, STEP4),
        )
        existing_ids = [row[0] for row in cursor.fetchall()]
        if existing_ids:
            cursor.executemany(
                f'UPDATE "{TIME_TABLE}" SET seconds = ?, recorded_at = CURRENT_TIMESTAMP WHERE id = ?',
                [(seconds, row_id) for row_id in existing_ids],
            )
        else:
            cursor.execute(
                f'INSERT INTO "{TIME_TABLE}" (episode_id, step, provider, seconds) VALUES (?, ?, NULL, ?)',
                (episode_id, STEP4, seconds),
            )
        conn.commit()
    finally:
        conn.close()


def redo_one_db(db_path: Path, threshold: int) -> None:
    print(f"\n{db_path.name} (fuzzy threshold={threshold})")

    if not _has_time_table(db_path):
        print(f"  [SKIP] no '{TIME_TABLE}' table in this file")
        return

    backup_path = db_path.with_suffix(db_path.suffix + ".pre_redo_imdbization.bak")
    if not backup_path.exists():
        backup_path.write_bytes(db_path.read_bytes())

    # imdb_batch_validation.py always reads config.DB_PATH fresh at call
    # time (never caches it in a local/module constant), so pointing it at
    # THIS file makes every read/write below - via the app's own classes -
    # operate on this specific snapshot.
    config.DB_PATH = db_path

    conn = sqlite3.connect(db_path)
    episode_ids = [row[0] for row in conn.execute("SELECT DISTINCT episode_id FROM credits").fetchall()]
    conn.close()
    print(f"  {len(episode_ids)} episode(s)")

    validator = IMDBBatchValidatorWithCodeAssignment(fuzzy_enabled=True, fuzzy_threshold=threshold)
    total_processed = 0

    for episode_id in episode_ids:
        credits = validator.get_unprocessed_credits(episode_id=episode_id, force_reprocess=True)
        if not credits:
            continue

        start = time.time()
        validator.process_credits_fast(credits)
        elapsed = time.time() - start

        _update_step4_timing(db_path, episode_id, elapsed)
        total_processed += len(credits)
        print(f"    {episode_id}: {len(credits)} credit(s) in {elapsed:.2f}s")

    # process_credits_fast() overwrites (not accumulates) stats['total_credits']
    # on every call - fix it up to the true per-file total before printing.
    validator.stats['total_credits'] = total_processed
    validator.print_final_stats()


def main() -> None:
    original_db_path = config.DB_PATH
    if len(sys.argv) > 1:
        db_files = sorted(Path(arg) for arg in sys.argv[1:])
        missing = [p for p in db_files if not p.exists()]
        if missing:
            print(f"File(s) not found: {[str(p) for p in missing]}")
            return
    else:
        db_files = sorted(DB_DIR.glob("*.db"))
    matched = [(p, find_threshold(p)) for p in db_files]
    unmatched = [p for p, t in matched if t is None]
    matched = [(p, t) for p, t in matched if t is not None]

    if unmatched:
        print(f"Skipping {len(unmatched)} file(s) without a FUZZY##_ prefix: {[p.name for p in unmatched]}")

    if not matched:
        print(f"No FUZZY##_*.db files found in {DB_DIR}")
        return

    print(f"Found {len(matched)} database file(s) to consider in {DB_DIR}")
    try:
        for db_path, threshold in matched:
            try:
                redo_one_db(db_path, threshold)
            except Exception as e:
                print(f"  [ERROR] {db_path.name}: {e}")
    finally:
        config.DB_PATH = original_db_path


if __name__ == "__main__":
    main()
