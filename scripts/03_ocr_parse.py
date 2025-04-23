#!/usr/bin/env python3
"""
03_ocr_parse.py: Parse OCR frames into structured credits JSON and SQLite DB

Usage:
    python 03_ocr_parse.py --episode-id tt1234567 \
                           --lang ita+eng \
                           --conf-min 0.60 \
                           [--verbose]
"""
import argparse
import json
import logging
import re
import string
import sqlite3
import sys
import hashlib
from pathlib import Path

# Ensure Pillow is available
try:
    from PIL import Image
except ImportError:
    print(
        "Error: 'Pillow' is not installed.\n"
        "Please install it by running: pip install pillow\n"
        "or add 'pillow' to requirements.txt and run pip install -r requirements.txt"
    )
    sys.exit(1)

# Ensure pytesseract is available
try:
    import pytesseract
except ImportError:
    print(
        "Error: 'pytesseract' is not installed.\n"
        "Please install it by running: pip install pytesseract\n"
        "or add 'pytesseract' to requirements.txt and run pip install -r requirements.txt"
    )
    sys.exit(1)


def load_role_map(json_path: Path) -> dict:
    """Load role dictionary JSON and build mapping from variant to canonical."""
    data = json.loads(json_path.read_text(encoding='utf-8'))
    role_map = {}
    for canonical, variants in data.items():
        role_map[canonical.lower()] = canonical
        for variant in variants:
            role_map[variant.lower()] = canonical
    return role_map


def detect_role(text: str, role_map: dict) -> str | None:
    """Detect if text starts with any role header variant."""
    txt = text.strip().strip(string.punctuation).lower()
    for key, val in role_map.items():
        if txt.startswith(key):
            return val
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Parse OCR frames into credits"
    )
    parser.add_argument('--episode-id', required=True,
                        help='Episode identifier, e.g., tt1234567')
    parser.add_argument('--lang', default='ita+eng',
                        help='Tesseract languages (e.g., ita+eng)')
    parser.add_argument('--conf-min', type=float, default=0.60,
                        help='Minimum confidence for auto-tagging')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    verbose = args.verbose
    conf_min = args.conf_min
    lang = args.lang
    ep_id = args.episode_id

    # Resolve paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    frames_dir = project_root / 'data' / 'frames' / ep_id
    ocr_dir = project_root / 'data' / 'ocr' / ep_id
    ocr_dir.mkdir(parents=True, exist_ok=True)

    # Load role dictionary
    role_json = script_path.parent / 'ruoli_professiopnali_inglese-italiano.json'
    if not role_json.is_file():
        print(f"Role JSON not found: {role_json}")
        sys.exit(1)
    role_map = load_role_map(role_json)
    if verbose:
        print(f"Loaded {len(role_map)} role variants from {role_json}")

    # OCR raw extraction
    raw = []
    for frame_path in sorted(frames_dir.glob('**/*.jpg')):
        if verbose:
            print(f"OCR frame: {frame_path}")
        img = Image.open(frame_path)
        config = f"--oem 3 --psm 7 -l {lang}"
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config=config
        )
        n = len(data.get('text', []))
        for i in range(n):
            text = data['text'][i].strip()
            if not text:
                continue
            try:
                conf = float(data['conf'][i]) / 100.0
            except Exception:
                conf = 0.0
            raw.append({"text": text, "conf": conf, "frame": str(frame_path)})
    if verbose:
        print(f"Total OCR entries: {len(raw)}")

    # Deduplication
    occurrences = {}
    for entry in raw:
        key = hashlib.sha1(entry['text'].strip().lower().encode('utf-8')).hexdigest()
        if key not in occurrences:
            occurrences[key] = {'first': entry, 'best': entry, 'count': 1}
        else:
            occurrences[key]['count'] += 1
            if entry['conf'] > occurrences[key]['best']['conf']:
                occurrences[key]['best'] = entry
    deduped = []
    for val in occurrences.values():
        if val['count'] >= 3:
            deduped.append(val['best'])
        else:
            deduped.append(val['first'])
    if verbose:
        print(f"Deduped entries: {len(deduped)}")

    # Parse lines into credits
    CAST_RE = re.compile(r"^(.*?)\s*[-:–—]\s*(.+)$")
    credits = []
    seq = 1
    current_role = None
    count_auto = 0
    count_review = 0
    sum_conf = 0.0

    for entry in deduped:
        text = entry['text']
        conf = entry['conf']
        # Role header detection
        role_header = detect_role(text, role_map)
        if role_header:
            current_role = role_header
            if verbose:
                print(f"Detected role header: {current_role}")
            continue

        record = {"seq": seq, "raw_text": text, "conf": conf}
        m = CAST_RE.match(text)
        if m and current_role in ("Actor", "Actress", "Cast"):
            record["person_name"] = m.group(1).strip()
            record["role_detail"] = m.group(2).strip()
            record["role_group"] = "Actor/Actress"
            record["need_review"] = conf < conf_min
        elif current_role:
            record["person_name"] = text.strip()
            record["role_detail"] = None
            record["role_group"] = current_role
            record["need_review"] = conf < conf_min
        else:
            record["person_name"] = None
            record["role_detail"] = None
            record["role_group"] = None
            record["need_review"] = True

        credits.append(record)
        if record["need_review"]:
            count_review += 1
        else:
            count_auto += 1
        sum_conf += conf
        seq += 1

    total = len(credits)

    # Write credits JSON
    credits_path = ocr_dir / 'credits.json'
    credits_path.write_text(
        json.dumps(credits, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    if verbose:
        print(f"Wrote credits JSON to {credits_path}")

    # Upsert into SQLite
    db_path = project_root / 'db' / 'tvcredits.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS episode(
            id TEXT PRIMARY KEY,
            title TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit(
            ep_id TEXT,
            seq INTEGER,
            raw_text TEXT,
            role_group TEXT,
            person_name TEXT,
            role_detail TEXT,
            conf REAL,
            need_review INTEGER,
            PRIMARY KEY(ep_id, seq),
            FOREIGN KEY(ep_id) REFERENCES episode(id)
        )""")
    for rec in credits:
        cur.execute("""
            INSERT OR REPLACE INTO credit
            (ep_id, seq, raw_text, role_group, person_name, role_detail, conf, need_review)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ep_id, rec['seq'], rec['raw_text'], rec['role_group'],
            rec.get('person_name'), rec.get('role_detail'), rec['conf'], int(rec['need_review'])
        ))
    conn.commit()
    conn.close()

    # Summary
    auto_pct = (count_auto / total * 100) if total else 0.0
    review_pct = (count_review / total * 100) if total else 0.0
    avg_conf = (sum_conf / total) if total else 0.0
    print(f"Total lines processed: {total}")
    print(f"Automatically tagged: {auto_pct:.1f}%")
    print(f"Need review: {review_pct:.1f}%")
    print(f"Average confidence: {avg_conf:.2f}")
    print("Precision/Recall estimated: n/a (human review pending)")

if __name__ == '__main__':
    main()
