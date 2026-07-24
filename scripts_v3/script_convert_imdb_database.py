#!/usr/bin/env python3
"""
Script to convert IMDB name.basics.tsv to Parquet format with normalized
names and name+profession search combinations.

Uses the SAME normalization pipeline (scripts_v3.utils) as everything else in
the project - a previous "standalone, no dependencies" duplicate here drifted
out of sync (its ASCII-only whitelist silently deleted non-ASCII letters like
German "ss" or Icelandic "eth", instead of just stripping accents), which
broke IMDB matching for any name containing them. Importing the real
functions instead of re-implementing them is what prevents that drift.

Usage:
    python script_convert_imdb_database.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts_v3.utils import normalize_name as _normalize_name_base
from scripts_v3.utils import normalize_name_with_nickname as _normalize_name_with_nickname_base


def normalize_name(name: str) -> str:
    """Normalize a name for IMDB matching - thin wrapper around the shared,
    is_person-aware utils.normalize_name (name.basics data is always people)
    that turns empty/missing names into "" instead of "none"."""
    if not name or not isinstance(name, str):
        return ""
    return _normalize_name_base(name, is_person=True)


def normalize_name_with_nickname(name: str):
    """Like normalize_name() above, but for the with-nickname companion
    column - thin wrapper around utils.normalize_name_with_nickname
    (name.basics data is always people). Returns None when there's no
    quoted nickname aside in `name` (e.g. plain "Roy Moore", vs. "Roy
    'Bucky' Moore" -> 'roy "bucky" moore')."""
    if not name or not isinstance(name, str):
        return None
    return _normalize_name_with_nickname_base(name, is_person=True)


def tsv_to_parquet_imdb_normalization():
    """
    Process IMDb name.basics.tsv file and output Parquet file 
    with all necessary columns for IMDB validation and profession matching.
    Includes name+profession combinations for enhanced fuzzy matching.
    """
    # Paths - adjust these if needed
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'db' / 'name.basics.tsv'
    # Write Parquet to the same filename expected by the rest of the project
    # (config.IMDB_PARQUET_PATH -> db/normalized_names.parquet)
    parquet_output_path = project_root / 'db' / 'normalized_names.parquet'
    
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        print(f"Please ensure name.basics.tsv is in the db/ folder")
        return
    
    print(f"📂 Loading IMDb data from {input_path}...")
    print(f"   File size: {input_path.stat().st_size / (1024**3):.2f} GB")
    
    try:
        # Use default C parser - it's faster and more reliable for well-formed TSV files
        df = pd.read_csv(
            input_path, 
            sep='\t', 
            dtype=str, 
            na_values='\\N',
            low_memory=False,
            encoding='utf-8'
        )
        print(f"✅ Loaded {len(df):,} records")
    except Exception as e:
        print(f"❌ Error loading TSV file with default parser: {e}")
        print(f"\nTrying with Python engine (slower but more robust)...")
        try:
            # Fallback to Python engine without incompatible options
            df = pd.read_csv(
                input_path,
                sep='\t',
                dtype=str,
                na_values='\\N',
                engine='python',
                on_bad_lines='skip',
                encoding='utf-8'
            )
            print(f"✅ Loaded {len(df):,} records (some malformed lines may have been skipped)")
        except Exception as py_error:
            print(f"❌ Python engine also failed: {py_error}")
            print(f"\nThe TSV file appears to be corrupted.")
            print(f"Please download a fresh copy from: https://datasets.imdbws.com/name.basics.tsv.gz")
            return

    print("🔄 Normalizing names...")
    df['normalizedName'] = df['primaryName'].apply(normalize_name)
    df['normalizedNameWithNickname'] = df['primaryName'].apply(normalize_name_with_nickname)

    print("🔍 Creating name+profession search combinations...")
    def create_search_combinations(row):
        """Create all possible name+profession combinations for fuzzy matching."""
        normalized_name = row['normalizedName']
        if pd.isna(normalized_name) or not normalized_name:
            return []
        
        professions = row['primaryProfession']
        if pd.isna(professions) or not professions:
            # No professions listed - just use name
            return [normalized_name]
        
        # Split professions and create combinations
        prof_list = [p.strip() for p in professions.split(',') if p.strip()]
        combinations = []
        
        for prof in prof_list:
            # Add name + profession combination
            combinations.append(f"{normalized_name} {prof}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_combinations = []
        for combo in combinations:
            if combo not in seen:
                seen.add(combo)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    df['search_combinations'] = df.apply(create_search_combinations, axis=1)
    print(f"✅ Created search combinations for {len(df):,} records")

    # Include all necessary columns for profession matching
    df_out = df[['nconst', 'normalizedName', 'normalizedNameWithNickname', 'primaryName', 'primaryProfession', 'birthYear', 'deathYear', 'search_combinations']]

    print(f"💾 Saving processed data to {parquet_output_path}...")
    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(parquet_output_path, index=False)
    
    file_size_mb = parquet_output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Done! Parquet file saved: {parquet_output_path}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"📝 Records: {len(df_out):,}")
    print(f"🔍 Enhanced fuzzy matching enabled with name+profession combinations")


if __name__ == '__main__':
    tsv_to_parquet_imdb_normalization()
