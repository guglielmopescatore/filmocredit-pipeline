#!/usr/bin/env python3
"""
Script to recreate the IMDB parquet file with all necessary fields.
This ensures the parquet file has primaryProfession and other required fields.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import scripts_v3
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts_v3 import config
from scripts_v3.utils import normalize_name


def recreate_imdb_parquet():
    """
    Recreate the IMDB parquet file with all necessary fields including primaryProfession.
    """
    input_path = config.IMDB_TSV_PATH
    parquet_output_path = config.IMDB_PARQUET_PATH

    print(f"🔍 Loading IMDb data from {input_path}...")
    
    if not input_path.exists():
        print(f"❌ Error: IMDB TSV file not found at {input_path}")
        print("Please ensure the name.basics.tsv file is in the db/ directory")
        return False
    
    try:
        df = pd.read_csv(input_path, sep='\t', dtype=str, na_values='\\N')
        print(f"✅ Loaded {len(df):,} records from IMDB TSV file")
    except Exception as e:
        print(f"❌ Error loading TSV file: {e}")
        return False

    print("🔄 Normalizing names...")
    df['normalizedName'] = df['primaryName'].apply(normalize_name)

    # Include all necessary columns for profession matching
    df_out = df[['nconst', 'normalizedName', 'primaryName', 'primaryProfession', 'birthYear', 'deathYear']]
    
    print(f"📊 Final dataset has {len(df_out):,} records with columns:")
    for col in df_out.columns:
        non_null_count = df_out[col].notna().sum()
        print(f"  - {col}: {non_null_count:,} non-null values")

    # Ensure output directory exists
    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving processed data to {parquet_output_path}...")
    try:
        df_out.to_parquet(parquet_output_path, index=False)
        print("✅ Parquet file created successfully!")
        
        # Verify the file was created and has the right size
        if parquet_output_path.exists():
            file_size = parquet_output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.2f} MB")
            
            # Test reading the file to ensure it's valid
            test_df = pd.read_parquet(parquet_output_path)
            print(f"✅ Verification: Successfully read {len(test_df):,} records from parquet file")
            print(f"✅ Columns in parquet: {list(test_df.columns)}")
            
            return True
        else:
            print("❌ Error: Parquet file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error saving parquet file: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting IMDB parquet file recreation...")
    success = recreate_imdb_parquet()
    
    if success:
        print("\n🎉 IMDB parquet file recreation completed successfully!")
        print("The parquet file now contains all necessary fields including primaryProfession.")
    else:
        print("\n❌ IMDB parquet file recreation failed!")
        sys.exit(1) 