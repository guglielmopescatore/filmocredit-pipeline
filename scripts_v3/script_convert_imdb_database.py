import pandas as pd
import re
import os
from scripts_v3 import config
from scripts_v3.utils import normalize_name


def tsv_to_parquet_imdb_normalization():
    """
    Process IMDb name.basics.tsv file and output Parquet file 
    with all necessary columns for IMDB validation and profession matching.
    """
    input_path = config.IMDB_TSV_PATH
    parquet_output_path = config.IMDB_PARQUET_PATH

    print(f"Loading IMDb data from {input_path}...")
    df = pd.read_csv(input_path, sep='\t', dtype=str, na_values='\\N')

    print("Normalizing names...")
    df['normalizedName'] = df['primaryName'].apply(normalize_name)

    # Include all necessary columns for profession matching
    df_out = df[['nconst', 'normalizedName', 'primaryName', 'primaryProfession', 'birthYear', 'deathYear']]

    print(f"Saving processed data to {parquet_output_path}...")
    df_out.to_parquet(parquet_output_path, index=False)

    print("Done.")
