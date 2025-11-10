import pandas as pd

df = pd.read_csv(
    "sourcedata/clips_metadata_with_patterns.csv",
    dtype=str,          # <- force everything to string
    low_memory=False    # <- avoids partial type guessing
)

df.to_parquet("sourcedata/clips_metadata_with_patterns.parquet")