import pandas as pd
import pyarrow.parquet as pq

df_im = pq.read_table("sourcedata/clips_metadata_from_im.parquet")
df_im = df_im.to_pandas(types_mapper=None)
df_base = pq.read_table("sourcedata/clips_metadata_with_patterns.parquet")
df_base = df_base.to_pandas(types_mapper=None)

df_im.to_csv("sourcedata/clips_metadata_from_im.csv", index=False)
df_base.to_csv("sourcedata/clips_metadata_with_patterns.csv", index=False)