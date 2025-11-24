import pandas as pd

df = pd.read_parquet('sourcedata/df_metrics.parquet')

df['scene'] = df['scene_full_name'].str[4:].astype(int)
print(df.head(20))

df.to_parquet('sourcedata/df_metrics.parquet', index=False)