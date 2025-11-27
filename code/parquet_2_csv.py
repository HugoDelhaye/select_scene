"""
Script to convert Parquet files to CSV format.

This script uses pyarrow to read Parquet files and converts them to CSV format
with proper type handling.
"""

import os
import pandas as pd
import pyarrow.parquet as pq


def read_parquet_to_pandas(parquet_path):
    """
    Read Parquet file using pyarrow and convert to pandas DataFrame.

    Args:
        parquet_path: Path to Parquet file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas(types_mapper=None)
    return df


def save_as_csv(df, csv_path):
    """
    Save dataframe as CSV file.

    Args:
        df: Dataframe to save
        csv_path: Output path for CSV file
    """
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")


def convert_file(parquet_path, csv_path=None):
    """
    Convert a Parquet file to CSV format.

    Args:
        parquet_path: Path to input Parquet file
        csv_path: Path to output CSV file. If None, uses same name with .csv extension
    """
    if csv_path is None:
        csv_path = parquet_path.replace('.parquet', '.csv')

    print(f"Reading {parquet_path}...")
    df = read_parquet_to_pandas(parquet_path)

    print(f"Converting to CSV...")
    save_as_csv(df, csv_path)


def main():
    """
    Main function to convert clips metadata Parquet files to CSV.
    """
    # Convert imitation model metadata
    print("Converting imitation model metadata...")
    convert_file(
        "sourcedata/clips_metadata_from_im.parquet",
        "sourcedata/clips_metadata_from_im.csv"
    )

    # Convert human/PPO metadata
    print("\nConverting human/PPO metadata...")
    convert_file(
        "sourcedata/clips_metadata_with_patterns.parquet",
        "sourcedata/clips_metadata_with_patterns.csv"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
