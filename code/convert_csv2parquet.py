"""
Script to convert CSV files to Parquet format.

This script reads CSV files with proper type handling and converts them to
Parquet format for better performance and storage efficiency.
"""

import os
import pandas as pd


def read_csv_safe(csv_path):
    """
    Read CSV file with safe type handling.

    Forces all columns to string dtype to avoid pandas' type guessing issues.

    Args:
        csv_path: Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(
        csv_path,
        dtype=str,          # Force everything to string
        low_memory=False    # Avoid partial type guessing
    )
    return df


def save_as_parquet(df, parquet_path):
    """
    Save dataframe as Parquet file.

    Args:
        df: Dataframe to save
        parquet_path: Output path for Parquet file
    """
    df.to_parquet(parquet_path)
    print(f"Saved to {parquet_path}")


def convert_file(csv_path, parquet_path=None):
    """
    Convert a CSV file to Parquet format.

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file. If None, uses same name with .parquet extension
    """
    if parquet_path is None:
        parquet_path = csv_path.replace('.csv', '.parquet')

    print(f"Reading {csv_path}...")
    df = read_csv_safe(csv_path)

    print(f"Converting to Parquet...")
    save_as_parquet(df, parquet_path)


def main():
    """
    Main function to convert clips metadata CSV to Parquet.
    """
    csv_path = "sourcedata/clips_metadata_with_patterns.csv"
    parquet_path = "sourcedata/clips_metadata_with_patterns.parquet"

    convert_file(csv_path, parquet_path)
    print("Done!")


if __name__ == "__main__":
    main()
