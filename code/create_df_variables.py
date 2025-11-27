"""
Script to extract PPO player position variables from JSON files.

This script reads JSON files containing frame-by-frame player position data
for PPO agent gameplay clips and creates a structured dataframe.
"""

import glob
import json
import os
from pathlib import Path
import pandas as pd


def get_json_file_list(base_path=None):
    """
    Get list of JSON files containing PPO variables.

    Args:
        base_path: Base path to search for files. If None, uses default path.

    Returns:
        list: List of file paths
    """
    if base_path is None:
        home = str(Path.home())
        base_path = os.path.join(
            home, 'Téléchargements', 'mario_learning-sourcedata',
            'sourcedata', 'ppo_*', 'sub-*', 'ses-*', 'beh', 'variables', '*.json'
        )

    file_list = glob.glob(base_path)
    return file_list


def create_dataframe_structure(file_list):
    """
    Create initial dataframe structure with file paths.

    Args:
        file_list: List of JSON file paths

    Returns:
        pd.DataFrame: Initial dataframe with json_path column
    """
    columns = (
        'Subject', 'World', 'Level', 'Scene', 'LevelFullName',
        'SceneFullName', 'ClipCode', 'Learning_Phase',
        'player_x_posHi', 'player_x_posLo', 'player_y_pos', 'json_path'
    )

    df_variables = pd.DataFrame(columns=columns)
    df_variables['json_path'] = file_list
    df_variables['Subject'] = 'ppo'

    return df_variables


def extract_metadata_from_paths(df_variables):
    """
    Extract metadata from file paths using regex patterns.

    Args:
        df_variables: Dataframe with json_path column

    Returns:
        pd.DataFrame: Dataframe with extracted metadata columns
    """
    df_variables = df_variables.copy()

    # Extract world, level, scene, clip from filename
    df_variables[['World', 'Level', 'Scene', 'ClipCode']] = (
        df_variables['json_path'].str.extract(
            r'level-w(\d+)l(\d+)_scene-(\d+)_clip-(\d+).json'
        )
    )

    # Create composite identifiers
    df_variables['LevelFullName'] = 'w' + df_variables['World'] + 'l' + df_variables['Level']
    df_variables['SceneFullName'] = (
        df_variables['World'] + '-' +
        df_variables['Level'] + '-' +
        df_variables['Scene']
    )

    # Extract learning phase from path
    df_variables['Learning_Phase'] = (
        df_variables['json_path'].str.extract(r'ppo_mario_(ep-\w+)')
    )

    return df_variables


def extract_position_data(df_variables):
    """
    Read JSON files and extract player position data.

    Args:
        df_variables: Dataframe with json_path column

    Returns:
        pd.DataFrame: Dataframe with position data columns filled
    """
    df_variables = df_variables.copy()

    player_x_hi = []
    player_x_lo = []
    player_y = []

    for path in df_variables['json_path']:
        with open(path, "r") as f:
            data = json.load(f)

        player_x_hi.append(data["player_x_posHi"])
        player_x_lo.append(data["player_x_posLo"])
        player_y.append(data["player_y_pos"])

    df_variables['player_x_posHi'] = player_x_hi
    df_variables['player_x_posLo'] = player_x_lo
    df_variables['player_y_pos'] = player_y

    return df_variables


def save_variables(df_variables, output_dir='sourcedata', base_filename='ppo_clips_variables'):
    """
    Save variables dataframe to CSV and Parquet formats.

    Args:
        df_variables: Dataframe to save
        output_dir: Output directory
        base_filename: Base filename (without extension)
    """
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")

    df_variables.to_csv(csv_path, index=False)
    df_variables.to_parquet(parquet_path)

    print(f"Variables saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Parquet: {parquet_path}")


def main():
    """
    Main pipeline to extract PPO variables from JSON files.
    """
    print("Getting JSON file list...")
    file_list = get_json_file_list()
    print(f"Found {len(file_list)} JSON files")

    print("Creating dataframe structure...")
    df_variables = create_dataframe_structure(file_list)

    print("Extracting metadata from file paths...")
    df_variables = extract_metadata_from_paths(df_variables)

    print("Reading position data from JSON files...")
    df_variables = extract_position_data(df_variables)

    print("Saving variables...")
    save_variables(df_variables)

    print("Done!")


if __name__ == "__main__":
    main()
