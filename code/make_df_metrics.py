"""
Script to create the main metrics dataframe from raw gameplay data.

This script processes metadata and variable data from human, imitation learning,
and PPO sources to compute learning metrics for each scene.
"""

import os
import pandas as pd
import utils


def load_data(sourcedata_dir='sourcedata'):
    """
    Load all required parquet files.

    Args:
        sourcedata_dir: Directory containing source data files

    Returns:
        tuple: (df_variable_hum, df_variable_ppo, df_meta_im, df_meta_hum_ppo)
    """
    df_variable_hum = pd.read_parquet(os.path.join(sourcedata_dir, 'df_variables_hum.parquet'))
    df_variable_ppo = pd.read_parquet(os.path.join(sourcedata_dir, 'df_variables_ppo.parquet'))
    df_meta_im = pd.read_parquet(os.path.join(sourcedata_dir, 'clips_metadata_from_im.parquet'))
    df_meta_hum_ppo = pd.read_parquet(os.path.join(sourcedata_dir, 'clips_metadata_with_patterns.parquet'))

    return df_variable_hum, df_variable_ppo, df_meta_im, df_meta_hum_ppo


def preprocess_metadata(df_meta_hum_ppo):
    """
    Preprocess metadata by converting 'Cleared' column to binary.

    Args:
        df_meta_hum_ppo: Metadata dataframe for human/PPO subjects

    Returns:
        pd.DataFrame: Preprocessed metadata
    """
    df_meta_hum_ppo = df_meta_hum_ppo.copy()
    df_meta_hum_ppo['Cleared'] = df_meta_hum_ppo['Cleared'].map({'True': 1, 'False': 0})
    return df_meta_hum_ppo


def convert_column_types(df_meta, df_variables):
    """
    Convert data types for key columns.

    Args:
        df_meta: Metadata dataframe
        df_variables: Variables dataframe

    Returns:
        tuple: (df_meta, df_variables) with corrected types
    """
    df_meta = df_meta.copy()
    df_variables = df_variables.copy()

    # Metadata type conversions
    df_meta['Scene'] = df_meta['Scene'].astype(int)
    df_meta['Average_speed'] = df_meta['Average_speed'].astype(float)
    df_meta['Duration'] = df_meta['Duration'].astype(float)
    df_meta['X_Traveled'] = df_meta['X_Traveled'].astype(float)

    # Variables type conversions
    df_variables['Scene'] = df_variables['Scene'].astype(int)

    return df_meta, df_variables


def complete_imitation_columns(df_meta):
    """
    Complete missing columns for imitation model subjects.

    Args:
        df_meta: Metadata dataframe

    Returns:
        pd.DataFrame: Metadata with completed imitation model columns
    """
    df_meta = df_meta.copy()
    mask_im = df_meta['Model'].str.startswith('sub-0')

    df_meta.loc[mask_im, 'Learning_Phase'] = df_meta.loc[mask_im, 'Model'].str[:]
    df_meta.loc[mask_im, 'Subject'] = "im_" + df_meta.loc[mask_im, 'Model'].str[:6]
    df_meta.loc[mask_im, 'SceneFullName'] = (
        df_meta.loc[mask_im, 'World'].astype(str) + '-' +
        df_meta.loc[mask_im, 'Level'].astype(str) + '-' +
        df_meta.loc[mask_im, 'Scene'].astype(str)
    )

    return df_meta


def complete_ppo_columns(df_meta):
    """
    Complete missing columns for PPO agent.

    Args:
        df_meta: Metadata dataframe

    Returns:
        pd.DataFrame: Metadata with completed PPO columns
    """
    df_meta = df_meta.copy()
    mask_ppo = df_meta['Model'].str.startswith('ep')

    df_meta.loc[mask_ppo, 'Learning_Phase'] = df_meta.loc[mask_ppo, 'Model'].str[:]
    df_meta.loc[mask_ppo, 'Subject'] = "ppo"
    df_meta.loc[mask_ppo, 'Average_speed'] = (
        df_meta.loc[mask_ppo, 'X_Traveled'] / df_meta.loc[mask_ppo, 'Duration']
    )

    return df_meta


def compute_player_x_position(df_variables):
    """
    Compute full x position from high and low bytes.

    Args:
        df_variables: Variables dataframe

    Returns:
        pd.DataFrame: Variables with computed player_x_pos column
    """
    df_variables = df_variables.copy()
    df_variables['player_x_pos'] = (
        df_variables['player_x_posHi'] * 255 + df_variables['player_x_posLo']
    )
    return df_variables


def filter_scenes_by_completion(df_meta, df_variables, min_subjects=5):
    """
    Filter out scenes not completed by the minimum number of human subjects.

    Args:
        df_meta: Metadata dataframe
        df_variables: Variables dataframe
        min_subjects: Minimum number of subjects required

    Returns:
        tuple: (df_meta, df_variables) filtered by scene completion
    """
    df_meta = df_meta.copy()
    df_variables = df_variables.copy()

    # Identify scenes completed by fewer than min_subjects
    df_filter = df_meta[df_meta['Subject'].str.startswith('sub-')].groupby(['SceneFullName'])
    scenes_to_drop = {}

    for scene, df_scene in df_filter:
        if df_scene['Subject'].nunique() < min_subjects:
            scene_name = scene if isinstance(scene, str) else scene[0]
            scenes_to_drop[scene_name] = [len(df_scene)]

    # Filter out incomplete scenes
    df_meta = df_meta[~df_meta['SceneFullName'].isin(scenes_to_drop)]
    df_variables = df_variables[~df_variables['SceneFullName'].isin(scenes_to_drop)]

    return df_meta, df_variables


def create_metrics_structure(df_meta):
    """
    Create the initial metrics dataframe structure.

    Args:
        df_meta: Metadata dataframe

    Returns:
        pd.DataFrame: Initial metrics structure
    """
    cols = ['subject', 'learning_phase', 'scene_full_name',
            'delta_clr_tot', 'delta_spd_tot', 'delta_MAD_tot']
    df_metrics = pd.DataFrame(columns=cols)

    # Set up subject and scene combinations
    df_metrics['subject'] = df_meta['Subject'].unique()
    df_metrics['scene_full_name'] = [
        df_meta['SceneFullName'].unique().tolist()
    ] * df_metrics['subject'].nunique()

    # Add learning phases per subject
    for sub, df_sub in df_metrics.groupby('subject'):
        learning_phases = tuple(df_meta[df_meta['Subject'] == sub]['Learning_Phase'].unique())
        df_metrics.loc[df_metrics['subject'] == sub, 'learning_phase'] = pd.Series(
            [learning_phases],
            index=df_metrics.loc[df_metrics['subject'] == sub].index
        )

    # Explode to get one row per (subject, learning_phase, scene)
    df_metrics = df_metrics.explode('learning_phase').explode('scene_full_name').reset_index(drop=True)

    return df_metrics


def aggregate_metadata(df_meta):
    """
    Aggregate metadata by subject, learning phase, and scene.

    Args:
        df_meta: Metadata dataframe

    Returns:
        pd.DataFrame: Aggregated metadata
    """
    df_meta_agg = (
        df_meta
        .groupby(['Subject', 'Learning_Phase', 'SceneFullName'])
        .agg(
            count=('Cleared', 'size'),
            cleared=('Cleared', 'mean'),
            speed=('Average_speed', 'mean')
        )
        .reset_index()
    )

    return df_meta_agg


def aggregate_variables(df_variables):
    """
    Aggregate variables to compute MAD statistics.

    Args:
        df_variables: Variables dataframe

    Returns:
        pd.DataFrame: Aggregated variables with MAD metrics
    """
    df_vars_agg = (
        df_variables
        .groupby(['Subject', 'Learning_Phase', 'SceneFullName'])
        .apply(lambda df: pd.Series({'MAD_mean': utils.get_mads(df)['MAD_mean'].mean()}))
        .reset_index()
    )

    return df_vars_agg


def merge_aggregated_data(df_metrics, df_meta_agg, df_vars_agg):
    """
    Merge metadata and variable aggregations into metrics dataframe.

    Args:
        df_metrics: Base metrics dataframe
        df_meta_agg: Aggregated metadata
        df_vars_agg: Aggregated variables

    Returns:
        pd.DataFrame: Merged metrics dataframe
    """
    # Merge metadata aggregations
    df_agg = df_meta_agg.merge(
        df_vars_agg,
        on=['Subject', 'Learning_Phase', 'SceneFullName'],
        how='left'
    )

    # Rename columns to match metrics structure
    df_agg = df_agg.rename(columns={
        'Subject': 'subject',
        'Learning_Phase': 'learning_phase',
        'SceneFullName': 'scene_full_name'
    })

    # Merge into metrics
    df_metrics = df_metrics.merge(
        df_agg,
        on=['subject', 'learning_phase', 'scene_full_name'],
        how='left'
    )

    return df_metrics


def compute_learning_deltas(df_metrics):
    """
    Compute delta metrics showing learning progress.

    Args:
        df_metrics: Metrics dataframe

    Returns:
        pd.DataFrame: Metrics with computed deltas
    """
    df_metrics = (
        df_metrics
        .groupby(['subject', 'scene_full_name'], group_keys=False)
        .apply(utils.compute_deltas_for_group)
    )

    return df_metrics


def add_derived_columns(df_metrics):
    """
    Add derived columns for level and scene identifiers.

    Args:
        df_metrics: Metrics dataframe

    Returns:
        pd.DataFrame: Metrics with derived columns
    """
    df_metrics = df_metrics.copy()

    df_metrics['level_full_name'] = (
        'w' + df_metrics['scene_full_name'].str[0] +
        'l' + df_metrics['scene_full_name'].str[2]
    )
    df_metrics['scene'] = df_metrics['scene_full_name'].str[4:].astype(int)

    return df_metrics


def save_metrics(df_metrics, output_dir='sourcedata', filename='df_metrics.parquet'):
    """
    Save metrics dataframe to parquet file.

    Args:
        df_metrics: Metrics dataframe to save
        output_dir: Output directory
        filename: Output filename
    """
    output_path = os.path.join(output_dir, filename)
    df_metrics.to_parquet(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def main():
    """
    Main pipeline to create metrics dataframe from raw data.
    """
    # Load data
    print("Loading data...")
    df_variable_hum, df_variable_ppo, df_meta_im, df_meta_hum_ppo = load_data()

    # Preprocess
    print("Preprocessing metadata...")
    df_meta_hum_ppo = preprocess_metadata(df_meta_hum_ppo)

    # Combine datasets
    print("Combining datasets...")
    df_variables = pd.concat([df_variable_hum, df_variable_ppo], axis=0)
    df_meta = pd.concat([df_meta_im, df_meta_hum_ppo], axis=0)

    # Convert types
    print("Converting data types...")
    df_meta, df_variables = convert_column_types(df_meta, df_variables)

    # Complete missing columns
    print("Completing metadata columns...")
    df_meta = complete_imitation_columns(df_meta)
    df_meta = complete_ppo_columns(df_meta)

    # Compute player position
    print("Computing player positions...")
    df_variables = compute_player_x_position(df_variables)

    # Filter scenes
    print("Filtering scenes by completion...")
    df_meta, df_variables = filter_scenes_by_completion(df_meta, df_variables)

    # Create metrics structure
    print("Creating metrics structure...")
    df_metrics = create_metrics_structure(df_meta)

    # Aggregate data
    print("Aggregating metadata...")
    df_meta_agg = aggregate_metadata(df_meta)

    print("Aggregating variables...")
    df_vars_agg = aggregate_variables(df_variables)

    # Merge aggregations
    print("Merging aggregated data...")
    df_metrics = merge_aggregated_data(df_metrics, df_meta_agg, df_vars_agg)

    # Compute deltas
    print("Computing learning deltas...")
    df_metrics = compute_learning_deltas(df_metrics)

    # Add derived columns
    print("Adding derived columns...")
    df_metrics = add_derived_columns(df_metrics)

    # Save results
    print("Saving metrics...")
    save_metrics(df_metrics)

    print("Done!")


if __name__ == "__main__":
    main()
