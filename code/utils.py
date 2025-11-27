"""
Utility functions for computing learning metrics.

This module contains functions for calculating learning deltas, trajectory
variability (MAD), and other metrics used in the scene selection analysis.
"""

import numpy as np
import pandas as pd


def mad(values):
    """
    Calculate Median Absolute Deviation (MAD).

    Args:
        values: Array-like of numeric values

    Returns:
        float: Median absolute deviation
    """
    values = np.array(values)
    return np.median(np.abs(values - np.median(values)))


def get_mads(df):
    """
    Compute MAD of y-position for each x-position.

    This function calculates trajectory variability by computing the MAD
    of vertical positions (y) at each horizontal position (x).

    Args:
        df: DataFrame with player_x_pos and player_y_pos columns

    Returns:
        pd.DataFrame: DataFrame with MAD_mean column
    """
    # Explode position arrays
    df_exp = df.explode(['player_x_pos', 'player_y_pos'])

    # Calculate MAD of y-position for each x-position
    df_mad = (
        df_exp
        .groupby(['Subject', 'Learning_Phase', 'player_x_pos'])['player_y_pos']
        .apply(lambda g: mad(g.tolist()))
        .reset_index(name='MAD_y_for_x')
    )

    # Average MAD across all x-positions
    df_mad_mean = (
        df_mad
        .groupby(['Subject', 'Learning_Phase'])['MAD_y_for_x']
        .mean()
        .reset_index(name='MAD_mean')
    )

    return df_mad_mean


def compute_deltas(df, phase_to_ckpt, metric):
    """
    Compute sequential deltas between learning phase checkpoints.

    Note: This function is currently unused but kept for potential future use.

    Args:
        df: DataFrame with Subject, Learning_Phase, and metric columns
        phase_to_ckpt: Dictionary mapping learning phases to checkpoint numbers
        metric: Name of the metric column to compute deltas for

    Returns:
        pd.DataFrame: Wide-format DataFrame with delta values
    """
    df_tmp = df.copy()
    df_tmp["ckpt"] = df_tmp["Learning_Phase"].map(phase_to_ckpt)

    df_ckpt = (
        df_tmp.groupby(["Subject", "ckpt"])[metric]
        .mean()
        .reset_index(name=metric)
    )

    df_wide = df_ckpt.pivot(index="Subject", columns="ckpt", values=metric)
    df_shifted = df_wide.shift(-1, axis=1)
    df_delta = df_shifted - df_wide
    df_delta_shifted = df_delta.shift(1, axis=1)

    return df_delta_shifted


def compute_delta_tot(df, phase_to_ckpt, metric):
    """
    Compute total delta (max - min) for a metric across learning phases.

    Args:
        df: DataFrame with subject, learning_phase, and metric columns
        phase_to_ckpt: Dictionary mapping learning phases to checkpoint order
        metric: Name of the metric column to compute delta for

    Returns:
        pd.DataFrame: DataFrame with min, max, and total delta values
    """
    df_tmp = df.copy()
    df_tmp["ckpt"] = df_tmp["learning_phase"].map(phase_to_ckpt)
    df_ckpt = (
        df_tmp.groupby(["subject", "ckpt"])[metric]
        .mean()
        .reset_index(name=metric)
    )

    # Handle case where metric is mostly NaN
    if df_ckpt[metric].isna().sum() >= len(df_ckpt) - 1:
        return pd.DataFrame({
            metric + "_min": np.nan,
            metric + "_max": np.nan,
            "delta_tot": np.nan
        }, index=[0])

    # Remove NaN values
    df_ckpt = df_ckpt.dropna()

    # Find min and max checkpoints
    idx_max = df_ckpt.groupby("subject")["ckpt"].idxmax()
    df_max = df_ckpt.loc[idx_max, ["subject", metric]].rename(
        columns={metric: metric + "_max"}
    )

    idx_min = df_ckpt.groupby("subject")["ckpt"].idxmin()
    df_min = df_ckpt.loc[idx_min, ["subject", metric]].rename(
        columns={metric: metric + "_min"}
    )

    # Compute delta
    results = df_min.merge(df_max, on="subject")
    results['delta_tot'] = results[metric + "_max"] - results[metric + "_min"]

    return results


def order_learning_phases(order_phase, subject):
    """
    Order learning phases according to subject type.

    Args:
        order_phase: Array of phase names to order
        subject: Subject identifier

    Returns:
        list: Ordered list of learning phases
    """
    if subject.startswith('im'):
        # Imitation models: first phase ends with '=500', then sorted
        first_phase = [p for p in order_phase if p.endswith('=500')]
        rest = sorted([p for p in order_phase if not p.endswith('=500')])
        ordered = [first_phase[0]] + rest

        # Special case for im_sub-02
        if subject == 'im_sub-02':
            phase_to_move = 'sub-02_epoch=0-step=10000'
            if phase_to_move in ordered:
                ordered.remove(phase_to_move)
                ordered.append(phase_to_move)

        return ordered

    elif subject.startswith('sub-'):
        # Human subjects: fixed order
        return ['Early discovery', 'Late discovery',
                'Early practice', 'Late practice']

    else:
        # PPO or other: return as-is
        return list(order_phase)


def compute_deltas_for_group(df_group):
    """
    Compute all delta metrics for a subject-scene group.

    This function is the main entry point for computing learning deltas.
    It handles phase ordering and computes deltas for cleared rate, speed,
    and MAD metrics.

    Args:
        df_group: DataFrame group for a single subject-scene combination

    Returns:
        pd.DataFrame: Group dataframe with added delta columns
    """
    sub = df_group['subject'].iloc[0]

    # Order the learning phases
    order_phase = df_group['learning_phase'].unique()
    ordered_phases = order_learning_phases(order_phase, sub)

    # Create checkpoint mapping
    phase_to_ckpt = {phase: i for i, phase in enumerate(ordered_phases)}

    # Compute delta metrics
    delta_clr = compute_delta_tot(df_group, phase_to_ckpt, 'cleared')['delta_tot'].values[0]
    delta_spd = compute_delta_tot(df_group, phase_to_ckpt, 'speed')['delta_tot'].values[0]
    delta_mad = compute_delta_tot(df_group, phase_to_ckpt, 'MAD_mean')['delta_tot'].values[0]

    # Add deltas to the group dataframe
    df_res = df_group.copy()
    df_res['delta_clr_tot'] = delta_clr
    df_res['delta_spd_tot'] = delta_spd
    df_res['delta_MAD_tot'] = delta_mad

    return df_res
