import numpy as np
import pandas as pd

def compute_deltas(df, phase_to_ckpt, metric):

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

def mad(values):
    values = np.array(values)
    return np.median(np.abs(values - np.median(values)))

def get_mads(df):

    df_exp = (
        df
        .explode(['player_x_pos', 'player_y_pos'])
    )

    df_mad = (
        df_exp
        .groupby(['Subject', 'Learning_Phase', 'player_x_pos'])['player_y_pos']
        .apply(lambda g: mad(g.tolist()))
        .reset_index(name='MAD_y_for_x')
    )

    df_mad_mean = (
        df_mad
        .groupby(['Subject', 'Learning_Phase'])['MAD_y_for_x']
        .mean()
        .reset_index(name='MAD_mean')
    )

    return df_mad_mean

def compute_delta_tot(df, phase_to_ckpt, metric):

    df_tmp = df.copy()
    df_tmp["ckpt"] = df_tmp["Learning_Phase"].map(phase_to_ckpt)
    df_ckpt =df_tmp.groupby(["Subject", "ckpt"])[metric].mean().reset_index(name=metric)

    idx_max = df_ckpt.groupby("Subject")["ckpt"].idxmax()
    df_max = df_ckpt.loc[idx_max, ["Subject", metric]].rename(columns={metric: metric+"_max"})    
    idx_min = df_ckpt.groupby("Subject")["ckpt"].idxmin()
    df_min = df_ckpt.loc[idx_min, ["Subject", metric]].rename(columns={metric: metric+"_min"})

    results = df_min.merge(df_max, on="Subject")
    results['delta_tot'] = results[metric+"_max"] - results[metric+"_min"]
    return results