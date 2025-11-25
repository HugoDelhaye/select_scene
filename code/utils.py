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
    df_tmp["ckpt"] = df_tmp["learning_phase"].map(phase_to_ckpt)
    df_ckpt =df_tmp.groupby(["subject", "ckpt"])[metric].mean().reset_index(name=metric)

    if df_ckpt[metric].isna().sum() >= len(df_ckpt)-1:
        return pd.DataFrame({metric+"_min":np.nan
                            , metric+"_max": np.nan, 
                            "delta_tot": np.nan}, index=[0])
    
    df_ckpt = df_ckpt.dropna()
    idx_max = df_ckpt.groupby("subject")["ckpt"].idxmax()
    df_max = df_ckpt.loc[idx_max, ["subject", metric]].rename(columns={metric: metric+"_max"})    
    idx_min = df_ckpt.groupby("subject")["ckpt"].idxmin()
    df_min = df_ckpt.loc[idx_min, ["subject", metric]].rename(columns={metric: metric+"_min"})

    results = df_min.merge(df_max, on="subject")
    results['delta_tot'] = results[metric+"_max"] - results[metric+"_min"]

    
    return results

def compute_deltas_for_group(df_group):
    sub = df_group['subject'].iloc[0]

    # ---- ORDERING THE PHASES ----
    order_phase = df_group['learning_phase'].unique()

    if sub.startswith('im'):
        first_phase = [p for p in order_phase if p.endswith('=500')]
        rest = sorted([p for p in order_phase if not p.endswith('=500')])

        order_phase = [first_phase[0]] + rest

        if sub == 'im_sub-02':
            order_phase.remove('sub-02_epoch=0-step=10000')
            order_phase.append('sub-02_epoch=0-step=10000')

    elif sub.startswith('sub-'):
        order_phase = ['Early discovery', 'Late discovery',
                       'Early practice', 'Late practice']

    # ---- CKPT MAPPING ----
    phase_to_ckpt = {phase: i for i, phase in enumerate(order_phase)}

    # ---- DELTA CALCULATIONS ----
    delta_clr = compute_delta_tot(df_group, phase_to_ckpt, 'cleared')['delta_tot'].values[0]
    delta_spd = compute_delta_tot(df_group, phase_to_ckpt, 'speed')['delta_tot'].values[0]
    delta_mad = compute_delta_tot(df_group, phase_to_ckpt, 'MAD_mean')['delta_tot'].values[0]

    # ---- RETURN A CLEAN DF ----
    df_res = df_group.copy()
    df_res['delta_clr_tot'] = delta_clr
    df_res['delta_spd_tot'] = delta_spd
    df_res['delta_MAD_tot'] = delta_mad

    return df_res
