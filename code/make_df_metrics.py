import pandas as pd
import os
import utils

## load df ##

df_variable_hum = pd.read_parquet(os.path.join('sourcedata', 'df_variables_hum.parquet'))
df_variable_ppo = pd.read_parquet(os.path.join('sourcedata', 'df_variables_ppo.parquet'))
df_meta_im = pd.read_parquet(os.path.join('sourcedata', 'clips_metadata_from_im.parquet'))
df_meta_hum_ppo = pd.read_parquet(os.path.join('sourcedata', 'clips_metadata_with_patterns.parquet'))

df_meta_hum_ppo['Cleared'] = df_meta_hum_ppo['Cleared'].map({'True': 1, 'False': 0})

df_variables = pd.concat([df_variable_hum, df_variable_ppo], axis=0)
df_meta = pd.concat([df_meta_im, df_meta_hum_ppo], axis=0)

# change data types #
df_meta['Scene'] = df_meta['Scene'].astype(int)
df_meta['Average_speed'] = df_meta['Average_speed'].astype(float)
df_meta['Duration'] = df_meta['Duration'].astype(float)
df_meta['X_Traveled'] = df_meta['X_Traveled'].astype(float)

df_variables['Scene'] = df_variables['Scene'].astype(int)

# complete columns
mask_im = df_meta['Model'].str.startswith('sub-0')
df_meta.loc[mask_im, 'Learning_Phase'] = df_meta.loc[mask_im, 'Model'].str[:]
df_meta.loc[mask_im, 'Subject'] = "im_"+ df_meta.loc[mask_im, 'Model'].str[:6]
df_meta.loc[mask_im, 'SceneFullName'] = df_meta.loc[mask_im, 'World'].astype(str)+'-'+df_meta.loc[mask_im, 'Level'].astype(str)+'-'+df_meta.loc[mask_im, 'Scene'].astype(str)

mask_ppo = df_meta['Model'].str.startswith('ep')
df_meta.loc[mask_ppo, 'Learning_Phase'] = df_meta.loc[mask_ppo, 'Model'].str[:]
df_meta.loc[mask_ppo, 'Subject'] = "ppo"
df_meta.loc[mask_ppo, 'Average_speed'] = df_meta.loc[mask_ppo, 'X_Traveled'] / df_meta.loc[mask_ppo, 'Duration']

df_variables['player_x_pos'] = df_variables['player_x_posHi']*255 + df_variables['player_x_posLo']

## filter the scenes completed by the 5 subjects ##
df_filter = df_meta[df_meta['Subject'].str.startswith('sub-')].groupby(['SceneFullName'])
full_scenes = []
scenes_to_drop = {}
for scene, df_scene in df_filter:
    if df_scene['Subject'].nunique() < 5:
        scenes_to_drop[scene[0]] = [len(df_scene)]
    else:
        full_scenes.append(scene if isinstance(scene, str) else scene[0])


df_meta = df_meta[~df_meta['SceneFullName'].isin(scenes_to_drop)]
df_variables = df_variables[~df_variables['SceneFullName'].isin(scenes_to_drop)]

## create the final dataframe ##
cols = ['subject', 'learning_phase', 'scene_full_name', 'count', 'cleared', 'speed',  'MAD_mean', 'delta_clr_tot', 'delta_spd_tot', 'delta_MAD_tot']
df_metrics = pd.DataFrame(columns=cols)
df_metrics['subject'] = df_meta['Subject'].unique()
df_metrics['scene_full_name'] = [df_meta['SceneFullName'].unique().tolist()] * df_metrics['subject'].nunique()

for sub, df_sub in df_metrics.groupby('subject'):
    df_metrics.loc[df_metrics['subject'] == sub, 'learning_phase'] = pd.Series([tuple(df_meta[df_meta['Subject']==sub]['Learning_Phase'].unique())],
                                                                               index=df_metrics.loc[df_metrics['subject'] == sub].index)
    
df_metrics = df_metrics.explode('learning_phase').explode('scene_full_name').reset_index(drop=True)

for idx, row in df_metrics.iterrows():

    sub = row['subject']
    phase = row['learning_phase']
    scene = row['scene_full_name']

    print(f"Calculating metrics for subject {sub}, phase {phase}, scene {scene}...")
    
    mask_meta = (df_meta['Subject'] == sub) & (df_meta['Learning_Phase'] == phase) & (df_meta['SceneFullName'] == scene)
    mask_vars = (df_variables['Subject'] == sub) & (df_variables['Learning_Phase'] == phase) & (df_variables['SceneFullName'] == scene)
    df_tmp_meta = df_meta[mask_meta]
    df_tmp_var = df_variables[mask_vars]
    
    df_metrics.at[idx, 'count'] = len(df_tmp_meta)
    df_metrics.at[idx, 'cleared'] = df_tmp_meta['Cleared'].mean()
    df_metrics.at[idx, 'speed'] = df_tmp_meta['Average_speed'].mean()
    df_metrics.at[idx, 'MAD_mean'] = utils.get_mads(df_tmp_var)['MAD_mean'].mean()

for sub_scn, df_sub_scn in df_metrics.groupby(['subject', 'scene_full_name']):
    sub = sub_scn[0]
    scn = sub_scn[1]

    print(f"Processing subject {sub} and scene {scn}...")
    mask = (df_metrics['subject'] == sub) & (df_metrics['scene_full_name'] == scn)
    df_tmp = df_metrics[mask]
    
    phase_to_ckpt = {phase: idx for idx, phase in enumerate(sorted(df_tmp['learning_phase'].unique()))}

    df_delta_clr = utils.compute_delta_tot(df_tmp, phase_to_ckpt, 'cleared')
    df_delta_spd = utils.compute_delta_tot(df_tmp, phase_to_ckpt, 'speed')
    df_delta_mad = utils.compute_delta_tot(df_tmp, phase_to_ckpt, 'MAD_mean')
    
    df_metrics.loc[mask, 'delta_clr_tot'] = df_delta_clr['delta_tot'].values[0]
    df_metrics.loc[mask, 'delta_spd_tot'] = df_delta_spd['delta_tot'].values[0]
    df_metrics.loc[mask, 'delta_MAD_tot'] = df_delta_mad['delta_tot'].values[0]

df_metrics['level_full_name'] = 'w' + df_metrics['scene_full_name'].str[0] + 'l' + df_metrics['scene_full_name'].str[2]
df_metrics['scene'] = df_metrics['scene_full_name'].str[4:].astype(int)

# save dataframe ##
df_metrics.to_parquet(os.path.join('sourcedata', 'df_metrics.parquet'), index=False)