## idée : créer un .parquet avec toute les métadonnées présentes 
## dans les infos/*.json qui viennent de mario.curiosity_scene_agents

# Étape 0 : mettre les .josn a la bonne place ----------- OK
# hum -> téléchargement 
# ppo -> «...»
# imi -> github/mario_curiosrity_scene_agents/outputdata/

# Étape 1: aller chercher tous les fichiers .json dans sourcedata

# Étape 2.1 : extraire les métadonnées des noms de fichiers:
#'subject', 'world', 'level', 'scene', 'clip_code', 'level_full_name', 'scene_full_name'
#éTAPE 2.2 : extraire les métadonnées des fichiers .json:
#'phase', 'learning_phase', 'scene_full_name', 'delta_clr_tot', 'delta_spd_tot', 'delta_MAD_tot', 'count', 'cleared', 'speed',
#'MAD_mean', 'level_full_name', 'scene'
# Étape 2.2 : ouvrir les fichiers et extraire les métadonnées

# Étape 3 : compléter le df avec les métadonnées extraites

# Étape 4 : sauvegarder le df en .parquet

import glob
import json
import pandas as pd
import numpy as np
import utils


def list_json_files(directory):
    """Liste tous les fichiers .json dans le répertoire donné."""
    return glob.glob(f"{directory}/*.json")

def get_imi_subject(df):
    mask = df['subject'].str.match(r"sub-\d{2}_")
    df.loc[mask, 'subject'] = (
        'imi-' +
        df.loc[mask, 'subject'].str.extract(r"(sub-\d{2})")[0]
    )
    return df

def get_hum_phase(df, mask):
    for path in df.loc[mask, 'json_path']:
        with open(path, "r") as f:
            data = json.load(f)
            phase = data["phase"]
            df.loc[df['json_path'] == path, 'phase'] = phase
    return df

def get_hum_learning_phase(df):
    mask = df['learning_phase'].isna() # identify human subjects
    df = get_hum_phase(df, mask)
    
    mask = df["Model"].eq("human")
    med = (
        df.loc[mask]
            .groupby(["Subject", "Phase", "LevelFullName"])["ClipCode"]
            .transform("median")
    )
    df.loc[mask, "Phase_stage"] = np.where(
        df.loc[mask, "ClipCode"] < med,
        "Early",
        "Late"
    )

    return df

def aggregate_metadata(df):
    """Agregates metadata from JSON files into the dataframe."""
    df['clip_code'] = df['json_path'].str.extract(
        r'_clip-(\d+).json'
        )
    df['subject'] = df['json_path'].str.extract(
        r'sourcedata/(ppo+|sub-\d{2}_epoch=\d+-step=\d+|sub-\d+)'
        )
    df = get_imi_subject(df)
    df['learning_phase'] = df['json_path'].str.extract(
        r'_(ep-\d+|epoch=\d+-step=\d+)'
        )
    df = get_hum_learning_phase(df)

    print(df['learning_phase'].unique())

def creat_meta_df():
    # list all json files
    json_files_hum = list_json_files('sourcedata/sub-0?/ses-*/beh/infos')
    json_files_ppo = list_json_files('sourcedata/ppo*/sub-*/ses-*/beh/infos')
    json_files_imi = list_json_files('sourcedata/sub-*_*/sub-*/ses-*/beh/infos')
    json_files = json_files_hum + json_files_ppo + json_files_imi

    print(f"Found {len(json_files)} JSON files.")
    print(json_files[-1])
    df = pd.DataFrame({'json_path': json_files})
    
    
    # extract metadata and aggregate into dataframe
    df = aggregate_metadata(df)

creat_meta_df()
