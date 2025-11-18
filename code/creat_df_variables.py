import glob
import pandas as pd
from pathlib import Path
import os
import json

home = str(Path.home())
path = os.path.join(home,'Téléchargements', 'mario_learning-sourcedata', 'sourcedata', 'ppo_*', 'sub-*', 'ses-*', 'beh', 'variables', '*.json')
file_list_base = glob.glob(path)

columns = ( 'Subject', 'World', 'Level', 'Scene','LevelFullName', 'SceneFullName','ClipCode',
        'Learning_Phase', 'player_x_posHi', 'player_x_posLo', 'player_y_pos','json_path')

# to add 'Phase', 'Phase_stage'

df_variables = pd.DataFrame(columns=columns)

df_variables['json_path'] = file_list_base
df_variables['Subject'] = 'ppo'
df_variables[['World', 'Level', 'Scene', 'ClipCode']] = df_variables['json_path'].str.extract(r'level-w(\d+)l(\d+)_scene-(\d+)_clip-(\d+).json')
df_variables['LevelFullName'] = 'w' + df_variables['World'] + 'l' + df_variables['Level']
df_variables['SceneFullName'] = df_variables['World']+'-'+df_variables['Level']+'-'+df_variables['Scene']
df_variables['Learning_Phase'] = df_variables['json_path'].str.extract(r'ppo_mario_(ep-\w+)')

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

df_variables.to_csv("sourcedata/ppo_clips_variables.csv", index=False)
df_variables.to_parquet("sourcedata/ppo_clips_variables.parquet")