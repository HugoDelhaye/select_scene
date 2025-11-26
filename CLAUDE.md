# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes Super Mario Bros gameplay data from multiple sources (human players, imitation learning models, and PPO agents) to select scenes based on learning metrics. The pipeline processes gameplay clips, extracts variables, computes learning metrics, and identifies scenes showing interesting learning patterns across different subject types.

## Environment Setup

```bash
python -m venv .scene_selection_env
source .scene_selection_env/bin/activate
pip install ipykernel
python -m ipykernel install --user --name .scene_selection_env --display-name "Python (scene_selection)"
pip install -r requirements.txt
```

The virtual environment is located in `env/` (committed to repo).

## Data Pipeline Architecture

The analysis follows a multi-stage pipeline that transforms raw gameplay data into scene selection metrics:

### 1. Data Sources (`sourcedata/`)
- `clips_metadata_from_im.parquet`: Metadata from imitation learning clips
- `clips_metadata_with_patterns.parquet`: Metadata from human/PPO clips
- `df_variables_hum.parquet`, `df_variables_ppo.parquet`: Frame-by-frame player position variables
- `df_metrics.parquet`: Final computed metrics (output of pipeline)

### 2. Core Processing Scripts (`code/`)

**`make_df_metrics.py`** - Main pipeline orchestrator:
- Loads and merges metadata from multiple sources (human, imitation, PPO)
- Combines position variables across all subjects
- Filters scenes to only those completed by all 5 human subjects
- Aggregates per-scene statistics (clear rate, speed, MAD)
- Computes delta metrics showing learning progress using `utils.compute_deltas_for_group()`
- Outputs `sourcedata/df_metrics.parquet`

**`creat_df_variables.py`** - Extracts PPO variables:
- Reads JSON files containing frame-by-frame player position data
- Parses file paths to extract world, level, scene, clip codes
- Extracts `player_x_posHi`, `player_x_posLo`, `player_y_pos` from JSONs
- Outputs `sourcedata/ppo_clips_variables.parquet`

**`utils.py`** - Core metric computation:
- `compute_deltas_for_group()`: Computes learning deltas for a subject-scene group
  - Handles phase ordering differently for imitation models vs humans
  - Special case for `im_sub-02` phase ordering
  - Computes total delta (max - min) across learning phases for cleared rate, speed, and MAD
- `compute_delta_tot()`: Calculates delta between min/max checkpoints for a metric
- `mad()`: Median Absolute Deviation calculation
- `get_mads()`: Aggregates MAD of y-position for each x-position (trajectory variability)

### 3. Subject Types

The codebase distinguishes three subject types:
- **`hum`**: Human players (`sub-01` through `sub-06`)
- **`imi`**: Imitation learning models (`im_sub-01`, etc.) with checkpoint-based phases
- **`ppo`**: PPO reinforcement learning agent with episode-based phases

### 4. Learning Phases

**Human subjects**: `Early discovery`, `Late discovery`, `Early practice`, `Late practice`

**Imitation models**: Checkpoint-based (e.g., `sub-01_epoch=0-step=500`, `sub-01_epoch=0-step=2000`)
- First phase always ends with `=500`
- Special ordering exception for `im_sub-02` (moves `epoch=0-step=10000` to end)

**PPO agent**: Episode-based (e.g., `ep-xxx`)

### 5. Key Metrics

- **`delta_clr_tot`**: Total change in clear rate (success) from min to max checkpoint
- **`delta_spd_tot`**: Total change in average speed across learning
- **`delta_MAD_tot`**: Total change in trajectory variability (MAD of y-position per x-position)

## Analysis Workflow

Primary analysis is in `notebook/select_scene.ipynb`:
1. Load `df_metrics.parquet`
2. Filter out scenes with <10 clips per phase for human subjects
3. Group by scene and subject type, compute mean deltas
4. Select top/bottom N scenes for each metric and subject type
5. Identify scenes appearing in multiple subject types (intersection analysis)

## Data Format Conversions

- `convert_csv2parquet.py`: CSV → Parquet (forces string dtype to avoid type guessing issues)
- `parquet_2_csv.py`: Parquet → CSV using pyarrow for proper type handling

## Important Notes

- Position data uses two bytes: `player_x_pos = player_x_posHi * 255 + player_x_posLo`
- Scene filtering ensures all 5 human subjects completed the scene
- The `df_metrics` structure has one row per (subject, learning_phase, scene_full_name) combination
- MAD metrics are only available for scenes with position variable data (not all scenes)
