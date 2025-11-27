# Scene Selection Analysis

Analysis of Super Mario Bros gameplay data to identify scenes with interesting learning patterns across human players, imitation learning models, and PPO agents.

## Quick Start

### Setup

This project uses [airoh](https://github.com/airoh-pipeline/airoh) for task automation and reproducible workflows.

```bash
# Install dependencies (includes airoh and invoke)
pip install -r requirements.txt

# Or use airoh task for setup
invoke setup

# (Optional) Install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name .scene_selection_env --display-name "Python (scene_selection)"
```

To see all available tasks:
```bash
invoke --list
```

See [.invoke-cheatsheet.md](.invoke-cheatsheet.md) for a quick reference of all tasks.

### Run Interactive Dashboard

**Option 1: Web Dashboard** (Recommended - Fast & Modern)
```bash
# Using airoh task
invoke dashboard

# Or directly
python run_dashboard.py
# Opens at http://localhost:8050
```

**Option 2: Export to HTML** (For Sharing)
```bash
# Export all scenes using airoh task
invoke export-html

# Export specific level or scene
invoke export-html --level=w1l1
invoke export-html --level=w1l1 --scene=0

# Or use the Python script directly
python export_dashboard.py --output-dir html_export

# Then open html_export/index.html in your browser
# Share the entire html_export/ folder with collaborators
```

**Option 3: Jupyter Notebook** (Legacy)
```bash
jupyter notebook notebook/figure_exploration.ipynb
```

## Data Pipeline

Generate metrics from raw data:
```bash
# Using airoh task (recommended)
invoke process-data

# Or directly
python code/make_df_metrics.py
```

### Available Tasks

See all available airoh tasks:
```bash
invoke --list
```

Key tasks:
- `invoke setup` - Install Python dependencies
- `invoke process-data` - Generate metrics from raw data
- `invoke dashboard` - Launch interactive web dashboard
- `invoke export-html` - Export scenes to standalone HTML
- `invoke stats` - Display dataset statistics
- `invoke list-scenes` - List all available scenes
- `invoke clean` - Remove generated outputs

## Project Structure

- `code/` - Core processing and visualization modules
- `notebook/` - Jupyter notebooks for analysis
- `sourcedata/` - Data files (parquet format)
- `run_dashboard.py` - Launch web dashboard
- `export_dashboard.py` - Export to HTML files

See `CLAUDE.md` for detailed architecture documentation.