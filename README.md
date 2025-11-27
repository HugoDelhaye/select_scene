# Scene Selection Analysis

Analysis of Super Mario Bros gameplay data to identify scenes with interesting learning patterns across human players, imitation learning models, and PPO agents.

## Quick Start

### Setup

```bash
# Create and activate virtual environment
python -m venv .scene_selection_env
source .scene_selection_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name .scene_selection_env --display-name "Python (scene_selection)"
```

### Run Interactive Dashboard

**Option 1: Web Dashboard** (Recommended - Fast & Modern)
```bash
python run_dashboard.py
# Opens at http://localhost:8050
```

**Option 2: Export to HTML** (For Sharing)
```bash
# Export all scenes
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
python code/make_df_metrics.py
```

## Project Structure

- `code/` - Core processing and visualization modules
- `notebook/` - Jupyter notebooks for analysis
- `sourcedata/` - Data files (parquet format)
- `run_dashboard.py` - Launch web dashboard
- `export_dashboard.py` - Export to HTML files

See `CLAUDE.md` for detailed architecture documentation.