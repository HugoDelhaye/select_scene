"""
Invoke tasks for scene selection pipeline.

This file defines automated tasks for the Super Mario Bros scene selection
analysis pipeline using the airoh framework.

Usage:
    invoke --list                # Show all available tasks
    invoke setup                 # Install Python dependencies
    invoke process-data          # Generate metrics from raw data
    invoke dashboard             # Launch interactive web dashboard
    invoke export-html           # Export all scenes to HTML
    invoke clean                 # Remove generated outputs
"""

from invoke import task
from airoh.utils import setup_env_python, ensure_dir_exist, clean_folder


@task
def setup(c):
    """
    Install Python dependencies from requirements.txt.

    This creates/activates the virtual environment and installs all required packages.
    """
    setup_env_python(c)


@task
def ensure_dirs(c):
    """
    Ensure all required directories exist.
    """
    ensure_dir_exist(c, "sourcedata")
    ensure_dir_exist(c, "output")


@task(pre=[ensure_dirs])
def process_data(c):
    """
    Generate metrics dataframe from raw gameplay data.

    This runs the main data processing pipeline that:
    - Loads metadata and variable data for all subjects
    - Filters scenes by completion criteria
    - Computes learning metrics (clear rate, speed, MAD)
    - Outputs sourcedata/df_metrics.parquet
    """
    code_dir = c.config.get("directories", {}).get("code", "code")
    c.run(f"python {code_dir}/make_df_metrics.py")


@task
def process_ppo_variables(c):
    """
    Extract PPO variables from JSON files.

    This parses PPO agent gameplay data and creates the PPO variables parquet file.
    """
    code_dir = c.config.get("directories", {}).get("code", "code")
    c.run(f"python {code_dir}/create_df_variables.py")


@task
def dashboard(c):
    """
    Launch the interactive web dashboard.

    Opens the Dash web application at http://localhost:8050 for exploring
    learning metrics across all scenes and subjects.
    """
    host = c.config.get("dashboard", {}).get("host", "localhost")
    port = c.config.get("dashboard", {}).get("port", 8050)
    print(f"Starting dashboard at http://{host}:{port}")
    c.run("python run_dashboard.py")


@task
def export_html(c, level=None, scene=None):
    """
    Export scenes to standalone HTML files.

    Args:
        level: (Optional) Specific level to export (e.g., w1l1)
        scene: (Optional) Specific scene number to export (requires level)

    Examples:
        invoke export-html                    # Export all scenes
        invoke export-html --level=w1l1       # Export all scenes from level w1l1
        invoke export-html --level=w1l1 --scene=0  # Export single scene
    """
    output_dir = c.config.get("export", {}).get("output_dir", "html_export")

    cmd = f"python export_dashboard.py --output-dir {output_dir}"
    if level:
        cmd += f" --level {level}"
    if scene is not None:
        cmd += f" --scene {scene}"

    c.run(cmd)


@task
def convert_csv_to_parquet(c):
    """
    Convert CSV files to Parquet format.

    This is useful for converting legacy CSV data files.
    """
    code_dir = c.config.get("directories", {}).get("code", "code")
    c.run(f"python {code_dir}/convert_csv2parquet.py")


@task
def convert_parquet_to_csv(c):
    """
    Convert Parquet files to CSV format.

    This is useful for sharing data with tools that don't support Parquet.
    """
    code_dir = c.config.get("directories", {}).get("code", "code")
    c.run(f"python {code_dir}/parquet_2_csv.py")


@task
def clean(c):
    """
    Remove generated output files.

    This cleans up HTML exports and other generated artifacts.
    """
    clean_folder(c, "output", pattern="*.html")
    print("Cleaned generated outputs")


@task(pre=[process_data])
def run(c):
    """
    Run the complete pipeline: process data and launch dashboard.

    This is the main task that processes all data and opens the dashboard.
    """
    dashboard(c)


@task
def list_scenes(c):
    """
    List all available scenes in the dataset.

    Reads df_metrics.parquet and displays unique scenes.
    """
    c.run("""python -c "
import pandas as pd
df = pd.read_parquet('sourcedata/df_metrics.parquet')
scenes = df['scene_full_name'].unique()
print('Available scenes:')
for scene in sorted(scenes):
    print(f'  - {scene}')
print(f'\\nTotal: {len(scenes)} scenes')
"
""")


@task
def stats(c):
    """
    Display dataset statistics.

    Shows summary statistics about subjects, scenes, and learning phases.
    """
    c.run("""python -c "
import pandas as pd
df = pd.read_parquet('sourcedata/df_metrics.parquet')

print('Dataset Statistics')
print('=' * 50)
print(f'Subjects: {df[\"subject\"].nunique()}')
print(f'Scenes: {df[\"scene_full_name\"].nunique()}')
print(f'Levels: {df[\"level_full_name\"].nunique()}')
print(f'Total rows: {len(df)}')
print()
print('Subjects by type:')
for subject_type in ['hum', 'imi', 'ppo']:
    count = df[df[\"subject\"].str.startswith(subject_type)][\"subject\"].nunique()
    if count > 0:
        print(f'  {subject_type}: {count}')
"
""")
