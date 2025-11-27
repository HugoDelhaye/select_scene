# Airoh Integration Summary

This document summarizes the integration of [airoh](https://github.com/airoh-pipeline/airoh) into the scene selection pipeline for standardized, reproducible workflows.

## What is Airoh?

Airoh is a lightweight Python task library built on the `invoke` framework, designed for reproducible research workflows. It provides pre-built, modular task definitions that streamline workflow automation while maintaining clean project structure.

## Changes Made

### 1. New Configuration Files

#### `invoke.yaml`
Central configuration file defining:
- Directory structure (`code/`, `notebook/`, `sourcedata/`, `html_export/`)
- Data file paths
- Python environment settings
- Dashboard and export configurations

#### `tasks.py`
Task definitions for common operations:
- `invoke setup` - Install Python dependencies
- `invoke process-data` - Generate metrics from raw data
- `invoke process-ppo-variables` - Extract PPO variables
- `invoke dashboard` - Launch interactive web dashboard
- `invoke export-html` - Export scenes to HTML
- `invoke stats` - Display dataset statistics
- `invoke list-scenes` - List all available scenes
- `invoke clean` - Remove generated outputs
- `invoke convert-csv-to-parquet` - CSV to Parquet conversion
- `invoke convert-parquet-to-csv` - Parquet to CSV conversion

### 2. Updated Dependencies

Added to `requirements.txt`:
```
airoh>=0.1.4
invoke>=2.0
```

### 3. Updated Documentation

Both `README.md` and `CLAUDE.md` now include:
- Airoh installation instructions
- Task usage examples
- References to invoke commands alongside direct Python script execution

## Getting Started

### Installation

```bash
# Install all dependencies including airoh
pip install -r requirements.txt
```

### View Available Tasks

```bash
# List all available invoke tasks
invoke --list
```

### Quick Start Examples

```bash
# Setup environment
invoke setup

# Process data
invoke process-data

# Launch dashboard
invoke dashboard

# Export all scenes to HTML
invoke export-html

# Export specific level
invoke export-html --level=w1l1

# Display statistics
invoke stats
```

## Benefits of Airoh Integration

1. **Standardized Workflow**: Common tasks are now documented and executable via simple commands
2. **Reproducibility**: Configuration-driven approach ensures consistent execution
3. **Discoverability**: `invoke --list` shows all available operations
4. **Flexibility**: Tasks can still be run directly via Python scripts
5. **Documentation**: Task descriptions explain what each operation does
6. **YODA Compliance**: Follows YODA (Your Open Data Architecture) principles for research projects

## Directory Structure (YODA-inspired)

```
select_scene/
├── code/                      # Analysis and processing code
│   ├── make_df_metrics.py    # Main pipeline
│   ├── create_df_variables.py # PPO variable extraction
│   ├── utils.py              # Utility functions
│   ├── dashboard.py          # Dashboard functions
│   └── ...
├── notebook/                  # Jupyter notebooks
├── sourcedata/               # Raw and processed data
├── html_export/              # Generated HTML outputs
├── invoke.yaml               # Airoh configuration
├── tasks.py                  # Task definitions
├── requirements.txt          # Python dependencies
├── README.md                 # User documentation
└── CLAUDE.md                 # Developer documentation
```

## Backward Compatibility

All existing Python scripts continue to work as before. The airoh tasks provide a convenient wrapper but don't replace direct script execution:

```bash
# Both approaches work:
invoke process-data              # Using airoh task
python code/make_df_metrics.py   # Direct execution
```

## Next Steps

To fully activate the airoh environment:

1. Install dependencies: `pip install -r requirements.txt`
2. Try listing tasks: `invoke --list`
3. Run your first task: `invoke stats`

## Resources

- [Airoh GitHub Repository](https://github.com/airoh-pipeline/airoh)
- [Airoh Documentation](https://airoh-pipeline.github.io/airoh/)
- [Airoh Template Repository](https://github.com/airoh-pipeline/airoh-template)
- [Invoke Framework](https://www.pyinvoke.org/)
