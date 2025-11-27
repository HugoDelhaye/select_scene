#!/usr/bin/env python
"""
Launcher script for the Scene Exploration Dashboard.

This script runs the Dash web server for interactive exploration.
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from dash_app import run_server

if __name__ == '__main__':
    run_server()
