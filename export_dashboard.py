#!/usr/bin/env python
"""
Export the dashboard as a standalone HTML file.

This script generates a single HTML file containing all scenes, images,
and interactive controls. No server needed - just open in a browser.
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from export_dashboard import main

if __name__ == '__main__':
    main()
