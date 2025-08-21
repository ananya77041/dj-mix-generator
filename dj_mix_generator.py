#!/usr/bin/env python3
"""
DJ Mix Generator - Legacy Entry Point
Redirects to the new CLI implementation for backward compatibility
"""

import sys
from pathlib import Path

# Add src directory to Python path for new structure
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Import and run the new CLI
from cli.main import main

if __name__ == "__main__":
    sys.exit(main())