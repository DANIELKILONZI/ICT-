"""
Root conftest – ensure the repo root is on sys.path so that
``import python.xxx`` works when running pytest from the project root.
"""
import sys
from pathlib import Path

# Add the repo root (parent of this file) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
