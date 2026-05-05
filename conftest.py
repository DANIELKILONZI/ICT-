"""
Root conftest – ensure the repo root is on sys.path so that
``import python.xxx`` works when running pytest from the project root.

Also provides shared fixtures used across multiple test modules.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add the repo root (parent of this file) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def sample_csv_dir(tmp_path_factory) -> Path:
    """
    Generate EURUSD D1 / H1 / M5 synthetic CSV files once per test session.

    Uses a fixed seed so results are deterministic.  Shared by
    test_backtest_setup4.py and test_backtest_all_setups.py.
    """
    from scripts.generate_sample_data import generate  # noqa: PLC0415

    csv_dir = tmp_path_factory.mktemp("sample_csv")
    rng = np.random.default_rng(seed=42)
    for tf in ("D1", "H1", "M5"):
        generate("EURUSD", tf, csv_dir, rng, force=True)
    return csv_dir
