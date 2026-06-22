"""Shared output and figure locations for the experiment scripts.

One place to see where each pipeline stage reads and writes. Imported by the
sibling scripts (run them as `python scripts/<name>.py`, which puts this
directory on the import path). The runner shell scripts take the same locations
as OUTPUT_ROOT defaults.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
FIGURES = REPO / "figures"

# Synthetic suite (run_synthetic.sh -> these).
SYNTHETIC = OUTPUTS / "synthetic"
SYNTHETIC_BASELINES = SYNTHETIC / "baselines"

# Sleep-EDF (run_sleep_edf.sh -> these).
SLEEP_EDF = OUTPUTS / "sleep_edf"

# Bilinear interaction baseline (bilinear_baseline_interaction.py -> here).
BILINEAR = OUTPUTS / "bilinear_baseline"
