# External Datasets

This folder contains compact labeled univariate time-series CSVs for `run-csv`.
The CSVs are built locally and are not version controlled or redistributed; the
Sleep-EDF source data keeps its original PhysioNet license.

## Sleep-EDF Expanded (primary external dataset)

- Source: PhysioNet Sleep-EDF Expanded (downloaded via `mne`)
- Suggested univariate channel: `EEG Fpz-Cz`
- Suggested binary mappings:
  - `wake_sleep`: `W` vs `{R,1,2,3,4}`
  - `rem_nrem`: `R` vs `{1,2,3,4}`
- Output example: `sleep_edf_wake_sleep_small.csv`

Install extra dependency:

```bash
.venv/bin/pip install -e .[data]
```

Build dataset:

```bash
.venv/bin/python scripts/prepare_sleep_edf_small.py \
  --label-mapping wake_sleep \
  --subjects 0 \
  --recordings 1 \
  --output data/external/sleep_edf_wake_sleep_small.csv
```

Run precheck gate directly while preparing:

```bash
.venv/bin/python scripts/prepare_sleep_edf_small.py \
  --label-mapping wake_sleep \
  --window-size 120 \
  --step-size 60 \
  --min-label-windows 20 \
  --fail-if-below-threshold
```
