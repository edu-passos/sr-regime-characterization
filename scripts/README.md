# Experiment scripts

These reproduce the paper's results from the symbolic-regression runs. The `src/`
package is the method; the scripts here drive the experiments around it. Output
locations are defined once in `_paths.py`. `make paper-artifacts` runs the whole
downstream chain; the stages below show what each script does and its order.

Run scripts from the repository root, e.g. `python scripts/across_seed_inference.py`.

## Pipeline order

### 1. Data prep
- `prepare_sleep_edf_small.py` - download Sleep-EDF via MNE and write a labelled CSV; `--split-dir` also writes one CSV per subject (what `run_sleep_edf.sh` reads).

### 2. Symbolic-regression runs (slow; need a Julia runtime for PySR)
- `run_synthetic.sh` - the 20-seed synthetic suite, writing `outputs/synthetic/`.
- `run_sleep_edf.sh` - the per-subject Sleep-EDF runs, writing `outputs/sleep_edf/`.

### 3. Classical baselines
- `synthetic_baselines.py` - DFA, Lyapunov, entropies, AR(2), skewness, kurtosis on the synthetic windows.
- `sleep_edf_baselines.py` - skewness / kurtosis on the Sleep-EDF windows.
- `bilinear_baseline_interaction.py` - bilinear-AR oracle-vs-blind baseline on the interaction regime.

### 4. Inference
- `across_seed_inference.py` - across-seed test over the synthetic suite; writes `figures/results_master.csv`.
- `across_subject_inference_edf.py` - across-subject Wake-vs-Sleep test.
- `sleep_edf_loso_auc.py` - leave-one-subject-out AUC by feature set.
- `edf_memory_validation.py` - model-free lag-order validation (AR order, spectral slope).

### 5. Tables (LaTeX for the paper)
- `build_tables.py [specificity|sleep|all]` - the synthetic specificity table (`tab:specificity`) and the Sleep-EDF within-subject table (`tab:sleep-edf-baselines`).

### 6. Figures
- `render_fig_pipeline.py` - the pipeline schematic (Fig. 1).
- `fig_concept_radar.py` - synthetic concept radar.
- `fig_concept_radar_edf.py` - Sleep-EDF concept radar.

## Helpers
- `_paths.py` - shared output and figure locations (imported by the scripts above).
