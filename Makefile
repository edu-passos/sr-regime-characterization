PYTHON ?= python3

.PHONY: install test synthetic-run sleep-run baselines inference tables figures paper-artifacts all

install:
	$(PYTHON) -m pip install -e .[dev,data,baselines]

test:
	$(PYTHON) -m pytest -q

# Full symbolic-regression runs. Slow (many hours) and need a Julia runtime for PySR.
synthetic-run:
	bash scripts/run_synthetic.sh

sleep-run:
	bash scripts/run_sleep_edf.sh

# Recompute the baselines, inference, tables and figures from the run outputs.
baselines:
	$(PYTHON) scripts/synthetic_baselines.py
	$(PYTHON) scripts/sleep_edf_baselines.py
	$(PYTHON) scripts/bilinear_baseline_interaction.py

inference:
	$(PYTHON) scripts/across_seed_inference.py
	$(PYTHON) scripts/across_subject_inference_edf.py
	$(PYTHON) scripts/sleep_edf_loso_auc.py
	$(PYTHON) scripts/edf_memory_validation.py

tables:
	$(PYTHON) scripts/build_tables.py all

figures:
	$(PYTHON) scripts/render_fig_pipeline.py
	$(PYTHON) scripts/fig_concept_radar.py
	$(PYTHON) scripts/fig_concept_radar_edf.py

# Everything downstream of the symbolic-regression runs.
paper-artifacts: baselines inference tables figures

# Full pipeline from scratch (the two runs, then all artifacts).
all: synthetic-run sleep-run paper-artifacts
