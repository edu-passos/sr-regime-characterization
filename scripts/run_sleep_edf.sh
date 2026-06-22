#!/bin/bash
# Sleep-EDF run (consistent method with run_synthetic.sh):
#   - PySR LIBRARY DEFAULTS via the mvp preset (no per-dataset SR tuning).
#   - Ensemble-mean Hessian + held-out selection / decoupled gate (pipeline.py).
#   - Non-overlapping EEG windows W = step = 160 (the established Sleep-EDF window),
#     overriding mvp's synthetic-specific 120/60.
#   - S = n_seeds = 10, K = top_k = 3, max_lag 6, parsimony 0.01 -- same as synthetic.
#
# Serial + deterministic PySR, parallelised ACROSS SUBJECTS (one process per record),
# exactly as the synthetic parallelises across seeds. This turns an ~8.5-day single
# multithreaded run into ~1.4 days at 8 concurrent: each subject ~510 windows x 10
# seeds, 20 near-equal subjects, ceil(20/JOBS) waves.
#
#   JOBS - concurrent subjects (default = physical cores, capped by RAM).

set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHON_JULIACALL_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sleep_edf}"
SUBJECTS="$OUTPUT_ROOT/subjects"
mkdir -p "$OUTPUT_ROOT"

if ! ls "$SUBJECTS"/subject_*.csv >/dev/null 2>&1; then
    echo "No per-subject CSVs in $SUBJECTS -- run:"
    echo "  .venv/bin/python scripts/prepare_sleep_edf_small.py --label-mapping wake_sleep --split-dir $SUBJECTS --output data/external/sleep_edf_ws_20subjects.csv"
    exit 1
fi

# Concurrency: each serial PySR ~1 core, ~1.1 GB RSS. Default all physical cores; cap by RAM.
PHYS=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2}/^Socket\(s\)/{s=$2}END{print (c*s)+0}')
{ [ -z "$PHYS" ] || [ "$PHYS" -lt 1 ]; } 2>/dev/null && PHYS=$(nproc)
FREEM=$(free -m | awk '/Mem/{print $7}')
RAM_CAP=$(( (FREEM - 2000) / 1100 ))
AUTO=$PHYS
[ "$RAM_CAP" -lt "$AUTO" ] && AUTO=$RAM_CAP
[ "$AUTO" -lt 1 ] && AUTO=1
MAX_JOBS="${JOBS:-$AUTO}"

SUBJ_FILES=("$SUBJECTS"/subject_*.csv)
echo "Sleep-EDF (mvp/PySR-defaults, ensemble-mean Hessian, W=step=160, serial+deterministic): ${#SUBJ_FILES[@]} subjects, up to $MAX_JOBS concurrent, start $(date -Iseconds)"

run_subject() {
    local csv="$1"
    local rid; rid="$(basename "$csv" .csv | sed 's/^subject_//')"
    local out="$OUTPUT_ROOT/subject_$rid"
    echo "  [subject $rid] start $(date -Iseconds)"
    .venv/bin/ssp run-csv \
        --data-csv "$csv" \
        --label-col regime --feature-cols V1 --time-col t --record-col record_id \
        --output-dir "$out" \
        --preset mvp \
        --window-size 160 --step-size 160 \
        --n-seeds 10 --top-k 3 \
        --nesting-level 1 --cost-level 0 \
        --parsimony 0.01 \
        --pysr-parallelism serial \
        --checkpoint-dir "$out/checkpoints" \
        > "$out.log" 2>&1
    echo "  [subject $rid] done  $(date -Iseconds) rc=$?"
}

for csv in "${SUBJ_FILES[@]}"; do
    run_subject "$csv" &
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done
done
wait

echo "ALL ${#SUBJ_FILES[@]} SUBJECTS DONE (Sleep-EDF) at $(date -Iseconds). Outputs in $OUTPUT_ROOT/subject_*/"
