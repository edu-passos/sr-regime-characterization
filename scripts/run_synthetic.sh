#!/bin/bash
# Synthetic run. The method, vs an untuned single-fit baseline:
#
#   1. PySR search hyperparameters left at the LIBRARY DEFAULTS (v1.5.9) -- no
#      per-dataset tuning. (mvp preset now pins populations=31, population_size=27,
#      niterations=100, ncycles=380, maxsize=30, maxdepth=None.)
#   2. Second-order behavioural concepts (Hessian) computed over the full ensemble
#      and aggregated by the MEAN -- same footing as the Jacobian concepts.
#   3. Non-overlapping windows: W = step = 100 (two windows per 200-step regime block).
#
# Serial + deterministic PySR, parallelised across the independent seeds (same as the
# published run): reproducible (same seed => same numbers). Multithreading was tested
# and abandoned -- PySR's per-fit thread speedup on these small windows is only ~1.1x,
# not worth losing determinism for.
#
#   JOBS - concurrent seeds. At PySR defaults each process is ~1.6 GB, so RAM caps
#          concurrency to ~5 on this box.

set -uo pipefail
cd "$(dirname "$0")/.."

# Serial PySR is single-threaded; pin every thread pool to 1 so the N concurrent
# seed processes each stay on ~1 core and don't oversubscribe the 8 physical cores
# (Julia compute, BLAS, and OpenMP all default to "all cores" otherwise).
export PYTHON_JULIACALL_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/synthetic}"
# RESUME (do NOT wipe): the per-window checkpoints from the prior run used the
# identical config, so seeds 0-7's completed scenarios (memory/nonlinearity/
# interaction) are reused via the checkpoint store rather than recomputed.
mkdir -p "$OUTPUT_ROOT"

# 20 seeds (canonical). Override the SET with SYNTH_SEEDS to run a subset, e.g.
#   SYNTH_SEEDS="14 15 16 17 18 19"
# extends an existing 14-seed run with the 6 new seeds only -- seeds 0-13 are left
# untouched (their per-window checkpoints are not re-read because they are not in the set).
SEEDS=(${SYNTH_SEEDS:-$(seq 0 19)})

# Concurrency: the work is CPU-core-bound (each serial PySR ~1 core, ~1.05 GB RSS
# measured). Default to all physical cores; RAM (~1.1 GB/proc) is not the binding
# constraint on this box (14 GB). Capped at physical cores. Override with JOBS=N.
PHYS=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2}/^Socket\(s\)/{s=$2}END{print (c*s)+0}')
{ [ -z "$PHYS" ] || [ "$PHYS" -lt 1 ]; } 2>/dev/null && PHYS=$(nproc)
FREEM=$(free -m | awk '/Mem/{print $7}')
RAM_CAP=$(( (FREEM - 2000) / 1100 ))
AUTO=$PHYS
[ "$RAM_CAP" -lt "$AUTO" ] && AUTO=$RAM_CAP
[ "$AUTO" -lt 1 ] && AUTO=1
MAX_JOBS="${JOBS:-$AUTO}"

echo "Synthetic run (PySR defaults, ensemble-mean Hessian, W=step=100, serial+deterministic): ${#SEEDS[@]} seeds, up to $MAX_JOBS concurrent, start $(date -Iseconds)"

run_seed() {
    local seed="$1"
    echo "  [seed $seed] start $(date -Iseconds)"
    .venv/bin/ssp run-scenario-suite \
        --output-dir "$OUTPUT_ROOT/seed_$seed" \
        --scenarios memory_only,nonlinearity_only,interaction_only,amplitude_only \
        --n-steps 6000 \
        --preset mvp \
        --seed "$seed" \
        --n-seeds 10 \
        --top-k 3 \
        --nesting-level 1 \
        --cost-level 0 \
        --parsimony 0.01 \
        --amplitude-scale-b 2.0 \
        --window-size 100 \
        --step-size 100 \
        --pysr-parallelism serial \
        > "$OUTPUT_ROOT/seed_$seed.log" 2>&1
    echo "  [seed $seed] done  $(date -Iseconds) rc=$?"
}

for seed in "${SEEDS[@]}"; do
    run_seed "$seed" &
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done
done
wait

echo "ALL ${#SEEDS[@]} SEEDS DONE (PySR defaults, ensemble-mean Hessian, serial) at $(date -Iseconds). Outputs in $OUTPUT_ROOT/"
