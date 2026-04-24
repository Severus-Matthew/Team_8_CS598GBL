#!/bin/bash
set -eo pipefail
CONDA_BASE=$(conda info --base)
export CONDA_BASE
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate SE3nv
PYTHON_BIN="$CONDA_PREFIX/bin/python"
export PYTHON_BIN
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print(sys.executable); import torch; print(torch.__version__)"

# Run from RFdiffusion repo root
# Example:
# cd /home/manvi/Team_8_CS598GBL/RFdiffusion
# bash run_motif_batch_8gpu.sh

RF_DIR="/lambda/nfs/manvi/Team_8_CS598GBL/RFdiffusion"
INPUT_ROOT="/lambda/nfs/manvi/Team_8_CS598GBL/Assignment/Generate_designs/generated_designs"
OUT_ROOT="/lambda/nfs/manvi/Team_8_CS598GBL/outputs/custom90_128_motif_scaffolding"

NUM_DESIGNS="${NUM_DESIGNS:-1}"

GPUS=4
TASKS_PER_GPU=2
TOTAL_WORKERS=$((GPUS * TASKS_PER_GPU))

METHODS=("default" "fast_T20" "low_noise")
CONFIGS=("mid" "short" "nterm" "cterm" "balanced")

mkdir -p "$OUT_ROOT"
cd "$RF_DIR"

make_contig() {
    local L=$1
    local cfg=$2

    # Motif ranges are derived from protein length L.
    # All residue indices are 1-indexed on chain A.

    local mid_start mid_end short_start short_end
    local n_end c_start bal_start bal_end
    local left_min left_max right_min right_max

    # Scaffold ranges scale mildly with length.
    left_min=$(( L / 5 ))
    left_max=$(( L / 3 ))
    right_min=$(( L / 5 ))
    right_max=$(( L / 3 ))

    # Avoid tiny values
    if (( left_min < 10 )); then left_min=10; fi
    if (( right_min < 10 )); then right_min=10; fi
    if (( left_max < left_min + 5 )); then left_max=$((left_min + 5)); fi
    if (( right_max < right_min + 5 )); then right_max=$((right_min + 5)); fi

    # mid motif: central ~30% of protein
    mid_start=$(( L * 35 / 100 ))
    mid_end=$(( L * 65 / 100 ))

    # short motif: central ~12% of protein, min 10 residues
    short_start=$(( L * 44 / 100 ))
    short_end=$(( L * 56 / 100 ))
    if (( short_end - short_start + 1 < 10 )); then
        short_start=$(( L / 2 - 5 ))
        short_end=$(( L / 2 + 4 ))
    fi

    # nterm motif: first ~22%
    n_end=$(( L * 22 / 100 ))
    if (( n_end < 15 )); then n_end=15; fi

    # cterm motif: last ~22%
    c_start=$(( L - n_end + 1 ))

    # balanced motif: slightly N-centered internal motif, ~25%
    bal_start=$(( L * 22 / 100 ))
    bal_end=$(( L * 45 / 100 ))

    # Safety bounds
    if (( mid_start < 1 )); then mid_start=1; fi
    if (( short_start < 1 )); then short_start=1; fi
    if (( bal_start < 1 )); then bal_start=1; fi
    if (( mid_end > L )); then mid_end=$L; fi
    if (( short_end > L )); then short_end=$L; fi
    if (( bal_end > L )); then bal_end=$L; fi

    case "$cfg" in
        mid)
            echo "[${left_min}-${left_max}/A${mid_start}-${mid_end}/${right_min}-${right_max}]"
            ;;
        short)
            echo "[${left_min}-${left_max}/A${short_start}-${short_end}/${right_min}-${right_max}]"
            ;;
        nterm)
            echo "[A1-${n_end}/${left_min}-${left_max}]"
            ;;
        cterm)
            echo "[${left_min}-${left_max}/A${c_start}-${L}]"
            ;;
        balanced)
            echo "[${left_min}-${left_max}/A${bal_start}-${bal_end}/${right_min}-${right_max}]"
            ;;
        *)
            echo "Unknown config: $cfg" >&2
            exit 1
            ;;
    esac
}

run_one() {
    local gpu=$1
    local method=$2
    local len_str=$3
    local cfg=$4
    local pdb=$5

    local L=$((10#$len_str))
    local contig
    contig=$(make_contig "$L" "$cfg")

    local out_dir="${OUT_ROOT}/${method}/len_${len_str}/${cfg}"
    local out_prefix="${out_dir}/scaffold"

    mkdir -p "$out_dir"
    if ls "${out_prefix}"*.pdb 1> /dev/null 2>&1; then
        echo "[$(date)] SKIPPING (exists) | ${method} len_${len_str} ${cfg}"
        return
    fi

    echo "[$(date)] GPU ${gpu} | ${method} len_${len_str} ${cfg}"
    echo "PDB: ${pdb}"
    echo "CONTIG: ${contig}"
    echo "OUT: ${out_prefix}"

    CUDA_VISIBLE_DEVICES="$gpu" bash -c "
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate SE3nv
    $PYTHON_BIN scripts/run_inference.py \
        inference.input_pdb='$pdb' \
        inference.output_prefix='$out_prefix' \
        inference.num_designs='$NUM_DESIGNS' \
        'contigmap.contigs=${contig}'
    " > "${out_dir}/run.log" 2> "${out_dir}/run.err"

    echo "[$(date)] DONE GPU ${gpu} | ${method} len_${len_str} ${cfg}"
}

export -f make_contig
export -f run_one
export RF_DIR INPUT_ROOT OUT_ROOT NUM_DESIGNS

JOBS_FILE="${OUT_ROOT}/jobs.tsv"
rm -f "$JOBS_FILE"

for method in "${METHODS[@]}"; do
    for len in $(seq -w 90 128); do
        # convert 90 -> 090
        len_str=$(printf "%03d" "$((10#$len))")
        pdb="${INPUT_ROOT}/${method}/len_${len_str}/design_0.pdb"

        if [[ ! -f "$pdb" ]]; then
            echo "Skipping missing: $pdb"
            continue
        fi

        for cfg in "${CONFIGS[@]}"; do
            echo -e "${method}\t${len_str}\t${cfg}\t${pdb}" >> "$JOBS_FILE"
        done
    done
done

echo "Total jobs:"
wc -l "$JOBS_FILE"

echo "Launching with ${GPUS} GPUs and ${TASKS_PER_GPU} tasks/GPU = ${TOTAL_WORKERS} workers"

worker() {
    local worker_id=$1
    local gpu=$((worker_id % GPUS))

    while true; do
        local line
        line=$(flock -x 200 bash -c '
            if read -r line < "$1"; then
                sed -i "1d" "$1"
                echo "$line"
            fi
        ' bash "$JOBS_FILE" 200>"${JOBS_FILE}.lock")

        if [[ -z "${line:-}" ]]; then
            break
        fi

        IFS=$'\t' read -r method len_str cfg pdb <<< "$line"
        run_one "$gpu" "$method" "$len_str" "$cfg" "$pdb"
    done
}

for wid in $(seq 0 $((TOTAL_WORKERS - 1))); do
    worker "$wid" &
done

wait

echo "All motif scaffolding jobs completed."