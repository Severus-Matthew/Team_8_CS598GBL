#!/bin/bash
set -o pipefail

CONDA_BASE="/home/ubuntu/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate SE3nv

PYTHON_BIN="$CONDA_PREFIX/bin/python"

echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

INPUT_ROOT="/home/ubuntu/manvi/Team_8_CS598GBL/outputs/custom90_128_motif_scaffolding/low_noise"
OUTPUT_ROOT="/home/ubuntu/manvi/Team_8_CS598GBL/outputs/mpnn_custom90_128_motif_scaffolding/low_noise"

NUM_SEQ=1
TEMP="0.1"

GPUS=(0 1 2 3 4 5 6 7)
TASKS_PER_GPU=3
MAX_JOBS=$(( ${#GPUS[@]} * TASKS_PER_GPU ))

mkdir -p "$OUTPUT_ROOT"

echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MAX_JOBS=$MAX_JOBS"
echo "Checking input PDBs..."
find "$INPUT_ROOT" -mindepth 3 -maxdepth 3 -name "*.pdb" | head

run_one() {
    local pdb_file="$1"
    local gpu_id="$2"

    local len_name
    local motif_name
    local pdb_name
    local out_dir
    local log_file
    local err_file

    len_name=$(basename "$(dirname "$(dirname "$pdb_file")")")
    motif_name=$(basename "$(dirname "$pdb_file")")
    pdb_name=$(basename "$pdb_file" .pdb)

    out_dir="$OUTPUT_ROOT/$len_name/$motif_name/$pdb_name"
    mkdir -p "$out_dir"

    log_file="$out_dir/run.log"
    err_file="$out_dir/run.err"

    if ls "$out_dir"/*.fa >/dev/null 2>&1; then
        echo "[SKIP] Existing fasta found: $out_dir"
        return 0
    fi

    {
        echo "================================================"
        echo "START: $(date)"
        echo "PDB: $pdb_file"
        echo "GPU: $gpu_id"
        echo "OUT: $out_dir"
        echo "Python: $(which python)"
        echo "CUDA_VISIBLE_DEVICES=$gpu_id"
        "$PYTHON_BIN"- <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
        echo "================================================"
        echo "Running ProteinMPNN..."
    } > "$log_file" 2> "$err_file"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" /home/ubuntu/manvi/Team_8_CS598GBL/external/ProteinMPNN/protein_mpnn_run.py \
        --pdb_path "$pdb_file" \
        --out_folder "$out_dir" \
        --num_seq_per_target "$NUM_SEQ" \
        --sampling_temp "$TEMP" \
        >> "$log_file" 2>> "$err_file"

    status=$?

    if [[ $status -eq 0 ]]; then
        echo "DONE: $(date)" >> "$log_file"
        echo "[DONE] $pdb_file"
    else
        echo "FAILED with status $status at $(date)" >> "$err_file"
        echo "[FAILED] $pdb_file | see $err_file"
    fi

    return $status
}

job_count=0
total_found=0

for len_dir in "$INPUT_ROOT"/len_*; do
    [ -d "$len_dir" ] || continue

    for motif_dir in "$len_dir"/*; do
        [ -d "$motif_dir" ] || continue

        for pdb_file in "$motif_dir"/*.pdb; do
            [ -f "$pdb_file" ] || continue

            total_found=$((total_found + 1))
            gpu_id=${GPUS[$(( job_count % ${#GPUS[@]} ))]}

            run_one "$pdb_file" "$gpu_id" &

            job_count=$((job_count + 1))

            if (( job_count % MAX_JOBS == 0 )); then
                wait
            fi
        done
    done
done

wait

echo "Total PDB files found: $total_found"
echo "Total jobs launched: $job_count"
echo "All ProteinMPNN runs completed."