#!/usr/bin/env bash
set -euo pipefail

# Activate env before running this script:
# conda activate SE3nv

SCRIPT=/home/manvi/Team_8_CS598GBL/RFdiffusion/scripts/run_inference.py
OUTROOT=/home/manvi/Team_8_CS598GBL/Assignment/Generate_designs
NUM_DESIGNS=10

# User asked for 90-120 aa. If you want the full assignment challenge, change END_LEN to 128.
START_LEN=90
END_LEN=128

run_config () {
    local config_name="$1"
    shift
    local extra_args=("$@")

    for L in $(seq ${START_LEN} ${END_LEN}); do
        len_tag=$(printf "%03d" "${L}")
        out_prefix="${OUTROOT}/${config_name}/len_${len_tag}/design"

        mkdir -p "$(dirname "${out_prefix}")"

        echo "=================================================="
        echo "Running config=${config_name}, length=${L}"
        echo "Output prefix: ${out_prefix}"
        echo "=================================================="

        ${SCRIPT} \
          "contigmap.contigs=[${L}-${L}]" \
          inference.output_prefix="${out_prefix}" \
          inference.num_designs=${NUM_DESIGNS} \
          "${extra_args[@]}"
    done
}

# 1) Default
run_config "default"

# 2) Low-noise
run_config "low_noise" \
  denoiser.noise_scale_ca=0.5 \
  denoiser.noise_scale_frame=0.5

# 3) Fast
run_config "fast_T20" \
  diffuser.T=20

# 4) Very low-noise
run_config "very_low_noise" \
  denoiser.noise_scale_ca=0.0 \
  denoiser.noise_scale_frame=0.0

# # 5) Medium-fast
# run_config "medium_fast_T35" \
#   diffuser.T=35

# 6) Early-stop
run_config "early_stop_f10" \
  inference.final_step=10
