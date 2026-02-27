#!/bin/bash
# DesignBench copy of run_af3_a800.sh with /root/af_output bind mount fix.
# AF3 resources (sif, model) are from the original path.

# Parse arguments from refold_api.py
# $1: exp_name
# $2: input_json
# $3: output_dir
# $4: gpus (comma-separated, e.g., "0,1,2,3")
# $5: run_data_pipeline (True/False)
# $6: cache_dir
# $7: num_diffusion_samples (optional, default: 1)

exp_name="$1"
input_json="$2"
output_path="$3"
gpus="$4"
run_data_pipeline="$5"
cache_dir="$6"
num_diffusion_samples="${7:-1}"  # Default to 1 if not provided

# Original AF3 installation (sif, model, public_databases)
AF3_BASE="/home/nvme04/qtfeng/design/design/ref/af3_data/af3"
AF3_PUBLIC_DB="/home/nvme04/qtfeng/design/design/ref/af3_data/public_databases"

# Create output directory and log directory (for debugging)
mkdir -p "$output_path"
AF3_LOG_DIR="$(dirname "$output_path")/af3_log"
mkdir -p "$AF3_LOG_DIR"

# Convert comma-separated gpus to space-separated for launch.py
gpus_space_separated=${gpus//,/ }

# Build apptainer bind mounts
# - /root/af_output: some containers use this default output path
# - /app/alphafold/log: capture AF3 logs for debugging (see af3_log/ after run)
BIND_MOUNTS=(
    -B "$AF3_BASE/model:/root/models"
    -B "$AF3_PUBLIC_DB:/root/public_databases"
    -B "$output_path:$output_path"
    -B "$output_path:/root/af_output"
    -B "$AF3_LOG_DIR:/app/alphafold/log"
    -B "$input_json:$input_json"
    -B "/home/nvme04:/home/nvme04"
)

# Add cache_dir bind mount if provided
if [[ -n "$cache_dir" && "$cache_dir" != "None" && "$cache_dir" != "" ]]; then
    mkdir -p "$cache_dir"
    BIND_MOUNTS+=(-B "$cache_dir:$cache_dir")
fi

# Execute AlphaFold3 using apptainer
apptainer exec \
    --nv \
    --writable-tmpfs \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROMPT_COMMAND= \
    "${BIND_MOUNTS[@]}" \
    "$AF3_BASE/af3.sif" \
    python /app/alphafold/launch.py \
    --input_json "$input_json" \
    --output_dir "$output_path" \
    --run_data_pipeline "$run_data_pipeline" \
    --gpus $gpus_space_separated \
    --exp_name "$exp_name" \
    --num_diffusion_samples "$num_diffusion_samples"
