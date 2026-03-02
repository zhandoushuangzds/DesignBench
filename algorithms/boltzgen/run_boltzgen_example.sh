#!/bin/bash
# Example script to run BoltzGen with timing records
# Usage: bash run_boltzgen_example.sh

DESIGNBENCH_ROOT="/home/qiantai/zihan/designbench"
OUTPUT_DIR="/path/to/boltzgen_output"
BOLTZGEN_ENV="/DATA/disk0/qtfeng/miniforge3/envs/boltzgen"
GPU=0
NUM_DESIGNS=100

# Run for both antibody and nanobody
python ${DESIGNBENCH_ROOT}/algorithms/boltzgen/run_boltzgen_with_timing.py \
  --designbench_root ${DESIGNBENCH_ROOT} \
  --output_dir ${OUTPUT_DIR} \
  --boltzgen_env ${BOLTZGEN_ENV} \
  --task both \
  --num_designs ${NUM_DESIGNS} \
  --gpu ${GPU}

# Or run separately:
# Antibody only:
# python ${DESIGNBENCH_ROOT}/algorithms/boltzgen/run_boltzgen_with_timing.py \
#   --designbench_root ${DESIGNBENCH_ROOT} \
#   --output_dir ${OUTPUT_DIR}/antibody_only \
#   --boltzgen_env ${BOLTZGEN_ENV} \
#   --task antibody \
#   --num_designs ${NUM_DESIGNS} \
#   --gpu ${GPU}

# Nanobody only:
# python ${DESIGNBENCH_ROOT}/algorithms/boltzgen/run_boltzgen_with_timing.py \
#   --designbench_root ${DESIGNBENCH_ROOT} \
#   --output_dir ${OUTPUT_DIR}/nanobody_only \
#   --boltzgen_env ${BOLTZGEN_ENV} \
#   --task nanobody \
#   --num_designs ${NUM_DESIGNS} \
#   --gpu ${GPU}
