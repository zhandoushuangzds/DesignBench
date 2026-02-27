#!/usr/bin/env bash
# Single-target MotifBench test: 01_1LDB, first 5 samples only.
# Usage (from benchcore repo root, with conda/env activated):
#   bash scripts/run_motif_scaffolding_single_target_test.sh
#
# Or with Hydra overrides:
#   python scripts/run_motif_scaffolding_pipeline.py \
#     design_dir=/path/to/Motif_Benchmark/20260206_RFD3_ouputs/rfd3_outputs/01_1LDB \
#     motif_scaffolding.motif_list=[01_1LDB] \
#     motif_scaffolding.max_samples_per_motif=5 \
#     model_name=RFD3 \
#     gpus=0

set -e
BENCHCORE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESIGN_DIR="${1:-$BENCHCORE_ROOT/../Motif_Benchmark/20260206_RFD3_ouputs/rfd3_outputs/01_1LDB}"
cd "$BENCHCORE_ROOT"

echo "BenchCore root: $BENCHCORE_ROOT"
echo "Design dir:     $DESIGN_DIR"
echo "Running motif scaffolding (01_1LDB, first 5 samples)..."

python scripts/run_motif_scaffolding_pipeline.py \
  design_dir="$DESIGN_DIR" \
  motif_scaffolding.motif_list=[01_1LDB] \
  motif_scaffolding.max_samples_per_motif=5 \
  model_name=RFD3 \
  gpus=0

echo "Done. Check: $BENCHCORE_ROOT/motif_scaffolding_results/..."
