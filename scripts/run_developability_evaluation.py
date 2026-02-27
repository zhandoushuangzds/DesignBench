#!/usr/bin/env python3
"""
Standalone script for antibody developability evaluation. Uses the same API as
run_antibody_pipeline.py Step 5 (Evaluation.run_antibody_developability_evaluation).

Usage:
    python run_developability_evaluation.py <csv_file> <results_dir> [--output developability_metrics.csv] [--num-seeds 8]
    python run_developability_evaluation.py <csv_file> <results_dir> --antibody-id 01_7UXQ  # single design
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from evaluation.evaluation_api import Evaluation


def main():
    parser = argparse.ArgumentParser(
        description='Antibody developability evaluation (same as pipeline Step 5). '
                    'results_dir should contain refold/af3_out/ with AF3 CIFs.'
    )
    parser.add_argument('csv_file', help='CDR info CSV (e.g. cdr_info.csv)')
    parser.add_argument('results_dir', help='Pipeline root (contains refold/af3_out)')
    parser.add_argument('--output', '-o', default='developability_metrics.csv', help='Output CSV')
    parser.add_argument('--antibody-id', '-i', default=None, help='Process only this id (subset CSV)')
    parser.add_argument('--num-seeds', type=int, default=8, help='Max seeds per antibody')
    args = parser.parse_args()

    csv_file = os.path.abspath(args.csv_file)
    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isfile(csv_file):
        print(f"❌ CSV not found: {csv_file}")
        sys.exit(1)
    if not os.path.isdir(results_dir):
        print(f"❌ Results dir not found: {results_dir}")
        sys.exit(1)

    # Optional: filter to one antibody by writing a temp CSV
    if args.antibody_id:
        import pandas as pd
        df = pd.read_csv(csv_file)
        df = df[df['id'] == args.antibody_id]
        if df.empty:
            print(f"❌ Antibody id '{args.antibody_id}' not in CSV")
            sys.exit(1)
        fd, tmp = tempfile.mkstemp(suffix='.csv')
        try:
            os.close(fd)
            df.to_csv(tmp, index=False)
            csv_file = tmp
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    config = OmegaConf.create({})
    evaluator = Evaluation(config)
    evaluator.run_antibody_developability_evaluation(
        pipeline_dir=results_dir,
        cdr_info_csv=csv_file,
        output_csv=os.path.abspath(args.output),
        num_seeds=args.num_seeds,
    )

    if args.antibody_id:
        try:
            os.remove(tmp)
        except Exception:
            pass


if __name__ == '__main__':
    main()
