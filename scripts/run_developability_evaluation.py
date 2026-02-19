#!/usr/bin/env python3
"""
Script to run antibody developability evaluation using DevelopabilityScorer.

Usage:
    python run_developability_evaluation.py \
        --csv_file antibodies_fv.csv \
        --results_dir ./af3_results \
        --output developability_metrics.csv
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.developability import DevelopabilityScorer


def main():
    parser = argparse.ArgumentParser(
        description='Calculate antibody developability metrics (TAP/TNP)'
    )
    parser.add_argument(
        'csv_file',
        help='Input CSV file with CDR indices (e.g., antibodies_fv.csv)'
    )
    parser.add_argument(
        'results_dir',
        help='Root directory containing AF3 results (e.g., ./af3_results)'
    )
    parser.add_argument(
        '--output', '-o',
        default='developability_metrics.csv',
        help='Output CSV file (default: developability_metrics.csv)'
    )
    parser.add_argument(
        '--antibody-id', '-i',
        default=None,
        help='Process only a specific antibody ID (for testing)'
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=4,
        help='Number of seeds to process per antibody (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Load CSV
    df = pd.read_csv(args.csv_file)
    print(f"📖 Reading CSV file: {args.csv_file}")
    print(f"   Found {len(df)} antibodies")
    
    if args.antibody_id:
        df = df[df['id'] == args.antibody_id]
        if len(df) == 0:
            print(f"❌ Antibody {args.antibody_id} not found in CSV")
            return
        print(f"   Processing only: {args.antibody_id}")
    
    # Initialize scorer
    scorer = DevelopabilityScorer()
    results = []
    
    print(f"\n🔬 Processing {len(df)} antibodies...")
    
    for idx, row in df.iterrows():
        ab_id = row['id']
        print(f"\n[{idx+1}/{len(df)}] Processing {ab_id}...")
        
        # Detect mode
        mode = scorer.detect_mode(row)
        print(f"   Mode: {mode}")
        
        # Find CIF files
        search_path = os.path.join(args.results_dir, f"**/{ab_id}/**/model.cif")
        files = sorted([f for f in glob.glob(search_path, recursive=True) if 'seed-' in f])
        
        if not files:
            print(f"   ⚠️  No CIF files found for {ab_id}")
            # Create default result based on mode
            default_result = {
                'id': ab_id,
                'Mode': mode,
                'PSH': np.nan, 'PPC': np.nan, 'PNC': np.nan,
                'Total_CDR_Length': np.nan
            }
            if mode == 'TAP':
                default_result['SFvCSP'] = np.nan
            else:
                default_result['CDR3_Length'] = np.nan
                default_result['CDR3_Compactness'] = np.nan
            results.append(default_result)
            continue
        
        if len(files) < args.num_seeds:
            print(f"   ⚠️  Only {len(files)} CIF files found (expected {args.num_seeds})")
        
        # Process seeds
        ab_res = defaultdict(list)
        for f in files[:args.num_seeds]:
            m = scorer.calculate_metrics(f, row)
            if m:
                for k, v in m.items():
                    if k != 'Mode':  # Mode is the same for all seeds
                        ab_res[k].append(v)
        
        # Average across seeds
        entry = {'id': ab_id}
        if ab_res:
            entry['Mode'] = mode
            for k in ab_res.keys():
                vals = ab_res[k]
                if k == 'CDR3_Length':
                    # CDR3_Length should be the same across seeds, use first value
                    entry[k] = vals[0] if vals else np.nan
                elif k.endswith('_Category'):
                    # Categories: use most common (mode)
                    if vals:
                        from collections import Counter
                        most_common = Counter(vals).most_common(1)[0][0]
                        entry[k] = most_common
                    else:
                        entry[k] = 'Unknown'
                else:
                    entry[k] = np.nanmean(vals) if vals else np.nan
        else:
            # No valid results
            entry['Mode'] = mode
            entry.update({k: np.nan for k in ['PSH', 'PPC', 'PNC', 'Total_CDR_Length']})
            if mode == 'TAP':
                entry['SFvCSP'] = np.nan
            else:
                entry['CDR3_Length'] = np.nan
                entry['CDR3_Compactness'] = np.nan
        
        results.append(entry)
        
        # Format output
        psh_str = f"{entry.get('PSH', np.nan):.2f}" if not np.isnan(entry.get('PSH', np.nan)) else "NaN"
        ppc_str = f"{entry.get('PPC', np.nan):.0f}" if not np.isnan(entry.get('PPC', np.nan)) else "NaN"
        pnc_str = f"{entry.get('PNC', np.nan):.0f}" if not np.isnan(entry.get('PNC', np.nan)) else "NaN"
        
        if mode == 'TAP':
            sfvcsp_str = f"{entry.get('SFvCSP', np.nan):.2f}" if not np.isnan(entry.get('SFvCSP', np.nan)) else "NaN"
            print(f"   PSH: {psh_str}, PPC: {ppc_str}, PNC: {pnc_str}, SFvCSP: {sfvcsp_str}")
        else:
            cdr3_len_str = f"{entry.get('CDR3_Length', np.nan):.0f}" if not np.isnan(entry.get('CDR3_Length', np.nan)) else "NaN"
            compact_str = f"{entry.get('CDR3_Compactness', np.nan):.3f}" if not np.isnan(entry.get('CDR3_Compactness', np.nan)) else "NaN"
            print(f"   PSH: {psh_str}, PPC: {ppc_str}, PNC: {pnc_str}, CDR3_Length: {cdr3_len_str}, CDR3_Compactness: {compact_str}")
        
        if (idx + 1) % 10 == 0:
            print(f"\n   ✅ Progress: {idx+1}/{len(df)} antibodies processed")
    
    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to {args.output}")
    print(f"   Processed {len(results)} antibodies")
    
    # Print summary
    if len(result_df) > 0:
        print(f"\n📊 Summary:")
        print(f"   TAP (mAb): {len(result_df[result_df['Mode'] == 'TAP'])}")
        print(f"   TNP (Nanobody): {len(result_df[result_df['Mode'] == 'TNP'])}")


if __name__ == "__main__":
    main()
