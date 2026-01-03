"""
Experiment #1: Kernel Comparison Script

This script analyzes and compares results from multiple kernel runs
for the Hard Held-Out Views experiment.

Usage:
    python compare_kernels_exp1.py --results_dir ./out/gppvae_colab
"""

import os
import glob
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_kernel_name(run_dir):
    """Extract kernel name from directory name."""
    # Expected format: kernel_central_views_timestamp or kernel_random_timestamp
    match = re.match(r'(\w+)_(central_views|random)_(\d{8}_\d{6})', run_dir)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def load_run_metrics(run_path):
    """Load final metrics from a training run."""
    metrics = {}
    
    # Try to load from log.txt
    log_file = os.path.join(run_path, 'log.txt')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Parse last few lines for final metrics
            for line in lines[-20:]:
                if 'Final' in line and 'MSE' in line:
                    # Extract metrics from final summary
                    if 'training MSE:' in line:
                        metrics['mse_train'] = float(line.split(':')[-1].strip())
                    elif 'validation MSE:' in line:
                        metrics['mse_val'] = float(line.split(':')[-1].strip())
                    elif 'out-of-sample MSE:' in line:
                        metrics['mse_out'] = float(line.split(':')[-1].strip())
                elif 'Learned Lengthscale:' in line:
                    metrics['lengthscale'] = float(line.split(':')[-1].strip())
    
    # Check for weights (to verify training completed)
    weights_dir = os.path.join(run_path, 'weights')
    if os.path.exists(weights_dir):
        vae_weights = glob.glob(os.path.join(weights_dir, 'vae_weights.*.pt'))
        metrics['num_checkpoints'] = len(vae_weights)
    
    return metrics


def find_all_runs(results_dir):
    """Find all experiment runs in the results directory."""
    runs = []
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return runs
    
    # Look for subdirectories
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            kernel_name, mode, timestamp = extract_kernel_name(item)
            if kernel_name and mode == 'central_views':
                # This is an Experiment #1 run
                metrics = load_run_metrics(item_path)
                runs.append({
                    'kernel': kernel_name,
                    'mode': mode,
                    'timestamp': timestamp,
                    'path': item_path,
                    'dir_name': item,
                    **metrics
                })
    
    return runs


def compute_extreme_angle_metrics(run_path):
    """
    Compute metrics specifically for extreme angles.
    This requires parsing per-view metrics if available.
    """
    # Extreme view indices for Experiment #1
    extreme_views = [0, 1, 2, 6, 7, 8]  # ±45°, ±60°, ±90°
    
    # Try to find per-view metrics from saved data
    # (This would need to be saved during training)
    metrics_file = os.path.join(run_path, 'per_view_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            per_view = json.load(f)
            if 'mse_out_per_view' in per_view:
                extreme_mses = [per_view['mse_out_per_view'][str(v)] 
                               for v in extreme_views 
                               if str(v) in per_view['mse_out_per_view']]
                if extreme_mses:
                    return {
                        'mse_out_extreme_avg': np.mean(extreme_mses),
                        'mse_out_extreme_std': np.std(extreme_mses),
                        'mse_out_extreme_max': np.max(extreme_mses),
                    }
    
    return {}


def create_comparison_table(runs):
    """Create a formatted comparison table."""
    if not runs:
        print("❌ No runs found to compare")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(runs)
    
    # Sort by MSE_out (ascending - lower is better)
    if 'mse_out' in df.columns:
        df = df.sort_values('mse_out')
    
    print("\n" + "=" * 100)
    print("Kernel Comparison: Hard Held-Out Views Experiment")
    print("=" * 100)
    print(f"\nTotal runs found: {len(df)}")
    print(f"Experiment: Train on central views (±30°), test on extreme angles (±45°, ±60°, ±90°)")
    
    if df.empty:
        print("\n❌ No valid runs with metrics found")
        return
    
    print("\n📊 Results Table:")
    print("-" * 100)
    
    # Format table
    header = f"{'Kernel':<15} | {'MSE_train':<10} | {'MSE_val':<10} | {'MSE_out':<10} | {'Lengthscale':<12} | {'Checkpoints':<12}"
    print(header)
    print("-" * 100)
    
    for _, row in df.iterrows():
        kernel = row['kernel']
        mse_train = f"{row.get('mse_train', float('nan')):.6f}" if 'mse_train' in row else "N/A"
        mse_val = f"{row.get('mse_val', float('nan')):.6f}" if 'mse_val' in row else "N/A"
        mse_out = f"{row.get('mse_out', float('nan')):.6f}" if 'mse_out' in row else "N/A"
        lengthscale = f"{row.get('lengthscale', float('nan')):.3f}" if 'lengthscale' in row else "N/A"
        checkpoints = f"{row.get('num_checkpoints', 0)}" if 'num_checkpoints' in row else "0"
        
        print(f"{kernel:<15} | {mse_train:<10} | {mse_val:<10} | {mse_out:<10} | {lengthscale:<12} | {checkpoints:<12}")
    
    print("-" * 100)
    
    # Highlight winner
    if 'mse_out' in df.columns and not df['mse_out'].isna().all():
        best_kernel = df.loc[df['mse_out'].idxmin(), 'kernel']
        best_mse = df['mse_out'].min()
        worst_mse = df['mse_out'].max()
        improvement = (worst_mse / best_mse) if best_mse > 0 else float('inf')
        
        print(f"\n🏆 Winner: {best_kernel.upper()} (MSE_out: {best_mse:.6f})")
        print(f"   Best vs Worst: {improvement:.2f}× better!")
        
        # Structured kernels analysis
        structured = ['periodic', 'vonmises', 'matern', 'rbf']
        unstructured = ['legacy', 'fullrank']
        
        structured_runs = df[df['kernel'].isin(structured)]
        unstructured_runs = df[df['kernel'].isin(unstructured)]
        
        if not structured_runs.empty and not unstructured_runs.empty:
            best_structured = structured_runs['mse_out'].min()
            best_unstructured = unstructured_runs['mse_out'].min()
            advantage = (best_unstructured / best_structured) if best_structured > 0 else float('inf')
            
            print(f"\n📈 Structured vs Unstructured Kernels:")
            print(f"   Best structured (Periodic/VonMises/Matérn/RBF): {best_structured:.6f}")
            print(f"   Best unstructured (Legacy/FullRank): {best_unstructured:.6f}")
            print(f"   Structured kernel advantage: {advantage:.2f}× better!")
            
            if advantage > 2.0:
                print(f"   ✅ Hypothesis CONFIRMED: Structured kernels >> Unstructured for extrapolation!")
            elif advantage > 1.2:
                print(f"   ✓ Hypothesis supported: Structured kernels better for extrapolation")
            else:
                print(f"   ⚠️  Hypothesis weak: Similar performance between structured/unstructured")
    
    print("\n" + "=" * 100)
    
    # Save to CSV
    output_file = 'exp1_kernel_comparison.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare kernel results for Experiment #1')
    parser.add_argument('--results_dir', type=str, default='./out/gppvae_colab',
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    print("🔍 Searching for Experiment #1 runs...")
    print(f"   Results directory: {args.results_dir}")
    
    runs = find_all_runs(args.results_dir)
    
    if runs:
        print(f"\n✅ Found {len(runs)} experiment run(s)")
        for run in runs:
            print(f"   • {run['kernel']:12s} ({run['timestamp']})")
    else:
        print("\n❌ No Experiment #1 runs found!")
        print("   Make sure you've trained models with view_split_mode='by_view'")
        return
    
    create_comparison_table(runs)


if __name__ == '__main__':
    main()
