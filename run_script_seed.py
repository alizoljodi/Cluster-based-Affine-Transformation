import argparse
import subprocess
import time
import re
import statistics
import pandas as pd
import io
from typing import Optional, Tuple, List


ARCH_CHOICES = ['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet']


def arch_hparams(exp_name: str) -> Tuple[float, float, float]:
    """Return (weight, T, lamb_c) defaults following run_script.py for a given arch."""
    if exp_name in ("resnet18", "resnet50"):
        return 0.01, 4.0, 0.02
    if exp_name in ("regnetx_600m", "regnetx_3200m"):
        return 0.01, 4.0, 0.01
    if exp_name == "mobilenetv2":
        return 0.1, 1.0, 0.005
    if exp_name == "mnasnet":
        return 0.2, 1.0, 0.001
    # sensible default
    return 0.01, 4.0, 0.02


def parse_dataframe_from_output(stdout: str) -> Optional[pd.DataFrame]:
    """Parse the DataFrame from main_imagenet.py output."""
    try:
        # Look for the DataFrame section
        start_marker = "FINAL RESULTS DATAFRAME:"
        end_marker = "SUMMARY STATISTICS:"
        
        start_idx = stdout.find(start_marker)
        if start_idx == -1:
            print(f"❌ Could not find start marker '{start_marker}' in output")
            return None
        
        end_idx = stdout.find(end_marker, start_idx)
        if end_idx == -1:
            # Take rest of output if no end marker
            df_section = stdout[start_idx:]
            print(f"⚠️  No end marker '{end_marker}' found, using rest of output")
        else:
            df_section = stdout[start_idx:end_idx]
        
        # Extract the actual DataFrame content (skip header lines with ===)
        lines = df_section.split('\n')
        df_lines = []
        header_found = False
        valid_rows = 0
        skipped_rows = 0
        
        for line in lines:
            line = line.strip()
            if not line or '===' in line:
                continue
            # Skip the "Results saved to:" line
            if 'Results saved to:' in line or '.csv' in line:
                skipped_rows += 1
                continue
            if 'method' in line and 'num_clusters' in line:  # Header line
                header_found = True
                df_lines.append(line)
                print(f"📋 Found DataFrame header: {line}")
            elif header_found and line:
                # Validate that this line contains valid numeric data
                parts = line.split()
                if len(parts) >= 5:  # Should have at least method, num_clusters, pca_dim, alpha, top1_acc
                    try:
                        # Check if top1_acc (4th column) is a valid number
                        float(parts[4])
                        # If we get here, it's a valid numeric row
                        df_lines.append(line)
                        valid_rows += 1
                    except (ValueError, IndexError):
                        # Skip this line if it doesn't contain valid numeric data
                        skipped_rows += 1
                        continue
                else:
                    skipped_rows += 1
        
        print(f"📊 Parsing summary: {valid_rows} valid rows, {skipped_rows} skipped rows")
        
        if len(df_lines) < 2:  # Need at least header + 1 data row
            print(f"❌ Insufficient data: only {len(df_lines)} lines found (need at least 2)")
            return None
        
        # Create DataFrame from the parsed lines
        df_text = '\n'.join(df_lines)
        df = pd.read_csv(io.StringIO(df_text), sep=r'\s+', engine='python')
        
        # Additional validation: ensure numeric columns are actually numeric
        if 'top1_acc' in df.columns:
            df['top1_acc'] = pd.to_numeric(df['top1_acc'], errors='coerce')
        if 'top5_acc' in df.columns:
            df['top5_acc'] = pd.to_numeric(df['top5_acc'], errors='coerce')
        
        print(f"✅ Successfully created DataFrame with shape {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Error parsing DataFrame: {e}")
        return None


def run_one_seed(
    exp_name: str,
    data_path: str,
    arch: str,
    w_bits: int,
    a_bits: int,
    seed: int,
    extra_sleep: float,
    alpha: List[float],
    num_clusters: List[int],
    pca_dim: List[int],
) -> Tuple[Optional[pd.DataFrame], int, str]:
    print(f"\n{'='*80}")
    print(f"STARTING SEED {seed} EXPERIMENT")
    print(f"{'='*80}")
    print(f"[Seed {seed}] Configuration:")
    print(f"  - Architecture: {arch}")
    print(f"  - Weight bits: {w_bits}, Activation bits: {a_bits}")
    print(f"  - Alpha values: {alpha}")
    print(f"  - Number of clusters: {num_clusters}")
    print(f"  - PCA dimensions: {pca_dim}")
    print(f"  - Data path: {data_path}")
    
    weight, T, lamb_c = arch_hparams(exp_name)
    print(f"[Seed {seed}] Hyperparameters: weight={weight}, T={T}, lamb_c={lamb_c}")
    
    # Convert lists to space-separated strings for command line
    alpha_str = ' '.join(map(str, alpha))
    clusters_str = ' '.join(map(str, num_clusters))
    pca_str = ' '.join(map(str, pca_dim))
    
    cmd = (
        f"python main_imagenet.py --data_path {data_path} --arch {arch} "
        f"--n_bits_w {w_bits} --n_bits_a {a_bits} --weight {weight} --T {T} --lamb_c {lamb_c} --seed {seed} "
        f"--alpha {alpha_str} --num_clusters {clusters_str} --pca_dim {pca_str}"
    )
    print(f"[Seed {seed}] Executing command:")
    print(f"  {cmd}")
    
    start_time = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    execution_time = end_time - start_time
    
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    
    print(f"[Seed {seed}] Execution completed in {execution_time:.2f} seconds")
    
    if proc.returncode != 0:
        print(f"[Seed {seed}] ❌ FAILED - Process exited with code {proc.returncode}")
        print(f"[Seed {seed}] Error output:")
        print(f"{stderr}")
        df = None
    else:
        print(f"[Seed {seed}] ✅ SUCCESS - Process completed successfully")
        print(f"[Seed {seed}] Parsing results DataFrame...")
        df = parse_dataframe_from_output(stdout)
        if df is not None:
            print(f"[Seed {seed}] ✅ Successfully parsed DataFrame with {len(df)} rows")
            print(f"[Seed {seed}] DataFrame columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"[Seed {seed}] Methods tested: {df['method'].unique().tolist()}")
        else:
            print(f"[Seed {seed}] ❌ Failed to parse DataFrame from output")
            print(f"[Seed {seed}] Output preview (first 500 chars):")
            print(f"{stdout[:500]}...")
    
    print(f"[Seed {seed}] Sleeping for {extra_sleep} seconds before next seed...")
    time.sleep(extra_sleep)
    
    print(f"[Seed {seed}] Experiment finished")
    print(f"{'='*80}\n")
    
    return df, proc.returncode, stdout


def analyze_results(dataframes: List[Optional[pd.DataFrame]], seeds: List[int]) -> None:
    """Analyze and print results from multiple seed runs."""
    print(f"\n{'='*100}")
    print("ANALYZING RESULTS FROM MULTIPLE SEED RUNS")
    print(f"{'='*100}")
    
    # Combine all DataFrames
    valid_dfs = [df for df in dataframes if df is not None]
    failed_seeds = [seeds[i] for i, df in enumerate(dataframes) if df is None]
    
    print(f"📊 Analysis Summary:")
    print(f"  - Total seeds: {len(seeds)}")
    print(f"  - Successful runs: {len(valid_dfs)}")
    print(f"  - Failed runs: {len(failed_seeds)}")
    if failed_seeds:
        print(f"  - Failed seed numbers: {failed_seeds}")
    
    if not valid_dfs:
        print("❌ No valid results to analyze!")
        return
    
    # Concatenate all DataFrames
    print(f"\n🔗 Combining results from {len(valid_dfs)} successful runs...")
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"✅ Combined DataFrame shape: {combined_df.shape}")
    print(f"✅ Combined DataFrame columns: {list(combined_df.columns)}")
    
    print(f"\n📋 Combined results from all seeds:")
    print(f"{'='*100}")
    print(combined_df.to_string(index=False))
    
    print(f"\n📈 Computing aggregated statistics...")
    print(f"{'='*100}")
    
    # Group by method and configuration
    grouped = combined_df.groupby(['method', 'num_clusters', 'pca_dim', 'alpha'])
    
    results_summary = []
    baseline_stats = None
    
    for name, group in grouped:
        method, num_clusters, pca_dim, alpha = name
        
        # Calculate statistics
        top1_mean = group['top1_acc'].mean()
        top1_std = group['top1_acc'].std() if len(group) > 1 else 0.0
        top1_count = len(group)
        
        # Handle top5 (might have NaN values for baseline)
        top5_valid = group['top5_acc'].dropna()
        if len(top5_valid) > 0:
            top5_mean = top5_valid.mean()
            top5_std = top5_valid.std() if len(top5_valid) > 1 else 0.0
            top5_count = len(top5_valid)
        else:
            top5_mean = top5_std = top5_count = None
        
        result_entry = {
            'method': method,
            'num_clusters': num_clusters,
            'pca_dim': pca_dim,
            'alpha': alpha,
            'top1_mean': top1_mean,
            'top1_std': top1_std,
            'top1_count': top1_count,
            'top5_mean': top5_mean,
            'top5_std': top5_std,
            'top5_count': top5_count
        }
        
        # Store baseline stats separately
        if method == 'no_restoration':
            baseline_stats = result_entry
        
        results_summary.append(result_entry)
    
    # Print baseline statistics first
    if baseline_stats:
        print("\n" + "-" * 50)
        print("BASELINE (NO RESTORATION) STATISTICS:")
        print("-" * 50)
        print(f"Top-1 Accuracy: {baseline_stats['top1_mean']:.2f} ± {baseline_stats['top1_std']:.2f} (n={baseline_stats['top1_count']})")
        if baseline_stats['top5_mean'] is not None:
            print(f"Top-5 Accuracy: {baseline_stats['top5_mean']:.2f} ± {baseline_stats['top5_std']:.2f} (n={baseline_stats['top5_count']})")
        else:
            print("Top-5 Accuracy: N/A")
        print("-" * 50)
    
    # Print summary table for all configurations
    summary_df = pd.DataFrame(results_summary)
    
    print("\nALL CONFIGURATIONS:")
    print("Configuration\t\t\t\t\tTop-1 Acc\t\tTop-5 Acc")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        # Format configuration
        if row['method'] == 'no_restoration':
            config = "No Restoration (Baseline)"
        elif row['method'] == 'restoration':
            config = f"Restoration(clusters={row['num_clusters']}, pca={row['pca_dim']}, α={row['alpha']})"
        else:
            config = f"{row['method']}(clusters={row['num_clusters']}, pca={row['pca_dim']}, α={row['alpha']})"
        
        # Format top1
        top1_str = f"{row['top1_mean']:.2f}±{row['top1_std']:.2f} (n={row['top1_count']})"
        
        # Format top5
        if row['top5_mean'] is not None:
            top5_str = f"{row['top5_mean']:.2f}±{row['top5_std']:.2f} (n={row['top5_count']})"
        else:
            top5_str = "N/A"
        
        print(f"{config:<45}\t{top1_str:<20}\t{top5_str}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    print(f"\n{'='*100}")
    print("🚀 CAT MULTI-SEED EXPERIMENT RUNNER")
    print(f"{'='*100}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=ARCH_CHOICES)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="/home/alz07xz/imagenet")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds to sleep between runs")
    
    # CAT configuration parameters
    parser.add_argument("--alpha", type=float, nargs='+', default=[0.4], 
                        help="Alpha blending values for CAT evaluation")
    parser.add_argument("--num_clusters", type=int, nargs='+', default=[64], 
                        help="Number of clusters for CAT LUT building")
    parser.add_argument("--pca_dim", type=int, nargs='+', default=[-1], 
                        help="PCA dimensions; use -1 to disable PCA")
    
    args = parser.parse_args()

    print(f"🎯 Experiment Configuration:")
    print(f"  - Model: {args.exp_name}")
    print(f"  - Weight bits: {args.w_bits}, Activation bits: {args.a_bits}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1} ({args.num_seeds} total)")
    print(f"  - Sleep between runs: {args.sleep} seconds")
    print(f"  - Alpha values: {args.alpha}")
    print(f"  - Number of clusters: {args.num_clusters}")
    print(f"  - PCA dimensions: {args.pca_dim}")
    
    total_configs = len(args.alpha) * len(args.num_clusters) * len(args.pca_dim)
    print(f"  - Total configurations to test: {total_configs}")
    print(f"  - Total experiments: {args.num_seeds} seeds × {total_configs} configs = {args.num_seeds * total_configs}")
    
    print(f"\n⏰ Starting experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    dataframes: List[Optional[pd.DataFrame]] = []
    
    overall_start_time = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n🔄 Progress: {i+1}/{len(seeds)} seeds completed")
        df, code, _ = run_one_seed(
            exp_name=args.exp_name,
            data_path=args.data_path,
            arch=args.exp_name,
            w_bits=args.w_bits,
            a_bits=args.a_bits,
            seed=seed,
            extra_sleep=args.sleep,
            alpha=args.alpha,
            num_clusters=args.num_clusters,
            pca_dim=args.pca_dim,
        )
        dataframes.append(df)
    
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    print(f"\n{'='*100}")
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*100}")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📊 Results analysis starting...")

    analyze_results(dataframes, seeds)
    
    print(f"\n{'='*100}")
    print(f"🏁 EXPERIMENT RUNNER FINISHED")
    print(f"{'='*100}")


