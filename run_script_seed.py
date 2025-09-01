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
            return None
        
        end_idx = stdout.find(end_marker, start_idx)
        if end_idx == -1:
            # Take rest of output if no end marker
            df_section = stdout[start_idx:]
        else:
            df_section = stdout[start_idx:end_idx]
        
        # Extract the actual DataFrame content (skip header lines with ===)
        lines = df_section.split('\n')
        df_lines = []
        header_found = False
        
        for line in lines:
            line = line.strip()
            if not line or '===' in line:
                continue
            # Skip the "Results saved to:" line
            if 'Results saved to:' in line or '.csv' in line:
                continue
            if 'method' in line and 'num_clusters' in line:  # Header line
                header_found = True
                df_lines.append(line)
            elif header_found and line:
                # Validate that this line contains valid numeric data
                parts = line.split()
                if len(parts) >= 5:  # Should have at least method, num_clusters, pca_dim, alpha, top1_acc
                    try:
                        # Check if top1_acc (4th column) is a valid number
                        float(parts[4])
                        # If we get here, it's a valid numeric row
                        df_lines.append(line)
                    except (ValueError, IndexError):
                        # Skip this line if it doesn't contain valid numeric data
                        continue
        
        if len(df_lines) < 2:  # Need at least header + 1 data row
            return None
        
        # Create DataFrame from the parsed lines
        df_text = '\n'.join(df_lines)
        df = pd.read_csv(io.StringIO(df_text), sep=r'\s+', engine='python')
        
        # Additional validation: ensure numeric columns are actually numeric
        if 'top1_acc' in df.columns:
            df['top1_acc'] = pd.to_numeric(df['top1_acc'], errors='coerce')
        if 'top5_acc' in df.columns:
            df['top5_acc'] = pd.to_numeric(df['top5_acc'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Error parsing DataFrame: {e}")
        return None


def run_one_seed(
    exp_name: str,
    data_path: str,
    arch: str,
    w_bits: int,
    a_bits: int,
    seed: int,
    extra_sleep: float,
) -> Tuple[Optional[pd.DataFrame], int, str]:
    weight, T, lamb_c = arch_hparams(exp_name)
    cmd = (
        f"python main_imagenet.py --data_path {data_path} --arch {arch} "
        f"--n_bits_w {w_bits} --n_bits_a {a_bits} --weight {weight} --T {T} --lamb_c {lamb_c} --seed {seed}"
    )
    print(f"[seed={seed}] Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        print(f"[seed={seed}] Process exited with code {proc.returncode}. STDERR:\n{stderr}")
        df = None
    else:
        print(f"[seed={seed}] Completed. Parsing DataFrame...")
        df = parse_dataframe_from_output(stdout)
        if df is not None:
            print(f"[seed={seed}] Parsed DataFrame with {len(df)} rows")
        else:
            print(f"[seed={seed}] Failed to parse DataFrame")
    time.sleep(extra_sleep)
    return df, proc.returncode, stdout


def analyze_results(dataframes: List[Optional[pd.DataFrame]], seeds: List[int]) -> None:
    """Analyze and print results from multiple seed runs."""
    # Combine all DataFrames
    valid_dfs = [df for df in dataframes if df is not None]
    if not valid_dfs:
        print("No valid results to analyze!")
        return
    
    # Concatenate all DataFrames
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    
    print("\n" + "="*100)
    print("COMBINED RESULTS FROM ALL SEEDS:")
    print("="*100)
    print(combined_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("AGGREGATED STATISTICS (MEAN ± STD):")
    print("="*100)
    
    # Group by method and configuration
    grouped = combined_df.groupby(['method', 'num_clusters', 'pca_dim', 'alpha'])
    
    results_summary = []
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
        
        results_summary.append({
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
        })
    
    # Print summary table
    summary_df = pd.DataFrame(results_summary)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=ARCH_CHOICES)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="/datasets/imagenet")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds to sleep between runs")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    dataframes: List[Optional[pd.DataFrame]] = []

    for seed in seeds:
        df, code, _ = run_one_seed(
            exp_name=args.exp_name,
            data_path=args.data_path,
            arch=args.exp_name,
            w_bits=args.w_bits,
            a_bits=args.a_bits,
            seed=seed,
            extra_sleep=args.sleep,
        )
        dataframes.append(df)

    analyze_results(dataframes, seeds)


