#!/usr/bin/env python3
"""
Generate all .sh files for different models, W/A combinations, and clustering algorithms.
"""

models = {
    'resnet18': ('r18', 'ResNet18'),
    'resnet50': ('r50', 'ResNet50'),
    'mobilenetv2': ('mbv2', 'MobileNetV2'),
    'mnasnet': ('mnas', 'MNasNet')
}

wa_combos = [(2, 2), (2, 4), (4, 2), (4, 4)]
algorithms = ['kmeans', 'minibatch_kmeans', 'gmm', 'dbscan', 'agglomerative', 'spectral']

def create_sh_file(model_name, short_name, display_name, w_bits, a_bits, algo):
    """Create a single .sh file for the given configuration."""
    filename = f'run_{short_name}_w{w_bits}a{a_bits}_{algo}.sh'
    algo_upper = algo.upper().replace('_', '_')
    job_name = f'{short_name.upper()}_W{w_bits}A{a_bits}_{algo_upper}'
    
    content = f'''#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting {display_name} W{w_bits}A{a_bits} {algo_upper} experiment at $(date)"
python run_script_seed.py {model_name} --w_bits {w_bits} --a_bits {a_bits} --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.6 --num_clusters 1 8 16 24 32 48 64 96 128 192 256 --pca_dim 50 --clustering_algorithm {algo}
echo "Completed {display_name} W{w_bits}A{a_bits} {algo_upper} experiment at $(date)"
'''
    
    with open(filename, 'w') as f:
        f.write(content)
    return filename

if __name__ == '__main__':
    import os
    
    print("Generating .sh files for all model+W/A+algorithm combinations...")
    created_files = []
    skipped_files = []
    
    for model_name, (short_name, display_name) in models.items():
        for w_bits, a_bits in wa_combos:
            for algo in algorithms:
                filename = f'run_{short_name}_w{w_bits}a{a_bits}_{algo}.sh'
                
                # Skip if file already exists
                if os.path.exists(filename):
                    skipped_files.append(filename)
                    print(f"Skipped (exists): {filename}")
                    continue
                
                create_sh_file(model_name, short_name, display_name, w_bits, a_bits, algo)
                created_files.append(filename)
                print(f"Created: {filename}")
    
    print(f"\n✅ Total files created: {len(created_files)}")
    print(f"⏭️  Total files skipped (already exist): {len(skipped_files)}")
    print(f"   - Models: {len(models)}")
    print(f"   - W/A combinations: {len(wa_combos)}")
    print(f"   - Algorithms: {len(algorithms)}")
    print(f"   - Total possible: {len(models)} × {len(wa_combos)} × {len(algorithms)} = {len(models) * len(wa_combos) * len(algorithms)}")

