#!/usr/bin/env python3
"""Create all remaining .sh files that don't exist yet."""

import os

models = {
    'resnet50': ('r50', 'ResNet50'),
    'mobilenetv2': ('mbv2', 'MobileNetV2'),
    'mnasnet': ('mnas', 'MNasNet')
}

wa_combos = [(2, 2), (2, 4), (4, 2), (4, 4)]
algorithms = ['kmeans', 'minibatch_kmeans', 'gmm', 'dbscan', 'agglomerative', 'spectral']

# ResNet50: skip W2A2 (already created), create W2A4, W4A2, W4A4
r50_wa = [(2, 4), (4, 2), (4, 4)]

# MobileNetV2 and MNasNet: create all combinations
mbv2_wa = [(2, 2), (2, 4), (4, 2), (4, 4)]
mnas_wa = [(2, 2), (2, 4), (4, 2), (4, 4)]

def create_file(model_name, short_name, display_name, w_bits, a_bits, algo):
    filename = f'run_{short_name}_w{w_bits}a{a_bits}_{algo}.sh'
    if os.path.exists(filename):
        return False, filename
    
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
    return True, filename

created = []
skipped = []

# ResNet50: W2A4, W4A2, W4A4
for w_bits, a_bits in r50_wa:
    for algo in algorithms:
        success, filename = create_file('resnet50', 'r50', 'ResNet50', w_bits, a_bits, algo)
        if success:
            created.append(filename)
            print(f"Created: {filename}")
        else:
            skipped.append(filename)

# MobileNetV2: all combinations
for w_bits, a_bits in mbv2_wa:
    for algo in algorithms:
        success, filename = create_file('mobilenetv2', 'mbv2', 'MobileNetV2', w_bits, a_bits, algo)
        if success:
            created.append(filename)
            print(f"Created: {filename}")
        else:
            skipped.append(filename)

# MNasNet: all combinations
for w_bits, a_bits in mnas_wa:
    for algo in algorithms:
        success, filename = create_file('mnasnet', 'mnas', 'MNasNet', w_bits, a_bits, algo)
        if success:
            created.append(filename)
            print(f"Created: {filename}")
        else:
            skipped.append(filename)

print(f"\n✅ Created: {len(created)} files")
print(f"⏭️  Skipped (already exist): {len(skipped)} files")
print(f"📊 Total: {len(created) + len(skipped)} files processed")

