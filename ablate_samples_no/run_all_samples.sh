#!/bin/bash

echo "Starting all ResNet18 samples ablation experiments at $(date)"
echo "This will run experiments for sample counts: 1000, 5000, 10000, 100000, 200000, 500000, 1000000"
echo "Each sample count will be tested with 4 different bitwidth combinations: W2A2, W2A4, W4A2, W4A4"
echo "Each configuration will be run with 3 different seeds: 1001, 1002, 1003"
echo "Total experiments: 7 sample counts × 4 bitwidth combinations × 3 seeds = 84 experiments"
echo ""

# Define sample values and bitwidth combinations
sample_counts=(1000 5000 10000 100000 200000 500000 1000000)
bitwidth_combinations=("w2a2" "w2a4" "w4a2" "w4a4")

# Counter for tracking progress
total_experiments=$((${#sample_counts[@]} * ${#bitwidth_combinations[@]}))
current_experiment=0

for samples in "${sample_counts[@]}"; do
    for bitwidth in "${bitwidth_combinations[@]}"; do
        current_experiment=$((current_experiment + 1))
        script_name="samples_${samples}_resnet18_${bitwidth}.sh"
        
        echo "=========================================="
        echo "Experiment $current_experiment/$total_experiments: $script_name"
        echo "Sample count: $samples, Bitwidth: $bitwidth"
        echo "Seeds: 1001, 1002, 1003 (3 runs per script)"
        echo "=========================================="
        
        if [ -f "$script_name" ]; then
            echo "Submitting job: $script_name"
            sbatch "$script_name"
            echo "Job submitted successfully"
        else
            echo "ERROR: Script $script_name not found!"
        fi
        
        echo ""
        sleep 1  # Small delay between submissions
    done
done

echo "=========================================="
echo "All sample ablation experiments submitted!"
echo "Total scripts submitted: $total_experiments"
echo "Each script runs 3 seeds, so total experiments: $((total_experiments * 3))"
echo "Check job status with: squeue -u \$USER"
echo "Completed at $(date)"
echo "=========================================="
