#!/bin/bash

# Master script to run all ResNet18 ablation experiments
# This script submits all ablation jobs with descriptive output names

echo "=========================================="
echo "🚀 Starting ResNet18 Ablation Experiments"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Total experiments: 36 (4 bit configs × 9 init mode combinations)"
echo "=========================================="

# Create output directory for logs
mkdir -p logs

# Function to submit job with descriptive output name
submit_job() {
    local script_name=$1
    local output_name=$2
    
    echo "📤 Submitting: $script_name"
    echo "   Output: logs/$output_name"
    
    sbatch --output="logs/$output_name" "ablate_init/$script_name"
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Successfully submitted"
    else
        echo "   ❌ Failed to submit"
    fi
    echo ""
}

# W2A2 Experiments (9 combinations)
echo "🔬 W2A2 Experiments (Weight=2bit, Activation=2bit)"
echo "----------------------------------------"

submit_job "run_r18_w2a2_mse_mse.sh" "r18_w2a2_mse_mse_%j.out"
submit_job "run_r18_w2a2_mse_minmax.sh" "r18_w2a2_mse_minmax_%j.out"
submit_job "run_r18_w2a2_mse_minmax_scale.sh" "r18_w2a2_mse_minmax_scale_%j.out"
submit_job "run_r18_w2a2_minmax_mse.sh" "r18_w2a2_minmax_mse_%j.out"
submit_job "run_r18_w2a2_minmax_minmax.sh" "r18_w2a2_minmax_minmax_%j.out"
submit_job "run_r18_w2a2_minmax_minmax_scale.sh" "r18_w2a2_minmax_minmax_scale_%j.out"
submit_job "run_r18_w2a2_minmax_scale_mse.sh" "r18_w2a2_minmax_scale_mse_%j.out"
submit_job "run_r18_w2a2_minmax_scale_minmax.sh" "r18_w2a2_minmax_scale_minmax_%j.out"
submit_job "run_r18_w2a2_minmax_scale_minmax_scale.sh" "r18_w2a2_minmax_scale_minmax_scale_%j.out"

echo "🔬 W2A4 Experiments (Weight=2bit, Activation=4bit)"
echo "----------------------------------------"

submit_job "run_r18_w2a4_mse_mse.sh" "r18_w2a4_mse_mse_%j.out"
submit_job "run_r18_w2a4_mse_minmax.sh" "r18_w2a4_mse_minmax_%j.out"
submit_job "run_r18_w2a4_mse_minmax_scale.sh" "r18_w2a4_mse_minmax_scale_%j.out"
submit_job "run_r18_w2a4_minmax_mse.sh" "r18_w2a4_minmax_mse_%j.out"
submit_job "run_r18_w2a4_minmax_minmax.sh" "r18_w2a4_minmax_minmax_%j.out"
submit_job "run_r18_w2a4_minmax_minmax_scale.sh" "r18_w2a4_minmax_minmax_scale_%j.out"
submit_job "run_r18_w2a4_minmax_scale_mse.sh" "r18_w2a4_minmax_scale_mse_%j.out"
submit_job "run_r18_w2a4_minmax_scale_minmax.sh" "r18_w2a4_minmax_scale_minmax_%j.out"
submit_job "run_r18_w2a4_minmax_scale_minmax_scale.sh" "r18_w2a4_minmax_scale_minmax_scale_%j.out"

echo "🔬 W4A2 Experiments (Weight=4bit, Activation=2bit)"
echo "----------------------------------------"

submit_job "run_r18_w4a2_mse_mse.sh" "r18_w4a2_mse_mse_%j.out"
submit_job "run_r18_w4a2_mse_minmax.sh" "r18_w4a2_mse_minmax_%j.out"
submit_job "run_r18_w4a2_mse_minmax_scale.sh" "r18_w4a2_mse_minmax_scale_%j.out"
submit_job "run_r18_w4a2_minmax_mse.sh" "r18_w4a2_minmax_mse_%j.out"
submit_job "run_r18_w4a2_minmax_minmax.sh" "r18_w4a2_minmax_minmax_%j.out"
submit_job "run_r18_w4a2_minmax_minmax_scale.sh" "r18_w4a2_minmax_minmax_scale_%j.out"
submit_job "run_r18_w4a2_minmax_scale_mse.sh" "r18_w4a2_minmax_scale_mse_%j.out"
submit_job "run_r18_w4a2_minmax_scale_minmax.sh" "r18_w4a2_minmax_scale_minmax_%j.out"
submit_job "run_r18_w4a2_minmax_scale_minmax_scale.sh" "r18_w4a2_minmax_scale_minmax_scale_%j.out"

echo "🔬 W4A4 Experiments (Weight=4bit, Activation=4bit)"
echo "----------------------------------------"

submit_job "run_r18_w4a4_mse_mse.sh" "r18_w4a4_mse_mse_%j.out"
submit_job "run_r18_w4a4_mse_minmax.sh" "r18_w4a4_mse_minmax_%j.out"
submit_job "run_r18_w4a4_mse_minmax_scale.sh" "r18_w4a4_mse_minmax_scale_%j.out"
submit_job "run_r18_w4a4_minmax_mse.sh" "r18_w4a4_minmax_mse_%j.out"
submit_job "run_r18_w4a4_minmax_minmax.sh" "r18_w4a4_minmax_minmax_%j.out"
submit_job "run_r18_w4a4_minmax_minmax_scale.sh" "r18_w4a4_minmax_minmax_scale_%j.out"
submit_job "run_r18_w4a4_minmax_scale_mse.sh" "r18_w4a4_minmax_scale_mse_%j.out"
submit_job "run_r18_w4a4_minmax_scale_minmax.sh" "r18_w4a4_minmax_scale_minmax_%j.out"
submit_job "run_r18_w4a4_minmax_scale_minmax_scale.sh" "r18_w4a4_minmax_scale_minmax_scale_%j.out"

echo "=========================================="
echo "🎉 All 36 ablation experiments submitted!"
echo "=========================================="
echo "📁 Logs will be saved in: logs/"
echo "📊 Monitor jobs with: squeue -u \$USER"
echo "📋 Check specific job with: scontrol show job <job_id>"
echo "=========================================="
echo "Completed at: $(date)"