#!/bin/bash

# Master script to run all ResNet18 ablation experiments
# This script submits all ablation jobs without custom output files

echo "=========================================="
echo "🚀 Starting ResNet18 Ablation Experiments"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Total experiments: 36 (4 bit configs × 9 init mode combinations)"
echo "=========================================="

# Function to submit job
submit_job() {
    local script_name=$1
    
    echo "📤 Submitting: $script_name"
    
    sbatch --output= "$script_name"
    
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

submit_job "run_r18_w2a2_mse_mse.sh"
submit_job "run_r18_w2a2_mse_minmax.sh"
submit_job "run_r18_w2a2_mse_minmax_scale.sh"
submit_job "run_r18_w2a2_minmax_mse.sh"
submit_job "run_r18_w2a2_minmax_minmax.sh"
submit_job "run_r18_w2a2_minmax_minmax_scale.sh"
submit_job "run_r18_w2a2_minmax_scale_mse.sh"
submit_job "run_r18_w2a2_minmax_scale_minmax.sh"
submit_job "run_r18_w2a2_minmax_scale_minmax_scale.sh"

echo "🔬 W2A4 Experiments (Weight=2bit, Activation=4bit)"
echo "----------------------------------------"

submit_job "run_r18_w2a4_mse_mse.sh"
submit_job "run_r18_w2a4_mse_minmax.sh"
submit_job "run_r18_w2a4_mse_minmax_scale.sh"
submit_job "run_r18_w2a4_minmax_mse.sh"
submit_job "run_r18_w2a4_minmax_minmax.sh"
submit_job "run_r18_w2a4_minmax_minmax_scale.sh"
submit_job "run_r18_w2a4_minmax_scale_mse.sh"
submit_job "run_r18_w2a4_minmax_scale_minmax.sh"
submit_job "run_r18_w2a4_minmax_scale_minmax_scale.sh"

echo "🔬 W4A2 Experiments (Weight=4bit, Activation=2bit)"
echo "----------------------------------------"

submit_job "run_r18_w4a2_mse_mse.sh"
submit_job "run_r18_w4a2_mse_minmax.sh"
submit_job "run_r18_w4a2_mse_minmax_scale.sh"
submit_job "run_r18_w4a2_minmax_mse.sh"
submit_job "run_r18_w4a2_minmax_minmax.sh"
submit_job "run_r18_w4a2_minmax_minmax_scale.sh"
submit_job "run_r18_w4a2_minmax_scale_mse.sh"
submit_job "run_r18_w4a2_minmax_scale_minmax.sh"
submit_job "run_r18_w4a2_minmax_scale_minmax_scale.sh"

echo "🔬 W4A4 Experiments (Weight=4bit, Activation=4bit)"
echo "----------------------------------------"

submit_job "run_r18_w4a4_mse_mse.sh"
submit_job "run_r18_w4a4_mse_minmax.sh"
submit_job "run_r18_w4a4_mse_minmax_scale.sh"
submit_job "run_r18_w4a4_minmax_mse.sh"
submit_job "run_r18_w4a4_minmax_minmax.sh"
submit_job "run_r18_w4a4_minmax_minmax_scale.sh"
submit_job "run_r18_w4a4_minmax_scale_mse.sh"
submit_job "run_r18_w4a4_minmax_scale_minmax.sh"
submit_job "run_r18_w4a4_minmax_scale_minmax_scale.sh"

echo "=========================================="
echo "🎉 All 36 ablation experiments submitted!"
echo "=========================================="
echo "📊 Monitor jobs with: squeue -u \$USER"
echo "📋 Check specific job with: scontrol show job <job_id>"
echo "=========================================="
echo "Completed at: $(date)"
