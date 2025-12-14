#!/bin/bash

# Compact version - submits all jobs with minimal output
echo "🚀 Submitting all 36 ResNet18 ablation experiments..."

# Create logs directory
mkdir -p logs

# Submit all jobs
for script in ablate_init/run_r18_*.sh; do
    script_name=$(basename "$script")
    output_name="${script_name%.sh}_%j.out"
    echo -n "Submitting $script_name... "
    sbatch --output="logs/$output_name" "$script" && echo "✅" || echo "❌"
done

echo "🎉 All jobs submitted! Check logs/ directory for outputs."
echo "Monitor with: squeue -u \$USER"


