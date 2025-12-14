#!/bin/bash

# Script to check the status of all submitted ablation jobs

echo "=========================================="
echo "📊 ResNet18 Ablation Jobs Status"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

# Show queue status
echo "🔍 Current job queue:"
squeue -u $USER --format="%.8i %.12j %.8u %.2t %.10M %.6D %R" | head -1
squeue -u $USER --format="%.8i %.12j %.8u %.2t %.10M %.6D %R" | grep -E "(R18_|CAT_)"

echo ""
echo "📈 Job statistics:"
echo "Total jobs in queue: $(squeue -u $USER | tail -n +2 | wc -l)"
echo "Running jobs: $(squeue -u $USER -t R | tail -n +2 | wc -l)"
echo "Pending jobs: $(squeue -u $USER -t PD | tail -n +2 | wc -l)"

echo ""
echo "📁 Recent log files:"
if [ -d "logs" ]; then
    ls -lt logs/ | head -10
else
    echo "No logs directory found"
fi

echo ""
echo "💡 Useful commands:"
echo "  - Cancel all jobs: scancel -u \$USER"
echo "  - Cancel specific job: scancel <job_id>"
echo "  - View job details: scontrol show job <job_id>"
echo "  - View log: tail -f logs/<log_file>"
echo "=========================================="


