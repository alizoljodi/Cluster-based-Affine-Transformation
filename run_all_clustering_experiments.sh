#!/bin/bash
#SBATCH -J ALL_CLUSTERING_EXPERIMENTS
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 7-00:00:00

# Usage: sbatch run_all_clustering_experiments.sh <venv_path>
# Example: sbatch run_all_clustering_experiments.sh /home/alz07xz/project/PD-Quant/pd_quant/bin/activate

if [ -z "$1" ]; then
    echo "Error: Virtual environment path is required"
    echo "Usage: $0 <venv_path>"
    echo "Example: $0 /home/alz07xz/project/PD-Quant/pd_quant/bin/activate"
    exit 1
fi

VENV_PATH="$1"
SCRIPT_DIR="runs_clustering_no_cluster"

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Directory $SCRIPT_DIR does not exist"
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH"

echo "=========================================="
echo "Starting all clustering experiments"
echo "Virtual environment: $VENV_PATH"
echo "Script directory: $SCRIPT_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Get all .sh files in the directory, sorted
sh_files=$(find "$SCRIPT_DIR" -name "*.sh" -type f | sort)

total_files=$(echo "$sh_files" | wc -l)
current=0
successful=0
failed=0

# Run each script
for script in $sh_files; do
    current=$((current + 1))
    script_name=$(basename "$script")
    
    echo ""
    echo "=========================================="
    echo "[$current/$total_files] Running: $script_name"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Extract the python command from the script
    # The script has: python run_script_seed.py ...
    python_cmd=$(grep "^python " "$script" | head -1)
    
    if [ -z "$python_cmd" ]; then
        echo "Warning: Could not find python command in $script_name, skipping..."
        failed=$((failed + 1))
        continue
    fi
    
    # Run the python command (venv is already activated)
    echo "Executing: $python_cmd"
    eval "$python_cmd"
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Successfully completed: $script_name"
        successful=$((successful + 1))
    else
        echo "❌ Failed: $script_name (exit code: $exit_code)"
        failed=$((failed + 1))
    fi
    
    echo "Completed at: $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Summary:"
echo "  Total scripts: $total_files"
echo "  Successful: $successful"
echo "  Failed: $failed"
echo "=========================================="
