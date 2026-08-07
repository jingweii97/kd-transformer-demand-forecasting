#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=teacher_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/teacher_audit_%j.out
#SBATCH --error=logs/slurm/teacher_audit_%j.err

set -e

# Match the established teacher-job behavior in the Slurm working directory.
if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)"/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

echo "STARTING READ-ONLY TEACHER-STRENGTH AUDIT"
echo "SPLIT: validation only"
echo "TRAINING: disabled"
echo "HELD-OUT SET: disabled"

python scripts/audit_teacher_strength.py --env dicc --experiment full "$@"
