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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Match the established teacher-job behavior: set up/reuse the cluster scratch
# workspace before resolving existing checkpoint and artifact paths.
if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    mkdir -p logs/slurm
    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 9
            bash scripts/slurm/setup_hpc_storage.sh
        ) 9>logs/slurm/.hpc_storage_setup.lock
    else
        bash scripts/slurm/setup_hpc_storage.sh
    fi
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)"/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs/slurm
echo "STARTING READ-ONLY TEACHER-STRENGTH AUDIT"
echo "SPLIT: validation only"
echo "TRAINING: disabled"
echo "HELD-OUT SET: disabled"

python scripts/audit_teacher_strength.py --env dicc --experiment full "$@"
