#!/bin/bash -l
# Read-only arithmetic and cache-namespace preflight for Student-WIS/WIKD.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=student_wi_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/student_wi_preflight_%j.out
#SBATCH --error=logs/slurm/student_wi_preflight_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/audit_student_wrmsse_preflight.py" ]; then
    echo "Error: submit from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

CMD=(python scripts/audit_student_wrmsse_preflight.py --env dicc --soft-targets-dir artifacts/soft_targets --soft-targets-exp-name wi_e09_43841d3d_verified)
echo "READ-ONLY STUDENT-WIS/WIKD PREFLIGHT"
echo "command: ${CMD[*]}"
"${CMD[@]}"
