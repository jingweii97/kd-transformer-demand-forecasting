#!/bin/bash -l
# Student-WIS: WRMSSE-informed ground-truth supervision, no KD.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=student_wis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/student_wis_%j.out
#SBATCH --error=logs/slurm/student_wis_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/train_student.py" ]; then
    echo "Error: submit from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_NAME="student_wis"
OUTPUT_DIR="outputs/student/no_kd/${EXP_NAME}"
if [ -e "$OUTPUT_DIR" ]; then
    echo "Error: output directory already exists: $OUTPUT_DIR"
    echo "Refusing to overwrite or resume a retained-checkpoint experiment."
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

CMD=(python scripts/train_student.py --env dicc --experiment student_wis --exp-name "$EXP_NAME" --no-kd --supervised-loss wrmsse_informed)
echo "STUDENT-WIS: training-only WRMSSE-informed ground-truth supervision"
echo "checkpoint policy: top 5 by val_loss plus last.ckpt"
echo "command: ${CMD[*]}"
"${CMD[@]}"
