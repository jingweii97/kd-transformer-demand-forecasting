#!/bin/bash -l
# Evaluate every unique retained Student-WIS checkpoint, then write its selection manifest.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=select_student_wis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/select_student_wis_%j.out
#SBATCH --error=logs/slurm/select_student_wis_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/evaluate_select_student_checkpoints.py" ]; then
    echo "Error: submit from repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

RUN_DIR="outputs/student/no_kd/student_wis"
OUT_DIR="$RUN_DIR/common_validation_evaluation"
if [ ! -d "$RUN_DIR" ] || ! compgen -G "$RUN_DIR/*.ckpt" > /dev/null; then
    echo "Error: Student-WIS retained checkpoints are missing: $RUN_DIR"
    exit 1
fi
if [ -e "$OUT_DIR" ]; then
    echo "Error: refusing to overwrite existing selection output: $OUT_DIR"
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

CMD=(python scripts/evaluate_select_student_checkpoints.py --env dicc --experiment full --run-dir "$RUN_DIR" --variant-label "Student-WIS")
echo "VALIDATION ONLY: retained Student-WIS checkpoint selection"
echo "criterion: lowest exact common-validation full-hierarchy WRMSSE"
echo "command: ${CMD[*]}"
"${CMD[@]}"
