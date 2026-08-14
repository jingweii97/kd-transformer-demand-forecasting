#!/bin/bash -l
# Select Teacher-v2 exclusively by mean exact WRMSSE across d1520..d1526.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=select_phase_teacher_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/select_phase_teacher_v2_%j.out
#SBATCH --error=logs/slurm/select_phase_teacher_v2_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
RUN="outputs/teacher/phase_teacher_v2"
[[ -d "$RUN" && -f "$RUN/last.ckpt" ]] || { echo "Missing retained Teacher-v2 checkpoints"; exit 1; }
[[ ! -e "$RUN/phase_validation_evaluation" ]] || { echo "Refusing selection overwrite"; exit 1; }
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/evaluate_phase_validation_checkpoints.py --env dicc --experiment phase_teacher_v2 --run-dir "$RUN" --variant-label phase_teacher_v2 --model-type teacher --origins-file configs/origins/phase_balanced_validation_v2.json
