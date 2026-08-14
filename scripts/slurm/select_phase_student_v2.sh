#!/bin/bash -l
# Usage: sbatch scripts/slurm/select_phase_student_v2.sh <phase_wis_v2|phase_wikd_v2> <no_kd|kd>
#SBATCH --partition=gpu-a100
#SBATCH --job-name=select_phase_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/select_phase_v2_%j.out
#SBATCH --error=logs/slurm/select_phase_v2_%j.err
set -euo pipefail
[[ "$#" == 2 ]] || { echo "Usage: $0 <name> <no_kd|kd>"; exit 2; }
cd "${SLURM_SUBMIT_DIR:-$PWD}"
NAME="$1"; MODE="$2"; RUN="outputs/student/${MODE}/${NAME}"
[[ -d "$RUN" && -f "$RUN/last.ckpt" ]] || { echo "Missing retained checkpoints: $RUN"; exit 1; }
[[ ! -e "$RUN/phase_validation_evaluation" ]] || { echo "Refusing selection overwrite"; exit 1; }
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/evaluate_phase_validation_checkpoints.py --env dicc --experiment "$NAME" --run-dir "$RUN" --variant-label "$NAME" --model-type student --origins-file configs/origins/phase_balanced_validation_v2.json
