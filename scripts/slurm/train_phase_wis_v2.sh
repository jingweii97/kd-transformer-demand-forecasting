#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=phase_wis_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/phase_wis_v2_%j.out
#SBATCH --error=logs/slurm/phase_wis_v2_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
RUN="outputs/student/no_kd/phase_wis_v2"
[[ ! -e "$RUN" ]] || { echo "Refusing output overwrite: $RUN"; exit 1; }
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/train_student.py --env dicc --experiment phase_wis_v2 --exp-name phase_wis_v2 --no-kd --supervised-loss wrmsse_informed
