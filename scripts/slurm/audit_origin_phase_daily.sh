#!/bin/bash -l
# Frozen-checkpoint construct-validity audit; no training or checkpoint writes.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=origin_phase_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/origin_phase_audit_%j.out
#SBATCH --error=logs/slurm/origin_phase_audit_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="outputs/audits/origin_phase_daily_v1.csv"
[[ ! -e "$OUT" ]] || { echo "Refusing audit overwrite: $OUT"; exit 1; }
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/audit_forecast_origin_phase.py --env dicc --experiment full --origins 1813 1814 1815 1816 1817 1818 1819 1820 1821 1822 1908 1909 1910 1911 1912 1913 1914 --output "$OUT" --teacher-checkpoint outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt --wis-checkpoint outputs/student/no_kd/student_wis/student-wis-epoch=04-val_loss=1.476671.ckpt --wikd-checkpoint outputs/student/kd/student_wikd_wi_e09_verified/student-wikd-epoch=09-val_loss=1.452832.ckpt
