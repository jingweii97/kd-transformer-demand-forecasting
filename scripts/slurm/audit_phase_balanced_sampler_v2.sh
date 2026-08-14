#!/bin/bash -l
# Deterministic pre-launch audit; creates a new v2 audit artifact only.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=phase_v2_sampler_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --qos=short
#SBATCH --output=logs/slurm/phase_v2_sampler_audit_%j.out
#SBATCH --error=logs/slurm/phase_v2_sampler_audit_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="artifacts/audits/phase_v2_sampler_audit_cluster.json"
[[ ! -e "$OUT" ]] || { echo "Refusing audit overwrite: $OUT"; exit 1; }
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/audit_phase_balanced_sampler.py --env dicc --output "$OUT"
