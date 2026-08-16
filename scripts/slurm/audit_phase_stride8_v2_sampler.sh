#!/bin/bash -l
# Final stride-8 pre-launch audit. Does not write into prior v2 namespaces.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=phase_stride8_v2_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/phase_stride8_v2_audit_%j.out
#SBATCH --error=logs/slurm/phase_stride8_v2_audit_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="artifacts/audits/phase_stride8_v2_sampler_audit_cluster.json"
[[ ! -e "$OUT" ]] || { echo "Refusing audit overwrite: $OUT"; exit 1; }
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/audit_phase_stride8_v2_sampler.py --env dicc --output "$OUT"
