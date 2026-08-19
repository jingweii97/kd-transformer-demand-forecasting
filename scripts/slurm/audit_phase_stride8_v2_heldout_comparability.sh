#!/bin/bash -l
# Read-only post-evaluation integrity audit.  It never loads checkpoints or
# generates forecasts; it creates only a separate provenance sidecar on pass.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=audit_phase_s8_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/audit_phase_stride8_v2_heldout_comparability_%j.out
#SBATCH --error=logs/slurm/audit_phase_stride8_v2_heldout_comparability_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

OUT="outputs/evaluation/phase_stride8_v2_validation_selected_heldout/comparability_provenance_audit.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite provenance sidecar: $OUT"; exit 1; }

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

python scripts/audit_phase_stride8_heldout_comparability.py --env dicc --output "$OUT"
