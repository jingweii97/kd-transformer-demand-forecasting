#!/bin/bash -l
# Fresh stride-8 WIKD targets from the selected stride-8 Teacher-v2 only.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=phase_stride8_wikd_targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/phase_stride8_wikd_targets_%j.out
#SBATCH --error=logs/slurm/phase_stride8_wikd_targets_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
EXP_NAME="phase_stride8_wikd_v2_teacher_phase_selected"
ORIGINS="configs/origins/phase_stride8_v2.json"
TARGET_DIR="artifacts/soft_targets"
TEACHER_RUN="outputs/teacher/phase_stride8_teacher_v2"
MANIFEST="$TEACHER_RUN/phase_validation_evaluation/selected_checkpoint.json"
[[ -f "$MANIFEST" ]] || { echo "Missing stride-8 Teacher-v2 selection manifest"; exit 1; }
mapfile -t SELECTED < <(python scripts/resolve_selected_student_checkpoint.py --manifest "$MANIFEST" --run-dir "$TEACHER_RUN")
TEACHER="${SELECTED[0]}"; EXPECTED="${SELECTED[1]}"
[[ "$(sha256sum "$TEACHER" | awk '{print $1}')" == "$EXPECTED" ]] || { echo "Selected Teacher-v2 SHA mismatch"; exit 1; }
for store in CA_1 CA_2 CA_3 CA_4 TX_1 TX_2 TX_3 WI_1 WI_2 WI_3; do
  [[ ! -e "$TARGET_DIR/${EXP_NAME}_${store}.pt" && ! -e "$TARGET_DIR/${EXP_NAME}_${store}.json" ]] || { echo "Refusing cache overwrite: $store"; exit 1; }
done
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/generate_soft_targets.py --env dicc --experiment phase_stride8_wikd_v2 --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER" --origins-file "$ORIGINS"
python scripts/verify_soft_target_cache.py --env dicc --experiment phase_stride8_wikd_v2 --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER" --origins-file "$ORIGINS" --samples-per-store 3
