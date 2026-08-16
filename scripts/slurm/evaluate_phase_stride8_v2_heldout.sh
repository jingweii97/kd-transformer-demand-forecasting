#!/bin/bash -l
# Original held-out windows, evaluated only after all stride-8 selections.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_phase_stride8_v2_heldout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_phase_stride8_v2_heldout_%j.out
#SBATCH --error=logs/slurm/eval_phase_stride8_v2_heldout_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="outputs/evaluation/phase_stride8_v2_validation_selected_heldout"
[[ ! -e "$OUT" ]] || { echo "Refusing output overwrite: $OUT"; exit 1; }
teacher_run="outputs/teacher/phase_stride8_teacher_v2"
wis_run="outputs/student/no_kd/phase_stride8_wis_v2"
wikd_run="outputs/student/kd/phase_stride8_wikd_v2"
for run in "$teacher_run" "$wis_run" "$wikd_run"; do
  [[ -f "$run/phase_validation_evaluation/selected_checkpoint.json" ]] || { echo "Missing phase-selection manifest: $run"; exit 1; }
done
mapfile -t teacher_selected < <(python scripts/resolve_selected_student_checkpoint.py --manifest "$teacher_run/phase_validation_evaluation/selected_checkpoint.json" --run-dir "$teacher_run")
mapfile -t wis_selected < <(python scripts/resolve_selected_student_checkpoint.py --manifest "$wis_run/phase_validation_evaluation/selected_checkpoint.json" --run-dir "$wis_run")
mapfile -t wikd_selected < <(python scripts/resolve_selected_student_checkpoint.py --manifest "$wikd_run/phase_validation_evaluation/selected_checkpoint.json" --run-dir "$wikd_run")
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env
python scripts/evaluate_models.py --env dicc --experiment phase_stride8_wikd_v2 --exp-name phase_stride8_v2_validation_selected_heldout --teacher-checkpoint "${teacher_selected[0]}" --student-nokd-checkpoint "${wis_selected[0]}" --student-kd-checkpoint "${wikd_selected[0]}" --selected-student-label "Student-WIKD stride-8 v2"
