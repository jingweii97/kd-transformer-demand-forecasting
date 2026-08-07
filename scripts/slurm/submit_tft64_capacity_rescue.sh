#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=tft64_cap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/tft64_cap_%j.out
#SBATCH --error=logs/slurm/tft64_cap_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="tft64_capacity_rescue"

if [ ! -e "outputs" ] || [ ! -e "artifacts" ]; then
    echo "ERROR: outputs/ or artifacts/ is unavailable. Run scripts/slurm/setup_hpc_storage.sh once before submitting jobs."
    exit 1
fi

# Never overwrite an experiment; use a new EXP_NAME for a deliberate rerun.
if [ -d "outputs/teacher/$EXP_NAME" ]; then
    echo "Error: Output directory outputs/teacher/$EXP_NAME already exists. Aborting to prevent overwrite."
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)"/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs/slurm

python -c "from utils.config import load_config; c=load_config('dicc', 'tft64_capacity_rescue'); v={'hidden_size':c.teacher.hidden_size,'hidden_continuous_size':c.teacher.hidden_continuous_size,'lstm_layers':c.teacher.lstm_layers,'attention_head_size':c.teacher.attention_heads,'loss':c.teacher.loss,'output_size':1,'gradient_clip_val':c.teacher.gradient_clip_val,'resume_checkpoint':'none'}; e={'hidden_size':64,'hidden_continuous_size':16,'lstm_layers':2,'attention_head_size':4,'loss':'huber','output_size':1,'gradient_clip_val':0.1,'resume_checkpoint':'none'}; assert v == e, (v, e); print('EFFECTIVE CAPACITY-RESCUE CONFIG:'); [print(f'{k} = {v[k]}') for k in e]"

echo "FRESH TFT-64 CAPACITY-RESCUE INITIALIZATION"
echo "RESUME CHECKPOINT: NONE"
echo "INITIAL EPOCH: 0"
echo "INITIAL GLOBAL STEP: 0"

python scripts/train_teacher.py \
    --env dicc \
    --experiment tft64_capacity_rescue \
    --exp-name "$EXP_NAME" \
    "$@"

python scripts/verify_tft_run.py \
    --exp-name "$EXP_NAME" \
    --hidden-size 64 \
    --hidden-continuous-size 16 \
    --lstm-layers 2 \
    --attention-heads 4
