#!/usr/bin/env bash
#SBATCH --job-name=verify_expanse
#SBATCH --partition=gpu-shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_expanse_%j.out
#SBATCH --error=logs/verify_expanse_%j.err



#SBATCH --export=EXCA_CACHE_DIR=/expanse/projects/nemar/dtyoung/.cache/exa,HF_HOME=/expanse/projects/nemar/dtyoung/huggingface_cache,MNE_DATA=/expanse/projects/nemar/dtyoung/mne_data,WANDB_MODE=offline

# ── Environment setup ──
module purge
module load gpu
module load cuda12.2/toolkit/12.2.2

source activate /expanse/projects/nemar/dtyoung/conda_envs/neurolab

export EXCA_CACHE_DIR="/expanse/projects/nemar/dtyoung/.cache/exa"
export HF_HOME="/expanse/projects/nemar/dtyoung/huggingface_cache"
export MNE_DATA="/expanse/projects/nemar/dtyoung/mne_data"
export WANDB_MODE="offline"

# ── Working directory ──
cd /expanse/projects/nemar/adapter_finetuning

# ── Command ──
python scripts/train.py adapter=lora model=labram data=bcic2a
