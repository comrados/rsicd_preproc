#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/rsicd_preproc/result.log
#SBATCH -J rsicd_preproc
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_short
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "ResNet18"
python3 resnet18.py

echo "BERT"
python3 bert.py