#!/bin/bash
#SBATCH --job-name=mbart_ws_kws_fi
#SBATCH --workdir=/home/micheleb/finnishPoetryGeneration
#SBATCH -o run_mbart_ws_kws_fi.out
#SBATCH -M ukko2
#SBATCH -p gpu-short
#SBATCH -c 2
#SBATCH --time=10:00:00
#SBATCH --mem=8G

module purge
srun sh ./run_training.sh "train_gen_model.py training_config/mbart_kws.ini"
