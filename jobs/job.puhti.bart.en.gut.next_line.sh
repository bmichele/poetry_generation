#!/bin/bash
#SBATCH --job-name=bart.en.gut.next_line
#SBATCH --account=project_2005562
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=64G
#SBATCH -o slurm.bart.en.gut.next_line.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge
module load gcc/9.1.0
module load cuda/11.1.0
srun sh ./run_training_puhti.sh "../train_gen_model.py ../training_config/bart.en.gut.next_line.ini"
