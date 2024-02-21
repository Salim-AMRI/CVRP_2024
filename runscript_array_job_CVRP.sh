#!/bin/bash

#SBATCH --array=2,3,4,5,6,7,8,9
#SBATCH --time=6:00:00
#SBATCH -N1
#SBATCH --no-kill
#SBATCH --error=slurm-err-%j.out
#SBATCH --output=slurm-o-%j.out   
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000M
#SBATCH --gres=gpu:1
#SBATCH  -p A40-short


srun python main.py $1  --seed $SLURM_ARRAY_TASK_ID  --type_crossover $2
