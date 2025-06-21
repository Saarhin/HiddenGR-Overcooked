#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/error_log/error_%A_%a.log       # Error log file  for each task
#SBATCH --output=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/out_log/output_%A_%a.out 
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --mem=4G                     # Memory per task
#SBATCH --array=0-19                  # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3

module load python/3.10

source marl/bin/activate

export PYTHONPATH="${PYTHONPATH}:/home/scratch/saarhin/HiddenGR-Overcooked"

cd gym_cooking

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ./CC/parameters_mini.txt)

python main.py $PARAMS
