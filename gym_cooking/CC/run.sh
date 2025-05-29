#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/error_log/error_%A_%a.log       # Error log file  for each task
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --mem=4G                     # Memory per task
#SBATCH --array=0-19                  # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3

module load python/3.10

source marl/bin/activate

export PYTHONPATH="${PYTHONPATH}:/scratch/saarhin/HiddenGR-Overcooked"


PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ./gym_cooking/CC/parameters.txt)

python ./gym_cooking/main.py $PARAMS
