#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/error_log/error_%A_%a.log       # Error log file  for each task
#SBATCH --output=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/out_log/output_%A_%a.out 
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-19                  # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3

module load python/3.10
module load scipy-stack

source ~/envs/marl/bin/activate

# Copy only what's needed
cp -r /home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/ $SLURM_TMPDIR/gym_cooking/


cd $SLURM_TMPDIR/gym_cooking

# Run the script (adjust parameters as needed)
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ./CC/parameters.txt)
python main.py $PARAMS

# Copy back only results
scp -rp  $SLURM_TMPDIR/gym_cooking/policies_*  /home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/results/

