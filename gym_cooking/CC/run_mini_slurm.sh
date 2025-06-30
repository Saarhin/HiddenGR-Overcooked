#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/error_log/error_%A_%a.log       # Error log file  for each task
#SBATCH --output=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/out_log/output_%A_%a.out 
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-1                  # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3

SRC=/home/saarhin/scratch/HiddenGR-Overcooked

DEST=$SLURM_TMPDIR/

rsync -avh "$SRC/" "$DEST/"

cd $SLURM_TMPDIR/HiddenGR-Overcooked


module load python/3.10

source $SLURM_TMPDIR/HiddenGR-Overcooked/marl/bin/activate

export PYTHONPATH="${PYTHONPATH}:$SLURM_TMPDIR/HiddenGR-Overcooked"

cd gym_cooking

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ./CC/parameters_mini.txt)

python main.py $PARAMS

SRC_BASE=$SLURM_TMPDIR/HiddenGR-Overcooked/gym_cooking

DEST=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/results

rsync -avh "${SRC_BASE}"/policies_*  "$DEST"/

