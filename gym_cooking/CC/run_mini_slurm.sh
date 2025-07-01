#!/bin/bash
#SBATCH --job-name=parallel_jobs       # Job name
#SBATCH --error=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/error_log/error_%A_%a.log       # Error log file  for each task
#SBATCH --output=/home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/out_log/output_%A_%a.out 
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-1                  # Array index range (adjust based on parameter file size)
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-mtaylor3

module load python/3.10
module load scipy-stack

# Create and activate a clean virtual environment
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install only required packages
pip install --no-index --upgrade pip
pip install --no-index -r /home/saarhin/scratch/HiddenGR-Overcooked/requirements_cc.txt
pip install --no-index wheel  # Required for some packages

# Base packages
pip install --no-index \
    termcolor \
    tqdm \
    dill \
    "gym==0.17.2" \
    "numpy==1.23"  # pddlgym needs older numpy

# Then install pddlgym with explicit dependencies
pip install --no-index --no-deps pddlgym

# Copy only what's needed
rsync -avh /home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/ $SLURM_TMPDIR/gym_cooking/

cd $SLURM_TMPDIR/gym_cooking

# Run the script (adjust parameters as needed)
python main.py YOUR_PARAMETERS_HERE

# Copy back only results
rsync -avh $SLURM_TMPDIR/gym_cooking/policies_* /home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/results/

