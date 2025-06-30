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

module purge
module load python/3.10
module load scipy-stack/2025a  # Explicit version

# 2. Create FRESH virtualenv WITHOUT system packages
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# 3. Install base packages FIRST
pip install --no-index --upgrade pip
pip install --no-index wheel setuptools

# 4. Install numpy FIRST with exact version
pip install --no-index "numpy==1.23.5"  # Middle ground version

# 5. Install other critical packages
pip install --no-index \
    termcolor \
    tqdm \
    dill \
    "gym==0.17.2" \
    "matplotlib<3.8"  # Version compatible with numpy 1.23

# 6. Install pddlgym from pre-downloaded wheel
# FIRST on login node: pip download pddlgym -d ~/wheels
pip install --no-index --find-links=~/wheels pddlgym

# 7. Copy and run your code
rsync -avh /home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/ $SLURM_TMPDIR/gym_cooking/
cd $SLURM_TMPDIR/gym_cooking
python main.py YOUR_PARAMETERS

# 8. Copy results
rsync -avh policies_* /home/saarhin/scratch/HiddenGR-Overcooked/gym_cooking/results/
