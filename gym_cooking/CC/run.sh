#!/bin/bash
#SBATCH --account=rrg-mbowling-ad_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=31125M
#SBATCH --time=00:30:00
#SBATCH --mail-user=slakins@ualberta.ca
#SBATCH --mail-type=ALL

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

python -m pip install --no-index -r compute-canada-reqs.txt

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ./scripts/parameters_leaps.txt)

python main_latent_search.py $PARAMS