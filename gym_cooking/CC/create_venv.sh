salloc --time=0:45:0 --ntasks=1 --mem-per-cpu="4G" --account=def-mtaylor3
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate
cp /home/$USER/scratch/HiddenGR-Overcooked/requirements.txt $SLURM_TMPDIR
pip install -r requirements.txt
tar -cavf venv.tar.xz .venv
cp venv.tar.xz /home/$USER/scratch/HiddenGR-Overcooked/