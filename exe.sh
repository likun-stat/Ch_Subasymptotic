#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --job-name=c1        # name for the job
#SBATCH -N 1                       # number of nodes
#SBATCH	--tasks-per-node=50
#SBATCH --mem-per-cpu=5G                       # total memory
#SBATCH --time 2-00:00                 # time limit in the form days-hours:minutes
#SBATCH --mail-user=lzxvc@umsystem.edu    # email address for notifications
#SBATCH --mail-type=FAIL,END           # email types            
#SBATCH --account=lzxvc-lab
#--------------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled

echo "### Starting at: $(date) ###"

## Module Commands
# use 'module avail python/' to find the latest version
module purge
module load gcc
module load openmpi
#module load gsl
module load netlib-lapack-blas
module load r/4.4.0
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate myenv
export LD_LIBRARY_PATH=/cluster/software/common/R/4.4.0/lib64/R/lib/:$LD_LIBRARY_PATH

## Run the python script
SCRIPT='sampler.py'
python -c "import numpy; print(numpy.__version__)" 
mpirun -n 50 python -u ${SCRIPT}

conda deactivate
echo "### Ending at: $(date) ###"
