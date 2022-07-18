#!/bin/bash
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wpharvey@uark.edu        # Where to send mail
#SBATCH -p gpu72
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task 32
#SBATCH --output=Cauchy-%j.out
#SBATCH --error=Cauchy-%j.out

cd $SLURM_SUBMIT_DIR
module load python/3.9-anaconda
source /share/apps/bin/conda-3.9.sh
conda activate tensorflow-probability


#args format is file.py testname testsetstartstring initialLR l1_reg l2_reg epochs dist dropout
python DistributionRunner.py Cauchy WaCo2015 1e-3 1e-5 1e-3 500 truncatedcauchy .5