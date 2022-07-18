#!/bin/bash
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wpharvey@uark.edu        # Where to send mail
#SBATCH -p gpu72
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task 32
#SBATCH --output=ViTTuner-%j.out
#SBATCH --error=ViTTuner-%j.out

cd $SLURM_SUBMIT_DIR
module load python/3.9-anaconda
source /share/apps/bin/conda-3.9.sh
conda activate tensorflow-probability


#args format is file.py testname testsetstartstring initialLR l1_reg l2_reg epochs
python ViTCCAdamWTuner.py ViTTuner WaCo2015 1e-4 1e-6 1e-4 180000