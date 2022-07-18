#!/bin/bash
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wpharvey@uark.edu        # Where to send mail
#SBATCH -p gpu72
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task 32
#SBATCH --output=High-%j.out
#SBATCH --error=High-%j.out

cd $SLURM_SUBMIT_DIR
module load python/3.9-anaconda
source /share/apps/bin/conda-3.9.sh
conda activate tensorflow-probability


#args format is file.py testname testsetstartstring initialLR l1_reg l2_reg epochs max_clouds max_radians min_half_axis max_half_axis
python OcclusionRunner.py High WaCo2015 2e-4 1e-5 1e-3 500 16 0.174533 40 60