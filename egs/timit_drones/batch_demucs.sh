#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=143:00:00
# set name of job
#SBATCH -J demucs

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# load the anaconda module
module load python3/anaconda
# if you need the custom conda environment:
#source activate custom
source activate drone_project
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=a.alex@qmul.ac.uk
# execute the program
python train.py model=demucs
