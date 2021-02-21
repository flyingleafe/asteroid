#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=1:0:0
#$ -l h_vmem=1G

module load cuda/10.1.243
module load cudnn/7.6-cuda-10.1
module load anaconda3/2020.02

conda activate phd
python "$@"
