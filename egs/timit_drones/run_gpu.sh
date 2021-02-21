#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 16
#$ -l gpu=2
#$ -cwd
#$ -j y

module load cuda/10.1.243
module load cudnn/7.6-cuda-10.1
module load anaconda3/2020.02

conda activate phd
python "$@"
