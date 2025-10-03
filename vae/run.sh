#!/bin/bash
#SBATCH --account=fy220020
#SBATCH --partition=batch
#SBATCH --job-name=VAE_AX2
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
nodes=$SLURM_JOB_NUM_NODES
tasks=$SLURM_NTASKS_PER_NODE

module load intel/19.0
module load mkl/19.0
module load openmpi-intel/3.1
module load anaconda3/5.2.0
source activate axdev 

python -u train.py > train.out 
