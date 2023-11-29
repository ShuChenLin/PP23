#!/bin/bash
#SBATCH -ptest
#SBATCH -N1
#SBATCH -n12

module load mpi

export MPIP="-y -l"
export LD_PRELOAD=/opt/mpiP/lib/libmpiP.so

srun -N1 -n12 ./hw1 536869888 testcases/40.in testcases/40.out
