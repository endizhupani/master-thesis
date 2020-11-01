#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=gpu2080
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --job-name=MPI-solver
#SBATCH --outpu=/scratch/tmp/e_zhup01/output.txt
#SBATCH --error=/scratch/tmp/e_zhup01/error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=endizhupani@uni-muenster.de

module load intelcuda/2019a
module load CMake/3.15.3

cd /home/e/e_zhup01/mpi_cuda_solver

./build-release.sh
export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verfügbar
# alternativ: Ethernet statt Infiniband:
export I_MPI_FABRICS=shm:tcp

mpirun /home/e/e_zhup01/mpi_cuda_solver/build/mpi_cuda_solver.exe 1200 0.5