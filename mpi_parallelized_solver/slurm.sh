#!/bin/bash
 
#SBATCH --export=NONE
#SBATCH --nodes=5
#SBATCH --partition=express
#SBATCH --cpus-per-task=32
# alternativ: gpuk20
#SBATCH --time 00:10:00
#SBATCH --exclusive

#SBATCH --job-name=MPI-solver
#SBATCH --outpu=/scratch/tmp/e_zhup01/output.txt
#SBATCH --error=/scratch/tmp/e_zhup01/error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=endizhupani@uni-muenster.de

module load GCC/8.2.0-2.31.1
module load icc/2019.1.144-GCC-8.2.0-2.31.1
module load ifort/2019.1.144-GCC-8.2.0-2.31.1
module load OpenMPI/3.1.3
module load GCCcore/8.2.0
module load CMake/3.15.3

cd /home/e/e_zhup01/mpi_parallelized_solver

./build-release.sh
export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verfügbar
# alternativ: Ethernet statt Infiniband: 
export I_MPI_FABRICS=shm:tcp

# dim #runs #gpus 
mpirun /home/e/e_zhup01/mpi_parallelized_solver/build/mpi_parallelized_solver.exe
# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>
