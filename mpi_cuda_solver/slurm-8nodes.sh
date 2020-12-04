#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=gpu2080
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00
#SBATCH --exclusive
#SBATCH --job-name=custom-solver-8nodes
#SBATCH --outpu=/scratch/tmp/e_zhup01/custom-impl-measurements/output_8nodes.txt
#SBATCH --error=/scratch/tmp/e_zhup01/custom-impl-measurements/error_8nodes.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=endizhupani@uni-muenster.de

module load intelcuda/2019a
module load CMake/3.15.3

cd /home/e/e_zhup01/mpi_cuda_solver

./build-release.sh
export OMP_NUM_THREADS=24

# vorl�ufig, bis MPI �ber Infiniband funktioniert
export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verf�gbar
# alternativ: Ethernet statt Infiniband:
export I_MPI_FABRICS=shm:tcp

# mpirun /home/k/kuchen/Muesli4/build/$1 $2 $3
# parameters: array dim #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/da_test 32 2

# parameters: area size (needs to be quadratic) #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/mandelbrotDA 10000 2

# parameters: #processes (= dim of DA), #throws, #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/piDA 1000 1000000 2

# parameters: #DMCols #DMRows #nGPU #nRuns #CpuPercentage

for cpu_p in `seq 0.1 0.1 0.3`; do
    for m_size in 512 1000 5000 10000; do
        for gpu_n in 1 4; do
        mpirun /home/e/e_zhup01/mpi_cuda_solver/build/mpi_cuda_solver.exe $m_size $gpu_n $cpu_p 5 "/scratch/tmp/e_zhup01/custom-impl-measurements/stats_n8_try3.csv"
        done
    done    
done


#srun nvprof --analysis-metrics -o /scratch/tmp/e_zhup01/muesli-jacobi-analysis.%p.nvprof /home/e/e_zhup01/muesli4/build/jacobi -numdevices=1
# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>