
#!/bin/bash
#SBATCH --job-name phero_aco_test_kernel
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu2080
#SBATCH --exclusive
#SBATCH --error /home/n/n_herr03/musket-build/out/phero_aco_test_kernel_order.dat
#SBATCH --output /home/n/n_herr03/musket-build/out/phero_aco_test_kernel_order.dat
#SBATCH --mail-type ALL
#SBATCH --mail-user n_herr03@uni-muenster.de
#SBATCH --time 7:30:00
#SBATCH --gres=gpu:1

echo "Runs; Iterations;Problem;ants;BestSequence;RouteDistance;Time;"
for BinSetup in 0 1 2 3 4 5
do
    for Ant in 1024 #2048 4096 8192
    do
        for Run in 1 2 3 4 5 6 7 8 9 10
        do
		bin/BPP_0 10 $BinSetup $Ant
           # srun /home/n/n_herr03/musket-build/build/aco_kernel_phero $RUN 15 $CITY $ANT
        done
    done
done
