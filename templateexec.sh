#!/bin/bash
#SBATCH --job-name yourjobname
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu2080
#SBATCH --exclusive
#SBATCH --error /path/to/error/file
#SBATCH --output /path/to/dat/file
#SBATCH --mail-type ALL
#SBATCH --mail-useryouremail
#SBATCH --time 7:30:00
#SBATCH --gres=gpu:1

echo "Runs;Time;"
for Run in 1 2 3 4 5 6 7 8 9 10
do
	srun /path/to/your/exec [additional arguments]
done

