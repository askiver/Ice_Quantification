#!/bin/bash
#SBATCH --job-name="Training ice quantification models"
#SBATCH --account=share-ie-idi
#SBATCH --time=02-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=128G

#SBATCH --nodes=1               # Allocate 1 nodes for the job
#SBATCH --gres=gpu:1            # Allocate 1 GPU for the job (per node)
#SBATCH --ntasks-per-node=1     # Allocate 1 task per node
#SBATCH --cpus-per-task=8       # Allocate 8 CPU cores per task
#SBATCH --nodelist=idun-01-[01-06],idun-06-[01-07],idun-07-[08-10],idun-08-01 # Specify the nodes to be used


#SBATCH --mail-user=sylte.a.iver@ntnu.no
#SBATCH --mail-type=None
#SBATCH --output=testing.txt

module purge

module load Python/3.12.3-GCCcore-13.3.0

source ~/project/pyenv/bin/activate

srun nvidia-smi

python main.py
