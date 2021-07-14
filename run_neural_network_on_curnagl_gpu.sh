#!/bin/bash -l

#SBATCH --mail-type ALL
#SBATCH --mail-user philippe.jacquet@unil.ch

#SBATCH --chdir /scratch/pjacquet/
#SBATCH --job-name NN
#SBATCH --output NN.out

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 01:00:00

module load gcc/9.3.0 cuda/11.2.2 cudnn/8.1.1.33-11.2 python/3.8.8

STARTTIME=$(date +%s)

source $HOME/venv_tensorflow_gpu/bin/activate

python $HOME/C-Square/neural_network_gpu.py

ENDTIME=$(date +%s)

echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete this task"
