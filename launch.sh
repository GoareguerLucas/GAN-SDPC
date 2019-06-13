#!/bin/sh
#SBATCH -J Job_GAN # <-- Change that
#SBATCH -p teslak20 # <-- Could be also teslak40 (less memory)
#SBATCH --gres=gpu:1 # <-- Specify that you want to use 1 gpu for your python prog
#SBATCH -A h146  # <-- Do not change, we are all working under the same project
#SBATCH -t 2-2  # <-- IMPORTANT :  this the duration of the simulation at the format : dd-hh:mm:ss. The simulation stopped even if not finished
#SBATCH -N 1 # <-- The number of node you want to use
#SBATCH -o ./test.out  # <-- the name of the file where the output of the simulation is written
#SBATCH -e ./error.err  # <-- the name of the file where errors of the simulation are written

module purge
module load userspace/all
module load python3/3.6.3

# cd W10_MNISTtreshold_dcgan
cd W11_AE_dcgan
python3 dcgan.py  # <-- Put here the name of the python prog you want to launch
