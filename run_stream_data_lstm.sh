#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=256:00:00
#SBATCH --output=Slurm%j.out
#SBATCH --account=def-bengioy
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

for exp_data in covType elec outdoor poker rialto weather covTypetwoclasses sea mixeddrift hyperplane chess outdoor rialtotwoclasses pokertwoclasses interRBF movingRBF border COIL overlap
    do
    for method in lstm
    do
    python stream_sgd_wfa_windowed.py --exp_data $exp_data --method $method --ne 200
    done
done

