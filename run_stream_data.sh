#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=Slurm%j.out
#SBATCH --account=def-bengioy
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt



for exp_data in covType poker rialto sea elec mixeddrift hyperplane chess weather
    do
    for method in wfa
    do
    python stream_sgd_wfa_windowed.py --exp_data $exp_data --method $method
    done
done

