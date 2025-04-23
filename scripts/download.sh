#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=unkillable-cpu                # Ask for unkillable job
#SBATCH --cpus-per-task=2                   # Ask for 2 CPUs
#SBATCH --mem=4G                            # Ask for 4 GB of RAM


module load python/3.10

source ~/.species/bin/activate

python scripts/download_inat_data.py