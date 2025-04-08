#!/bin/bash
#SBATCH --job-name=bacis
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --partition=long             
#SBATCH --cpus-per-task=16      
#SBATCH --gres=gpu:1                  

module load python/3.10

source $HOME/mothenv/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/species_discovery

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH

export PYTHONPATH

python scripts/train.py \
 --config configs/datasets/ood/c-america.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/mobilenet.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 120 \
    --optimizer.warmup_epochs 6 \
    --optimizer.lr 0.01 \
    --wandb.name baseline-c-america \
    --run_dir baseline/ami/c-america \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 
