#!/bin/bash
#SBATCH --job-name=openood
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=4G
#SBATCH --partition=unkillable           
#SBATCH --cpus-per-task=1      

module load python/3.10

source $HOME/.ami/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/species_discovery

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH

export PYTHONPATH

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/test.yml \
#       configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --dataset.max_classes 638 \
#     --search_mode.name binary \
#     --num_gpus 1 --num_workers 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/test.yml \
      configs/networks/resnet50.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/bioclip_preprocessor.yml \
    --dataset.max_classes 638 \
    --search_mode.name binary \
    --num_gpus 1 --num_workers 1 \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --merge_option merge \
