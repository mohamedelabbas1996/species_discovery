#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long           
#SBATCH --cpus-per-task=4  
#SBATCH --gres=gpu:rtx8000:1


module load python/3.10

source $HOME/.species/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/species_discovery

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH

export PYTHONPATH

distance_threshold=$1

python scripts/zero_shot_clustering.py \
 --config configs/datasets/  \
      configs/networks/resnet50.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /network/scratch/y/yuyan.chen/species_discovery/weights/ \
    --num_gpus 1 \
    --cluster.name agglomerative \
    --metric distance \
    --split_cost 1 \
    --merge_cost 2 \
    --search_mode.name distance \
    --distance_threshold $distance_threshold \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --wandb.name  \
    --merge_option merge \


