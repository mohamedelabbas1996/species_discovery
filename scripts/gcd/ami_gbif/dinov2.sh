#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --partition=long           
#SBATCH --cpus-per-task=16  
#SBATCH --gres=gpu:rtx8000:1


module load python/3.10

source $HOME/.ami/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/species_discovery

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH

export PYTHONPATH


python scripts/train_gcd.py \
 --config configs/datasets/clustering/ami_gbif.yml \
      configs/networks/dinov2.yml \
      configs/pipelines/train/train_gcd.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/contrastive_preprocessor.yml \
    --network.vit_name dinov2_vits14_reg \
    --num_gpus 2 \
    --optimizer.num_epochs 200 \
    --optimizer.weight_decay 0.000005 \
    --sup_con_weight 0.35 \
    --wandb.project gcd \
    --wandb.entity moth-ai \
    --wandb.name ami_gbif_dinov2_s \
    --merge_option merge \
    --k 1117 \

