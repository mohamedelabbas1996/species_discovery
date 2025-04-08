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

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/ami_trap_eccv.yml  \
      configs/networks/mobilenet.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /network/scratch/y/yuyan.chen/species_discovery/weights/panama_mobilenet_20240417_161141_30.pth \
    --search_mode.name binary \
    --num_gpus 1 \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --wandb.name ami_trap_mobilenet_AMI-C \
    --merge_option merge \

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/ami_trap_eccv.yml  \
      configs/networks/mobilenet.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /network/scratch/y/yuyan.chen/species_discovery/weights/panama_mobilenet_20240417_161141_30.pth \
    --search_mode.name brent \
    --num_gpus 1 \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --wandb.name ami_trap_mobilenet_AMI-C \
    --merge_option merge \


