#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long           
#SBATCH --cpus-per-task=4  
#SBATCH --gres=gpu:rtx8000:1


module load python/3.10

source $HOME/mothenv/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/species_discovery

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH

export PYTHONPATH

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/ami_trap_eccv.yml  \
      configs/networks/extended_net.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.backbone.name resnet50 \
    --network.pretrained True \
   --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/weights/openset/ami/c-america/5832994/checkpoints/model_best.pth \
    --search_mode.name binary \
    --num_gpus 1 \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --merge_option merge \

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/ami_trap_eccv.yml  \
      configs/networks/extended_net.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
     --network.backbone.name resnet50 \
   --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/weights/openset/ami/c-america/5832994/checkpoints/model_best.pth \
    --num_gpus 1 \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --merge_option merge \
