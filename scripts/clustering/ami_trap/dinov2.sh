#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long           
#SBATCH --cpus-per-task=4  
#SBATCH --gres=gpu:rtx8000:1


module load python/3.10

source $HOME/.ami/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/species_discovery

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH

export PYTHONPATH


python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/ami_trap_eccv.yml  \
        configs/networks/dinov2.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.vit_name dinov2_vitb14_reg \
    --num_gpus 1 \
    --metric pairwise \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --wandb.name ami_c-america_dinov2_vitb14_reg \
    --merge_option merge \

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/ami_trap_eccv.yml  \
        configs/networks/dinov2.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.vit_name dinov2_vits14_reg \
    --num_gpus 1 \
    --metric pairwise \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --wandb.name ami_c-america_dinov2_vits14_reg \
    --merge_option merge \


    # --network.pretrained True \
    # --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/6294829/checkpoints/model_best.pth \

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/ami_trap_eccv.yml  \
#       configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --network.vit_name dinov2_vits14_reg \
#     --search_mode.name binary \
#     --num_gpus 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/ami_trap_eccv.yml  \
#       configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --network.vit_name dinov2_vitb14_reg \
#     --search_mode.name binary \
#     --num_gpus 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/ami_trap_eccv.yml  \
#       configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --network.vit_name dinov2_vits14_reg \
#     --search_mode.name brent \
#     --num_gpus 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/ami_trap_eccv.yml  \
#       configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --network.vit_name dinov2_vitb14_reg \
#     --search_mode.name brent \
#     --num_gpus 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \
