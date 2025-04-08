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

python scripts/zero_shot_clustering.py \
 --config configs/datasets/clustering/test.yml  \
      configs/networks/resnet50.yml \
      configs/pipelines/test/test_clustering.yml \
      configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /network/scratch/y/yuyan.chen/species_discovery/weights/panama_resnet50_baseline_20240417_edbb46dd.pth \
    --search_mode.name brent \
    --num_gpus 1 \
    --wandb.project zero_shot_clustering \
    --wandb.entity moth-ai \
    --wandb.name ami_trap_resnet50_AMI-C \
    --merge_option merge \


# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/test.yml  \
#         configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --network.vit_name dinov2_vits14_reg \
#     --cluster.name agglomerative \
#     --num_gpus 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --wandb.name ami_c-america_dinov2_vitb14_reg \
#     --merge_option merge \



# python scripts/train_gcd.py \
#  --config configs/datasets/gcd/test.yml \
#       configs/networks/dinov2.yml \
#       configs/pipelines/train/train_gcd.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/contrastive_preprocessor.yml \
#     --num_gpus 1 \
#     --optimizer.num_epochs 5 \
#     --wandb.project gcd \
#     --wandb.entity moth-ai \
#     --merge_option merge \
#     --k 637 \


# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/test.yml   \
#       configs/networks/extended_net.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --dataset.max_classes 638 \
#     --network.backbone.name resnet50 \
#     --network.pretrained True \
#    --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/weights/openset/ami/c-america/5832994/checkpoints/model_best.pth \
#     --search_mode.name binary \
#     --num_gpus 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/test.yml \
#       configs/networks/dinov2.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --dataset.max_classes 638 \
#     --network.vit_name dinov2_vitb14_reg \
#     --search_mode.name binary \
#     --num_gpus 1 --num_workers 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \

# python scripts/zero_shot_clustering.py \
#  --config configs/datasets/clustering/test.yml \
#       configs/networks/mobilenet.yml \
#       configs/pipelines/test/test_clustering.yml \
#       configs/preprocessors/base_preprocessor.yml \
#     --network.pretrained True \
#     --network.checkpoint /network/scratch/y/yuyan.chen/species_discovery/weights/panama_mobilenet_20240417_161141_30.pth \
#     --search_mode.name binary \
#     --num_gpus 1 --num_workers 1 \
#     --wandb.project zero_shot_clustering \
#     --wandb.entity moth-ai \
#     --merge_option merge \
