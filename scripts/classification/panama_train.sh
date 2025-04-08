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

python scripts/train_classifier.py \
 --config configs/datasets/classification/panama.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/baseline.yml \
   --network.pretrained True \
   --dataset.num_classes 2360 \
    --network.checkpoint /network/scratch/y/yuyan.chen/species_discovery/weights/global_resnet50_20240828_b06d3b3a.pth \
    --optimizer.num_epochs 30 \
    --optimizer.warmup_epochs 4 \
    --optimizer.lr 0.001 \
    --wandb.project panama_classifier \
    --wandb.name resnet50_global_moth_pretrained \
    --run_dir species_discovery/checkpoints/classification \
    --dataset.train.batch_size 128 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 
