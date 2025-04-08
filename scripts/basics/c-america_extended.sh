#!/bin/bash
#SBATCH --job-name=bacis
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --partition=long             
#SBATCH --cpus-per-task=16      
#SBATCH --gres=gpu:1                  

module load python/3.10

source $HOME/mothenv/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/BioOSR

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH
PYTHONPATH=$CURDIR/src/OpenOOD:$PYTHONPATH

export PYTHONPATH

python $CURDIR/scripts/main.py \
 --config configs/datasets/ami/c-america_extended_20.yml  \
   $CURDIR/configs/networks/resnet50.yml \
    $CURDIR/configs/pipelines/train/baseline.yml \
    $CURDIR/configs/preprocessors/base_preprocessor.yml \
    --wandb.name extended_onehead_c-america \
    --network.name resnet50 \
    --network.pretrained True \
    --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/pretrained_models/model_5749457.pth \
    --optimizer.num_epochs 30 \
    --optimizer.warmup_epochs 2 \
    --optimizer.lr 0.001 \
    --run_dir ood_benchmark/weights/extended/ami/c-america \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 

