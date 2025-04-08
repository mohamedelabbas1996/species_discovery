#!/bin/bash
#SBATCH --job-name=bacis
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --partition=long             
#SBATCH --cpus-per-task=16      
#SBATCH --gres=gpu:1                  

module load python/3.10

source $HOME/mothenv/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/BioOSR_training

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH
PYTHONPATH=$CURDIR/src/OpenOOD:$PYTHONPATH

export PYTHONPATH

python $CURDIR/scripts/main.py \
 --config $CURDIR/configs/datasets/ami/ne-america.yml \
    $CURDIR/configs/preprocessors/base_preprocessor.yml \
    $CURDIR/configs/networks/resnet50.yml \
    $CURDIR/configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 120 \
    --optimizer.warmup_epochs 6 \
    --optimizer.lr 0.01 \
    --wandb.name baseline-ne-america \
    --run_dir baseline/ami/ne-america \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 
