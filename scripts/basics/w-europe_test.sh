#!/bin/bash
#SBATCH --job-name=bacis
#SBATCH --ntasks=1
#SBATCH --mem=48
#SBATCH --partition=main             
#SBATCH --cpus-per-task=8    
#SBATCH --gres=gpu:1                  

module load python/3.10

source $HOME/mothenv/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/BioOSR

PYTHONPATH=$CURDIR:$PYTHONPATH
PYTHONPATH=$CURDIR/src:$PYTHONPATH
PYTHONPATH=$CURDIR/src/OpenOOD:$PYTHONPATH

export PYTHONPATH

postprocessor_config=$1

python $CURDIR/scripts/eval_trained.py \
    --config configs/datasets/ami/w-europe.yml \
      configs/datasets/ami/w-europe_ood_test.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/networks/resnet50.yml \
      configs/pipelines/train/baseline.yml \
       "$postprocessor_config" \
    --wandb.project osr_posthoc_new \
    --wandb.entity moth-ai \
    --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/5749458/checkpoints/model_best.pth \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 8 \
    --merge_option merge \
    --seed 0 
