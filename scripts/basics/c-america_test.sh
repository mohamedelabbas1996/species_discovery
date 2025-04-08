#!/bin/bash
#SBATCH --job-name=bacis
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --partition=long             
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

# # extended
# python $CURDIR/scripts/eval_trained.py \
#     --config configs/datasets/ami/c-america.yml \
#       configs/datasets/ami/c-america_ood_test.yml \
#       configs/preprocessors/base_preprocessor.yml \
#       configs/networks/resnet50.yml \
#       configs/pipelines/train/baseline.yml \
#       "$postprocessor_config" \
#     --wandb.project osr_posthoc \
#     --wandb.entity moth-ai \
#     --wandb.name extended_onehead \
#     --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/weights/extended/ami/c-america/5978331/checkpoints/model_best.pth  \
#     --network.slurm_id 5978331 \
#     --dataset.train.batch_size 512 \
#     --num_gpus 1 --num_workers 8 \
#     --merge_option merge \
#     --seed 0 

# # novel branch
# python $CURDIR/scripts/eval_trained.py \
#     --config configs/datasets/ami/c-america.yml \
#       configs/datasets/ami/c-america_ood_test.yml \
#       configs/preprocessors/base_preprocessor.yml \
#       configs/networks/resnet50.yml \
#       configs/pipelines/train/baseline.yml \
#       "$postprocessor_config" \
#     --wandb.project osr_posthoc \
#     --wandb.entity moth-ai \
#     --wandb.name novel_branch_onehead \
#     --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/weights/novel_branch/ami/c-america/5978332/checkpoints/model_best.pth  \
#     --network.slurm_id 5749457 \
#     --dataset.train.batch_size 512 \
#     --num_gpus 1 --num_workers 8 \
#     --merge_option merge \
#     --seed 0 


# # baseline
# python $CURDIR/scripts/eval_trained.py \
#     --config configs/datasets/ami/c-america.yml \
#       configs/datasets/ami/c-america_ood_test.yml \
#       configs/preprocessors/base_preprocessor.yml \
#       configs/networks/resnet50.yml \
#       configs/pipelines/train/baseline.yml \
#       "$postprocessor_config" \
#     --wandb.project osr_posthoc \
#     --wandb.entity moth-ai \
#     --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/5749457/checkpoints/model_best.pth  \
#     --network.slurm_id 5749457 \
#     --dataset.train.batch_size 512 \
#     --num_gpus 1 --num_workers 8 \
#     --merge_option merge \
#     --seed 0 

python $CURDIR/scripts/eval_trained.py \
    --config configs/datasets/ami/c-america.yml \
      configs/datasets/ami/c-america_bci.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/networks/resnet50.yml \
      configs/pipelines/train/baseline.yml \
      "$postprocessor_config" \
    --wandb.project osr_posthoc_new \
    --wandb.entity moth-ai \
    --network.checkpoint /network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/5594655/checkpoints/model_best.pth  \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 1 \
    --merge_option merge \
    --seed 0 



  
    # 
    #   configs/datasets/ami/c-america_ood_test.yml \

# configs/datasets/ami/c-america_bci.yml 