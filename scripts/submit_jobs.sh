#!/bin/bash

sbatch scripts/clustering/ami_trap/resnet50_ms.sh 30
sbatch scripts/clustering/ami_trap/resnet50_ms.sh 40
# sbatch scripts/clustering/ami_trap/resnet50_ms.sh 55
# sbatch scripts/clustering/ami_trap/resnet50_ms.sh 77
# sbatch scripts/clustering/ami_trap/resnet50_ms.sh 88
# sbatch scripts/clustering/ami_trap/resnet50_ms.sh 100

# sbatch scripts/clustering/ami_trap/resnet50_agg.sh 100