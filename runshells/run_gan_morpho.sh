#!/bin/bash
#SBATCH --job-name=morpho-source1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## Transport datasets
#### Resolution 32
## thickness intensity slant
## source1
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/trainset /scratch/wei-cheng.lai/mnist_thick_source1/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/valset /scratch/wei-cheng.lai/mnist_thick_source1/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/testset /scratch/wei-cheng.lai/mnist_thick_source1/

## source2
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/trainset /scratch/wei-cheng.lai/mnist_thick_source2/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/valset /scratch/wei-cheng.lai/mnist_thick_source2/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/testset /scratch/wei-cheng.lai/mnist_thick_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 32
## StyleGAN-T
## Thickness-Intensity
## train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/source1_norm/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thick_source1/ --data_name mnist-thickness-intensity-slant --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario high --wandb_name cond_source1_norm --wandb_pj_v singlesource_morpho

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/source1_norm/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thick_source1/ --data_name mnist-thickness-intensity-slant --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 3200 --cond True --data_scenario high --wandb_name cond_source1_norm_re01 --wandb_pj_v singlesource_morpho --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source1_norm/stylegan-t/00001-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg6400/metric-fid50k_full.jsonl

## Thickness-Slant
## Train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/source2_norm/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thick_source2/ --data_name mnist-thickness-intensity-slant --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario high --wandb_name cond_source2_norm --wandb_pj_v singlesource_morpho

## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/source2_norm/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thick_source2/ --data_name mnist-thickness-intensity-slant --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 3200 --cond True --data_scenario high --wandb_name cond_source2_norm_re01 --wandb_pj_v singlesource_morpho --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source2_norm/stylegan-t/00001-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg6400/metric-fid50k_full.jsonl
