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
#SBATCH --time=50:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/causal_gans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## Transport datasets
#### Resolution 32
## Intensity
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity/trainset /scratch/wei-cheng.lai/mnist_thickness_intensity/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity/valset /scratch/wei-cheng.lai/mnist_thickness_intensity/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity/testset /scratch/wei-cheng.lai/mnist_thickness_intensity/

## Slant
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/trainset /scratch/wei-cheng.lai/mnist_thickness_slant/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/valset /scratch/wei-cheng.lai/mnist_thickness_slant/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/testset /scratch/wei-cheng.lai/mnist_thickness_slant/


echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 32
## StyleGAN-T
## Thickness-Intensity
## train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/thickness_intensity/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data_name mnist-thickness-intensity --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --cond True --wandb_name cond_thick_intensity --wandb_pj_v singlesource_morpho

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/thickness_intensity/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data_name mnist-thickness-intensity --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 6000 --cond True --wandb_name cond_thick_intensity_resume01 --wandb_pj_v singlesource_morpho --resume /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_intensity/stylegan-t/00001-stylegan3-t-condTrue-mnist-thickness-intensity-augnoaug-gpus1-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl


## Thickness-Slant
## Train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/thickness_slant/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thickness_slant/ --data_name mnist-thickness-slant --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --cond True --wandb_name cond_thick_slant --wandb_pj_v singlesource_morpho


## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/morpho/thickness_slant/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/mnist_thickness_slant/ --data_name mnist-thickness-slant --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 6000 --cond True --wandb_name cond_thick_slant_resume01 --wandb_pj_v singlesource_morpho --resume /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_slant/stylegan-t/00001-stylegan3-t-condTrue-mnist-thickness-slant-augnoaug-gpus1-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl
