#!/bin/bash
#SBATCH --job-name=fid_all_mnist
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/fid_%j.log # %j is job id

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
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/trainset /scratch/wei-cheng.lai/mnist_thickness_slant/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/valset /scratch/wei-cheng.lai/mnist_thickness_slant/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/testset /scratch/wei-cheng.lai/mnist_thickness_slant/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

## StyleGAN-T
## Multi-Source
### linear (gt)
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/multisources/morpho/plots/groundtruth/fids/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --source-gan multi --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/groundtruth/stylegan-t/00004-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl

### Estimated
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/multisources/morpho/plots/estimated/fids/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --source-gan multi --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/estimated/stylegan-t/00001-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl

## Single-Source
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/thickness_intensity/fids/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_intensity/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl

## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/thickness_slant/fids/ --dataset mnist-thickness-slant --data-path1 /scratch/wei-cheng.lai/mnist_thickness_slant/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_slant/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl

### Single Source but with two datasets
## source 1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/thickness_intensity/fids/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_intensity/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl
## source 2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/thickness_slant/fids/ --dataset mnist-thickness-slant --data-path1 /scratch/wei-cheng.lai/mnist_thickness_slant/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_slant/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl
