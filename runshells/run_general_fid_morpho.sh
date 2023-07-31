#!/bin/bash
#SBATCH --job-name=fid_morpho
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/fid_%j.log # %j is job id

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
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/trainset /scratch/wei-cheng.lai/mnist_thick_source2/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/valset /scratch/wei-cheng.lai/mnist_thick_source2/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/testset /scratch/wei-cheng.lai/mnist_thick_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

## StyleGAN-T
## Multi-Source
### linear (gt)
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/multisources/morpho/plots/groundtruth/fids/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --source-gan multi --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/groundtruth/stylegan-t/00004-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl

### Estimated
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/multisources/morpho/plots/estimated_half_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source1/ --data-path2 /scratch/wei-cheng.lai/mnist_thick_source2/ --source-gan multi --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/estimated_half_norm/stylegan-t/00008-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl

## Single-Source
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/source1_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source1/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source1_norm/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg3200/metric-fid50k_full.jsonl

## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/source2_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source2/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source2_norm/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg3200/metric-fid50k_full.jsonl

### Single Source but with two datasets
## source 1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/source1_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source1/ --data-path2 /scratch/wei-cheng.lai/mnist_thick_source2/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source1_norm/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg3200/metric-fid50k_full.jsonl
## source 2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/morpho/plots/source2_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source2/ --data-path2 /scratch/wei-cheng.lai/mnist_thick_source1/ --source-gan single --num-samples 50000 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source2_norm/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg3200/metric-fid50k_full.jsonl
