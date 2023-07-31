#!/bin/bash
#SBATCH --job-name=strata_mri
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=50gb
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/fid_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

#### Transport Datasets
#### Resolution 32
## thickness intensity slant
## source1
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/trainset /scratch/wei-cheng.lai/mnist_thick_source1/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/valset /scratch/wei-cheng.lai/mnist_thick_source1/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/testset /scratch/wei-cheng.lai/mnist_thick_source1/

## source2
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/trainset /scratch/wei-cheng.lai/mnist_thick_source2/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/valset /scratch/wei-cheng.lai/mnist_thick_source2/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/testset /scratch/wei-cheng.lai/mnist_thick_source2/

#### Resolution 256
## UKB source1
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1/trainset /scratch/wei-cheng.lai/mri_source1/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1/valset /scratch/wei-cheng.lai/mri_source1/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1/testset /scratch/wei-cheng.lai/mri_source1/

## UKB source2
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2/trainset /scratch/wei-cheng.lai/mri_source2/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2/valset /scratch/wei-cheng.lai/mri_source2/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2/testset /scratch/wei-cheng.lai/mri_source2/

## Retinal
## source1
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/trainset /scratch/wei-cheng.lai/retinal_source1/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/valset /scratch/wei-cheng.lai/retinal_source1/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/testset /scratch/wei-cheng.lai/retinal_source1/

## source2
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/trainset /scratch/wei-cheng.lai/retinal_source2/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/valset /scratch/wei-cheng.lai/retinal_source1/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/testset /scratch/wei-cheng.lai/retinal_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 32
## Morpho-MNIST
## Multi-Source
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/multisources/morpho/plots/estimated_half_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source1/ --data-path2 /scratch/wei-cheng.lai/mnist_thick_source2/ --num-samples 50000 --source-gan multi --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/estimated_half_norm/stylegan-t/00008-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl
## Single Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/morpho/plots/source1_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source1/ --data-path2 /scratch/wei-cheng.lai/mnist_thick_source2/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source1_norm/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg3200/metric-fid50k_full.jsonl
## Single Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/morpho/plots/source2_norm/fids/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_thick_source2/ --data-path2 /scratch/wei-cheng.lai/mnist_thick_source1/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source2_norm/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg3200/metric-fid50k_full.jsonl

### MRT
### Resolution 256
## StyleGAN-T
## Multi-Source
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/multisources/ukb/fids/ms_log --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source1/ --data-path2 /scratch/wei-cheng.lai/mri_source2/ --num-samples 40000 --source-gan multi --network /dhc/home/wei-cheng.lai/experiments/multisources/ukb/ukb_half_log/stylegan-t/00004-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl

## Single Source 1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/ukb/fids/source1_greymatter_log/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source1/ --data-path2 /scratch/wei-cheng.lai/mri_source2/ --num-samples 40000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1_greymatter_log/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg3000/metric-fid50k_full.jsonl

## Single Source 2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/ukb/fids/source2_ventricle_log/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source2/ --data-path2 /scratch/wei-cheng.lai/mri_source1/ --num-samples 40000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2_ventricle_log/stylegan-t/00004-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg5800/metric-fid50k_full.jsonl

## Retinal 
## Multi-Source
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/multisources/retinal/fids/ms_log --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --num-samples 40000 --source-gan multi --network /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half_log/stylegan-t/00005-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl
## Single Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/retinal/fids/source1_diastolic_log/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --num-samples 40000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source1_diastolic_log/stylegan-t/00005-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl

## Single Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/retinal/fids/source2_spherical_log/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source2/ --data-path2 /scratch/wei-cheng.lai/retinal_source1/ --num-samples 40000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source2_spherical_log/stylegan-t/00006-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg3400/metric-fid50k_full.jsonl
