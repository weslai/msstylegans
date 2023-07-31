#!/bin/bash
#SBATCH --job-name=fid_all_mri
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/fid_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

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
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/valset /scratch/wei-cheng.lai/retinal_source2/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/testset /scratch/wei-cheng.lai/retinal_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 256
## StyleGAN-T
## MRI
## Multi-Source
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/multisources/ukb/fids/ms_log --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source1/ --data-path2 /scratch/wei-cheng.lai/mri_source2/ --num-samples 50000 --source-gan multi --network /dhc/home/wei-cheng.lai/experiments/multisources/ukb/ukb_half_log/stylegan-t/00004-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl
## Single-Source
## Source1
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/ukb/fids/source1_greymatter_log --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source1/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1_greymatter_log/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg3000/metric-fid50k_full.jsonl

## Source2
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/ukb/fids/source2_ventricle_log --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source2/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2_ventricle_log/stylegan-t/00004-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg5800/metric-fid50k_full.jsonl

### Single Source but with two datasets
## source 1
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/ukb/fids/source1_greymatter_log --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source1/ --data-path2 /scratch/wei-cheng.lai/mri_source2/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1_greymatter_log/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg3000/metric-fid50k_full.jsonl

## source 2
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/ukb/fids/source2_ventricle_log --dataset ukb --data-path1 /scratch/wei-cheng.lai/mri_source2/ --data-path2 /scratch/wei-cheng.lai/mri_source1/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2_ventricle_log/stylegan-t/00004-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg5800/metric-fid50k_full.jsonl

## Retinal
# Multi-sources
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/multisources/retinal/fids/ms_log --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --num-samples 50000 --source-gan multi --network /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half_log/stylegan-t/00005-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl
# Single-Source
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/retinal/fids/source1_diastolic_log/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source1_diastolic_log/stylegan-t/00005-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl
## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/retinal/fids/source2_spherical_log/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source2/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source2_spherical_log/stylegan-t/00006-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg3400/metric-fid50k_full.jsonl
## single source but with two datasets
## source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/retinal/fids/source1_diastolic_log/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source1_diastolic_log/stylegan-t/00005-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl
## source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/evaluations/eval_general_fids.py --outdir ~/experiments/singlesource/retinal/fids/source2_spherical_log/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source2/ --data-path2 /scratch/wei-cheng.lai/retinal_source1/ --num-samples 50000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source2_spherical_log/stylegan-t/00006-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg3400/metric-fid50k_full.jsonl
