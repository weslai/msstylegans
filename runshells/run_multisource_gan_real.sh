#!/bin/bash
#SBATCH --job-name=retinal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=2
#SBATCH --time=72:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

#### Resolution 256
## UKB source1 (retinal)
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/trainset /scratch/wei-cheng.lai/retinal_source1/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/valset /scratch/wei-cheng.lai/retinal_source1/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/testset /scratch/wei-cheng.lai/retinal_source1/

#### Resolution 256
## UKB source2 (retinal)
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/trainset /scratch/wei-cheng.lai/retinal_source2/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/valset /scratch/wei-cheng.lai/retinal_source2/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/testset /scratch/wei-cheng.lai/retinal_source2/

## Eyepacs
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/kaggle_eyepacs_diabetic/trainset /scratch/wei-cheng.lai/retinal_eyepacs/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/kaggle_eyepacs_diabetic/valset /scratch/wei-cheng.lai/retinal_eyepacs/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/kaggle_eyepacs_diabetic/testset /scratch/wei-cheng.lai/retinal_eyepacs/

## MRI
## UKB source1
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1/trainset /scratch/wei-cheng.lai/mri_source1/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1/valset /scratch/wei-cheng.lai/mri_source1/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1/testset /scratch/wei-cheng.lai/mri_source1/
## UKB source2
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2/trainset /scratch/wei-cheng.lai/mri_source2/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2/valset /scratch/wei-cheng.lai/mri_source2/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2/testset /scratch/wei-cheng.lai/mri_source2/

## ADNI
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/trainset /scratch/wei-cheng.lai/adni_source/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/valset /scratch/wei-cheng.lai/adni_source/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/testset /scratch/wei-cheng.lai/adni_source/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Retinal
## StyleGAN-T
## Multi-Source
## train from the beginning
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_real_ms.py --outdir ~/experiments/multisources/real_ms/retinal_imgs/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/retinal_source1/ --data_name1 retinal --data2 /scratch/wei-cheng.lai/retinal_eyepacs/ --data_name2 eyepacs --gpus 1 --batch 160 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 7000 --cond True --data_scenario high --wandb_name cond_estimate --wandb_pj_v multisource_retinal_eyepacs

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_real_ms.py --outdir ~/experiments/multisources/real_ms/retinal_imgs/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/retinal_source1/ --data_name1 retinal --data2 /scratch/wei-cheng.lai/retinal_eyepacs/ --data_name2 eyepacs --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 4500 --cond True --data_scenario high --wandb_name cond_estimate_re01 --wandb_pj_v multisource_retinal_eyepacs --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/multisources/real_ms/retinal_imgs/stylegan-t/00001-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus1-batch160-gamma4-kimg3000/metric-fid50k_ms_real.jsonl

## Resume exact pickle
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/retinal/retinal_half/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/retinal_source1/ --data_name1 retinal --data2 /scratch/wei-cheng.lai/retinal_source2/ --data_name2 retinal --use_ground_truth False --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario half --wandb_name cond_estimate_new_resume03 --wandb_pj_v multisource_retinal --exact_resume True --resume /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half/stylegan-t/00003-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch128-gamma4-kimg9000/network-snapshot-001351.pkl

## MRI
## cond
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_real_ms.py --outdir ~/experiments/multisources/real_ms/mri_imgs/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/mri_source1/ --data_name1 ukb --data2 /scratch/wei-cheng.lai/adni_source/ --data_name2 adni --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 4000 --cond True --data_scenario high --wandb_name cond_estimate --wandb_pj_v multisource_mri_ukb_adni

## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_real_ms.py --outdir ~/experiments/multisources/real_ms/mri_imgs/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/mri_source1/ --data_name1 ukb --data2 /scratch/wei-cheng.lai/adni_source/ --data_name2 adni --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 4500 --cond True --data_scenario high --wandb_name cond_estimate_re02 --wandb_pj_v multisource_mri_ukb_adni --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/multisources/real_ms/mri_imgs/stylegan-t/00002-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch256-gamma4-kimg4000/metric-fid50k_ms_real.jsonl
