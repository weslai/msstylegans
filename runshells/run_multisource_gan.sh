#!/bin/bash
#SBATCH --job-name=ms_morpho-linear
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=66:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/causal_gans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## Transport datasets
#### Resolution 64
## UkB
#rsync -a --include '*.csv' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/ /scratch/wei-cheng.lai/ukb/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_nonlinear_resolution64/trainset /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_nonlinear_resolution64/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_nonlinear_resolution64/valset /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_nonlinear_resolution64/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_nonlinear_resolution64/testset /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_nonlinear_resolution64/

## Adni
#rsync -a --include '*.csv' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_mni_nonlinear_hippo_resolution64 /scratch/wei-cheng.lai/adni/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_mni_nonlinear_hippo_resolution64/trainset /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_hippo_resolution64/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_mni_nonlinear_hippo_resolution64/valset /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_hippo_resolution64/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_mni_nonlinear_hippo_resolution64/testset /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_hippo_resolution64/

#### Resolution 256
## UKB
rsync -a --include '*.csv' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_linear_freesurfer_resolution256_low /scratch/wei-cheng.lai/ukb/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/trainset /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/valset /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/testset /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/
## Adni
#rsync -a --include '*.csv' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 /scratch/wei-cheng.lai/adni/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/trainset /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_linear_hippo_resolution256/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/valset /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_linear_hippo_resolution256/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/testset /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_linear_hippo_resolution256/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 64
## StyleGAN-T
## UKB
## train from the beginning
## COND 
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/cmssg/mri/single_cond/ukb/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_nonlinear_resolution64/ --data_name ukb --gpus 1 --batch 128 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --hybrid 0 --cond True --if_causal False --wandb_name cond_mri_ukb_t --wandb_pj_v mri_generation 
## ADNI
## Start from scratch
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/cmssg/mri/single_cond/adni/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_hippo_resolution64/ --data_name adni --gpus 1 --batch 128 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --kimg 3500 --aug noaug --hybrid 0 --cond True --if_causal False --wandb_name cond_adni_t --wandb_pj_v mri_generation

## StyleGAN-R
## UKB
## ADNI


### Resolution 256
## StyleGAN-T
## UKB
## Train from the beginning
# Cond
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/syreal/mni_lin/domains/ukb_freesurfer/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_linear_freesurfer_resolution256/ --data_name ukb --gpus 2 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 14000 --cond True --without_volumes False --wandb_name ukb_cond-t_new --wandb_pj_v mni_lin256_generation
## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/syreal/mni_lin/domains/ukb_freesurfer_low/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_linear_freesurfer_resolution256_low/ --data_name ukb --gpus 2 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 12000 --cond True --without_volumes False --wandb_name ukb_cond-t_low_new_resume00 --wandb_pj_v mni_lin256_generation --resume /dhc/home/wei-cheng.lai/experiments/syreal/mni_lin/single_cond/ukb_freesurfer_low/stylegan-t/00001-stylegan3-t-condTrue-ukb-augnoaug-gpus2-batch128-gamma4-kimg12000/metric-fid50k_full.jsonl

## StyleGAN-R
## Train from the beginning
## UKB
## Cond
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/cmssg/mri256/single_cond/ukb/stylegan-r --cfg stylegan3-r --data /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_nonlinear/ --data_name ukb --gpus 1 --batch 64 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 3000 --hybrid 0 --cond True --if_causal False --wandb_name ukb_cond-r --wandb_pj_v mri256_generation
## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/cmssg/mri256/single_cond/ukb/stylegan-r --cfg stylegan3-r --data /scratch/wei-cheng.lai/ukb/T1_3T_coronal_mni_nonlinear/ --data_name ukb --gpus 1 --batch 64 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 2500 --hybrid 0 --cond True --if_causal False --wandb_name ukb_cond-r_resume00 --wandb_pj_v mri256_generation --resume /dhc/home/wei-cheng.lai/experiments/cmssg/mri256/single_cond/ukb/stylegan-r/00000-stylegan3-r-condTrue-causalFalse-ukb-augnoaug-gpus1-batch64-gamma4-kimg3000/metric-fid50k_full.jsonl

## StyleGAN-T
## ADNI
## Train from the beginning
# Cond
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/syreal/mni_lin/single_cond/adni/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_linear_hippo_resolution256/ --data_name adni --gpus 2 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 12000 --cond True --without_volumes False --wandb_name adni_cond-t_new --wandb_pj_v mni_lin256_generation
## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/syreal/single_cond/adni/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_linear_hippo_resolution256/ --data_name adni --gpus 1 --batch 64 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug ada --kimg 5000 --hybrid 0 --cond True --without_volumes False --if_causal False --wandb_name adni_cond-t_resume06 --wandb_pj_v mni_lin256_generation --resume /dhc/home/wei-cheng.lai/experiments/syreal/mni_lin/single_cond/adni/stylegan-t/00006-stylegan3-t-condTrue-causalFalse-adni-augada-gpus1-batch64-gamma4-kimg5000/metric-fid50k_full.jsonl

## StyleGAN-R
## ADNI
## Train from the beginning
# Cond
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/cmssg/mri256/single_cond/adni/stylegan-r --cfg stylegan3-r --data /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_hippo_resolution256/ --data_name adni --gpus 1 --batch 64 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 3000 --hybrid 0 --cond True --if_causal False --wandb_name adni_cond-r --wandb_pj_v mri256_generation
## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/causal-gan/stylegan3/train.py --outdir ~/experiments/cmssg/mri256/single_cond/adni/stylegan-r --cfg stylegan3-r --data /scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_hippo_resolution256/ --data_name adni --gpus 1 --batch 64 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 2500 --hybrid 0 --cond True --if_causal False --wandb_name adni_cond-r_resume00 --wandb_pj_v mri256_generation --resume /dhc/home/wei-cheng.lai/experiments/cmssg/mri256/single_cond/adni/stylegan-r/00000-stylegan3-r-condTrue-causalFalse-adni-augnoaug-gpus1-batch64-gamma4-kimg3000/metric-fid50k_full.jsonl

