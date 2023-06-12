#!/bin/bash
#SBATCH --job-name=ms_mri_estimate
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

PYTHONPATH=$PYTHONPATH:~/projects/causal_gans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

#### Resolution 256
## UKB source1
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/trainset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/valset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/testset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/

#### Resolution 256
## UKB source2
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/trainset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/valset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/testset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 256
## StyleGAN-T
## Multi-Source
## train from the beginning
## COND (estimate)
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train_ms.py --outdir ~/experiments/multisources/ukb/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data_name1 ukb --data2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --data_name2 ukb --use_ground_truth False --gpus 2 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --cond True --wandb_name cond_gt_sep --wandb_pj_v multisource_ukb

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train_ms.py --outdir ~/experiments/multisources/ukb/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data_name1 ukb --data2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --data_name2 ukb --use_ground_truth False --gpus 2 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 6000 --cond True --wandb_name cond_estimate_resume04 --wandb_pj_v multisource_ukb --resume /dhc/home/wei-cheng.lai/experiments/multisources/ukb/stylegan-t/00004-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch128-gamma4-kimg6000/metric-fid50k_full.jsonl
