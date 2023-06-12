#!/bin/bash
#SBATCH --job-name=ukb_source2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --time=55:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/causal_gans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
#export MASTER_PORT=13120
#export WORLD_SIZE=4

### get the first node name as master address
#echo "NODELIST="${SLURM_NODELIST}
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR


#### Resolution 256
## UKB source1
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/trainset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/valset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/testset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/

#### Resolution 256
## UKB source2
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/trainset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/valset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/
rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/testset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 256
## StyleGAN-T
## Source 1 (Age, Brain)
## train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/ukb/source1/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data_name ukb --gpus 1 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 10000 --cond True --wandb_name cond_source1_brain --wandb_pj_v singlesource_ukb

## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/ukb/source1/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data_name ukb --gpus 1 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 3000 --cond True --wandb_name cond_source1_brain_resume03 --wandb_pj_v singlesource_ukb --resume /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1/stylegan-t/00003-stylegan3-t-condTrue-ukb-augnoaug-gpus2-batch128-gamma4-kimg9000/metric-fid50k_full.jsonl

### Source 2 (Age, Ventricle)
## Train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/ukb/source2/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --data_name ukb --gpus 1 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --cond True --wandb_name cond_source2_ventri --wandb_pj_v singlesource_ukb

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train.py --outdir ~/experiments/singlesource/ukb/source2/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --data_name ukb --gpus 1 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 5000 --cond True --wandb_name cond_source2_ventri_resume05 --wandb_pj_v singlesource_ukb --resume /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg4000/metric-fid50k_full.jsonl

