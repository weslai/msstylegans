#!/bin/bash
#SBATCH --job-name=ms_mri_fid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/fid_%j.log # %j is job id

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
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/multisources/ukb/fids/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data-path2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --num-samples 20000 --source-gan multi --network /dhc/home/wei-cheng.lai/experiments/multisources/ukb/stylegan-t/00004-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch128-gamma4-kimg6000/metric-fid50k_full.jsonl


## Single Source 1
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/ukb/plots/source1/fids/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data-path2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --num-samples 20000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1/stylegan-t/00004-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg3000/metric-fid50k_full.jsonl

## Single Source 2
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/evaluations/eval_stratified_fids.py --outdir ~/experiments/singlesource/ukb/plots/source2/fids/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --data-path2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --num-samples 20000 --source-gan single --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg4000/metric-fid50k_full.jsonl
