#!/bin/bash
#SBATCH --job-name=ms_retinal
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
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/trainset /scratch/wei-cheng.lai/retinal_source2/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/valset /scratch/wei-cheng.lai/retinal_source2/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/testset /scratch/wei-cheng.lai/retinal_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 256
## StyleGAN-T
## Multi-Source
## train from the beginning
## COND (estimate)
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/retinal/retinal_half_log/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/retinal_source1/ --data_name1 retinal --data2 /scratch/wei-cheng.lai/retinal_source2/ --data_name2 retinal --use_ground_truth False --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario half --wandb_name cond_estimate_log --wandb_pj_v multisource_retinal

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/retinal/retinal_half_log/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/retinal_source1/ --data_name1 retinal --data2 /scratch/wei-cheng.lai/retinal_source2/ --data_name2 retinal --use_ground_truth False --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 5000 --cond True --data_scenario half --wandb_name cond_estimate_log_re04 --wandb_pj_v multisource_retinal --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half_log/stylegan-t/00004-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch256-gamma4-kimg5000/metric-fid50k_full.jsonl

## Resume exact pickle
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/retinal/retinal_half/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/retinal_source1/ --data_name1 retinal --data2 /scratch/wei-cheng.lai/retinal_source2/ --data_name2 retinal --use_ground_truth False --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario half --wandb_name cond_estimate_new_resume03 --wandb_pj_v multisource_retinal --exact_resume True --resume /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half/stylegan-t/00003-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch128-gamma4-kimg9000/network-snapshot-001351.pkl
