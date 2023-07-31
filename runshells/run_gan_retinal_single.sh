#!/bin/bash
#SBATCH --job-name=ret_source2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
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
## Retinal source1
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/trainset /scratch/wei-cheng.lai/retinal_source1/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/valset /scratch/wei-cheng.lai/retinal_source1/
#rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/testset /scratch/wei-cheng.lai/retinal_source1/

#### Resolution 256
## Retinal source2
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/trainset /scratch/wei-cheng.lai/retinal_source2/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/valset /scratch/wei-cheng.lai/retinal_source2/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/testset /scratch/wei-cheng.lai/retinal_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 256
## StyleGAN-T
## Source 1 (Age, BP)
## train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/retinal/source1_diastolic_log/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/retinal_source1/ --data_name retinal --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario high --wandb_name cond_source1_diastolic_log --wandb_pj_v singlesource_retinal

## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/retinal/source1_diastolic_log/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/retinal_source1/ --data_name retinal --gpus 1 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 5000 --cond True --data_scenario high --wandb_name cond_source1_diastolic_log_re04 --wandb_pj_v singlesource_retinal --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source1_diastolic_log/stylegan-t/00004-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg2800/metric-fid50k_full.jsonl

### Source 2 (Age, Cyclinder)
## Train from the beginning
## COND
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/retinal/source2_spherical_log/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/retinal_source2/ --data_name retinal --gpus 2 --batch 256 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 9000 --cond True --data_scenario high --wandb_name cond_source2_spherical_log --wandb_pj_v singlesource_retinal

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train.py --outdir ~/experiments/singlesource/retinal/source2_spherical_log/stylegan-t --cfg stylegan3-t --data /scratch/wei-cheng.lai/retinal_source2/ --data_name retinal --gpus 1 --batch 128 --gamma 4 --batch-gpu 32 --snap 10 --mirror 0 --aug noaug --kimg 3400 --cond True --data_scenario high --wandb_name cond_source2_spherical_log_re05 --wandb_pj_v singlesource_retinal --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source2_spherical_log/stylegan-t/00005-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg3200/metric-fid50k_full.jsonl
