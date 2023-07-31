#!/bin/bash
#SBATCH --job-name=ms_morpho
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=60:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/gan_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## Transport datasets
#### Resolution 32
## thickness intensity slant
## source1
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/trainset /scratch/wei-cheng.lai/mnist_source1/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/valset /scratch/wei-cheng.lai/mnist_source1/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/testset /scratch/wei-cheng.lai/mnist_source1/

## source2
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/trainset /scratch/wei-cheng.lai/mnist_source2/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/valset /scratch/wei-cheng.lai/mnist_source2/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/testset /scratch/wei-cheng.lai/mnist_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 32
## StyleGAN-T
## Multi-Source
## train from the beginning
## COND (linear)
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/morpho/groundtruth/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/mnist_source1/ --data_name1 mnist-thickness-intensity-slant --data2 /scratch/wei-cheng.lai/mnist_source2/ --data_name2 mnist-thickness-intensity-slant --use_ground_truth True --gpus 2 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 10000 --cond True --wandb_name cond_gt --wandb_pj_v multisource_morpho

## Resume
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/train_ms.py --outdir ~/experiments/multisources/morpho/groundtruth/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data_name1 mnist-thickness-intensity --data2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --data_name2 mnist-thickness-slant --use_ground_truth True --gpus 1 --batch 128 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 5000 --cond True --wandb_name cond_gt_sep_resume01 --wandb_pj_v multisource_morpho --resume /dhc/home/wei-cheng.lai/experiments/multisources/morpho/groundtruth/stylegan-t/00003-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-augnoaug-gpus1-batch128-gamma4-kimg14000/metric-fid50k_full.jsonl

## COND (estimate)
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/morpho/estimated_half_norm/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/mnist_source1/ --data_name1 mnist-thickness-intensity-slant --data2 /scratch/wei-cheng.lai/mnist_source2/ --data_name2 mnist-thickness-intensity-slant --use_ground_truth False --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 10000 --cond True --data_scenario half --wandb_name cond_gt_estimated_norm --wandb_pj_v multisource_morpho_half

## Resume
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/train_ms.py --outdir ~/experiments/multisources/morpho/estimated_half_norm/stylegan-t --cfg stylegan3-t --data1 /scratch/wei-cheng.lai/mnist_source1/ --data_name1 mnist-thickness-intensity-slant --data2 /scratch/wei-cheng.lai/mnist_source2/ --data_name2 mnist-thickness-intensity-slant --use_ground_truth False --gpus 1 --batch 256 --gamma 4 --batch-gpu 64 --snap 10 --mirror 0 --aug noaug --kimg 6000 --cond True --data_scenario half --wandb_name cond_estimate_norm_re05 --wandb_pj_v multisource_morpho_half --exact_resume False --resume /dhc/home/wei-cheng.lai/experiments/multisources/morpho/estimated_half_norm/stylegan-t/00005-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg5400/metric-fid50k_full.jsonl
