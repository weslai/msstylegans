#!/bin/bash
#SBATCH --job-name=ukb_gen_image
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/images_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/causal_gans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## Transport datasets
#### Resolution 32
## Intensity
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity/trainset /scratch/wei-cheng.lai/mnist_thickness_intensity/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity/valset /scratch/wei-cheng.lai/mnist_thickness_intensity/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity/testset /scratch/wei-cheng.lai/mnist_thickness_intensity/

## Slant
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/trainset /scratch/wei-cheng.lai/mnist_thickness_slant/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/valset /scratch/wei-cheng.lai/mnist_thickness_slant/
rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant/testset /scratch/wei-cheng.lai/mnist_thickness_slant/

### MRI
#### Resolution 256
## UKB source1
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/trainset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/valset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/testset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/

#### Resolution 256
## UKB source2
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/trainset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/valset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/
#rsync -a --include '*/' --include '*.gz' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/testset /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 32
## StyleGAN-T
## Multi-Source
## train from the beginning
## COND (linear)
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs.py --outdir ~/experiments/multisources/morpho/plots/groundtruth/twocovs/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --seeds 0,1,2,64-100 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/groundtruth/stylegan-t/00004-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl


## COND (estimate)
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs.py --outdir ~/experiments/multisources/morpho/plots/estimated/twocovs/ --dataset mnist-thickness-intensity --data-path1 /scratch/wei-cheng.lai/mnist_thickness_intensity/ --data-path2 /scratch/wei-cheng.lai/mnist_thickness_slant/ --seeds 0,1,2,64-100 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/estimated/stylegan-t/00001-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-augnoaug-gpus1-batch128-gamma4-kimg5000/metric-fid50k_full.jsonl


## MRI
## Estimated
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs.py --outdir ~/experiments/multisources/ukb/plots/twocovs/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data-path2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --seeds 0,1,2,64-75 --network /dhc/home/wei-cheng.lai/experiments/multisources/ukb/stylegan-t/00004-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch128-gamma4-kimg6000/metric-fid50k_full.jsonl

### Single Source
## Morpho
## Thickness intensity
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/morpho/plots/thickness_intensity/twocovs/ --dataset mnist-thickness-intensity --data-path /scratch/wei-cheng.lai/mnist_thickness_intensity/ --seeds 64-75 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_intensity/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-intensity-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl

## Thickness slant
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/morpho/plots/thickness_slant/twocovs/ --dataset mnist-thickness-slant --data-path /scratch/wei-cheng.lai/mnist_thickness_slant/ --seeds 64-75 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/thickness_slant/stylegan-t/00002-stylegan3-t-condTrue-mnist-thickness-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl


## MRI
## UKB
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/ukb/plots/source1/twocovs/ --dataset ukb --data-path /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --seeds 64-75 --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1/stylegan-t/00004-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg3000/metric-fid50k_full.jsonl

## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/ukb/plots/source2/twocovs/ --dataset ukb --data-path /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --seeds 64-75 --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg4000/metric-fid50k_full.jsonl


