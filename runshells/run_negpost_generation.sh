#!/bin/bash
#SBATCH --job-name=gen_image_retinal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/images_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## Transport datasets
#### Resolution 32
## Morpho Source 1
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/trainset /scratch/wei-cheng.lai/mnist_source1/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/valset /scratch/wei-cheng.lai/mnist_source1/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/testset /scratch/wei-cheng.lai/mnist_source1/

## Morpho Source 2
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/trainset /scratch/wei-cheng.lai/mnist_source2/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/valset /scratch/wei-cheng.lai/mnist_source2/
#rsync -a --include '*/' --include '*.png' --include '*.json' --exclude '*' /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/testset /scratch/wei-cheng.lai/mnist_source2/

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

### Retinal
### Resolution 256
## Source1
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/trainset /scratch/wei-cheng.lai/retinal_source1/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/valset /scratch/wei-cheng.lai/retinal_source1/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1/testset /scratch/wei-cheng.lai/retinal_source1/
## Source2
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/trainset /scratch/wei-cheng.lai/retinal_source2/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/valset /scratch/wei-cheng.lai/retinal_source2/
rsync -a --include '*/' --include '*.jpg' --include '*.json' --exclude '*' /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2/testset /scratch/wei-cheng.lai/retinal_source2/

echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

### Resolution 32
## StyleGAN-T
## Multi-Source
## train from the beginning
## Morpho-MNSIT
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visual_neg_pos.py --outdir ~/experiments/multisources/morpho/plots/estimated_half/neg_post_covs/ --dataset mnist-thickness-intensity-slant --data-path1 /scratch/wei-cheng.lai/mnist_source1/ --data-path2 /scratch/wei-cheng.lai/mnist_source2/ --group-by c1 --network /dhc/home/wei-cheng.lai/experiments/multisources/morpho/estimated_half/stylegan-t/00002-stylegan3-t-condTrue-multisource-mnist-thickness-intensity-slant-augnoaug-gpus2-batch256-gamma4-kimg10000/metric-fid50k_full.jsonl
## MRI
## Estimated
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs.py --outdir ~/experiments/multisources/ukb/plots/twocovs/ --dataset ukb --data-path1 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --data-path2 /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --seeds 0,1,2,64-75 --network /dhc/home/wei-cheng.lai/experiments/multisources/ukb/stylegan-t/00004-stylegan3-t-condTrue-multisource-ukb-augnoaug-gpus2-batch128-gamma4-kimg6000/metric-fid50k_full.jsonl

## Retinal
## Estimated
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visual_neg_pos.py --outdir ~/experiments/multisources/retinal/plots/neg_post_covs/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --group-by c1 --network /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half/stylegan-t/00001-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch128-gamma4-kimg7000/metric-fid50k_full.jsonl

#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visualizer_covs.py --outdir ~/experiments/multisources/retinal/plots/twocovs/c2/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --seeds 0-30 --group-by c2 --network /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half/stylegan-t/00001-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch128-gamma4-kimg7000/metric-fid50k_full.jsonl

#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visualizer_covs.py --outdir ~/experiments/multisources/retinal/plots/twocovs/c3/ --dataset retinal --data-path1 /scratch/wei-cheng.lai/retinal_source1/ --data-path2 /scratch/wei-cheng.lai/retinal_source2/ --seeds 0-30 --group-by c3 --network /dhc/home/wei-cheng.lai/experiments/multisources/retinal/retinal_half/stylegan-t/00001-stylegan3-t-condTrue-multisource-retinal-augnoaug-gpus2-batch128-gamma4-kimg7000/metric-fid50k_full.jsonl
### Single Source
## Morpho
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/morpho/plots/source1/twocovs/ --dataset mnist-thickness-intensity-slant --data-path /scratch/wei-cheng.lai/mnist_source1/ --seeds 0-30 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source1/stylegan-t/00001-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl

## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/morpho/plots/source2/twocovs/ --dataset mnist-thickness-intensity-slant --data-path /scratch/wei-cheng.lai/mnist_source2/ --seeds 0-30 --network /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/source2/stylegan-t/00001-stylegan3-t-condTrue-mnist-thickness-intensity-slant-augnoaug-gpus1-batch256-gamma4-kimg6000/metric-fid50k_full.jsonl

## MRI
## UKB
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/ukb/plots/source1/twocovs/ --dataset ukb --data-path /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source1/ --seeds 64-75 --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source1/stylegan-t/00004-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg3000/metric-fid50k_full.jsonl

## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/causal_gans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/ukb/plots/source2/twocovs/ --dataset ukb --data-path /scratch/wei-cheng.lai/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2/ --seeds 64-75 --network /dhc/home/wei-cheng.lai/experiments/singlesource/ukb/source2/stylegan-t/00005-stylegan3-t-condTrue-ukb-augnoaug-gpus1-batch128-gamma4-kimg4000/metric-fid50k_full.jsonl

## Retinal
## UKB
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/retinal/plots/source1/twocovs/ --dataset retinal --data-path /scratch/wei-cheng.lai/retinal_source1/ --seeds 0-30 --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source1_diastolic/stylegan-t/00001-stylegan3-t-condTrue-retinal-augnoaug-gpus2-batch128-gamma4-kimg9000/metric-fid50k_full.jsonl

## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/visualizer_covs_singlesource.py --outdir ~/experiments/singlesource/retinal/plots/source2/twocovs/ --dataset retinal --data-path /scratch/wei-cheng.lai/retinal_source2/ --seeds 0-30 --network /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/source2_spherical/stylegan-t/00001-stylegan3-t-condTrue-retinal-augnoaug-gpus1-batch128-gamma4-kimg7000/metric-fid50k_full.jsonl
