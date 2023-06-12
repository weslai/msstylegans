#!/bin/bash
#SBATCH --job-name=dest_ukb_multi
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --partition=vcpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=100gb
#SBATCH --time=04:30:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/data_tool_%j.log # %j is job id

## run dataset tool to create dataset

PYTHONPATH=$PYTHONPATH:~/causal-gan/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

### UKBiobank
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/utils.py
srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2 --dataset_name ukb --which_dataset train --resolution 256x256

srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2 --dataset_name ukb --which_dataset val --resolution 256x256

srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2 --dataset_name ukb --which_dataset test --resolution 256x256

### Oasis3
#srun --ntasks=1 ~/conda3/envs/stylegan3/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/Oasis3 --dest /dhc/groups/fglippert/Oasis3/t1_coronal_mni_re256 --resolution=256x256

### Adni
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/adni_t1_mprage --dest /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 --dataset_name adni --which_dataset train --resolution 256x256

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/adni_t1_mprage --dest /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 --dataset_name adni --which_dataset val --resolution 256x256

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/adni_t1_mprage --dest /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 --dataset_name adni --which_dataset test --resolution 256x256

### MorphoMNIST
## mnist (morphomnist) - in StyleGAN3 framework
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/thickness_intensity/t10k-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/thick_intensity_causal_test --resolution=32x32

## mnist - in StyleGAN3 framework
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/home/wei-cheng.lai/data/MNIST/raw/t10k-images-idx3-ubyte.gz --dest /dhc/home/wei-cheng.lai/data/MNIST/mnist/ --is_trainset 0 --resolution=32x32

## Thickness intensity
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/thickness_intensity/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_thickness_intensity --dataset_name mnist-thickness-intensity --which_dataset train --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/thickness_intensity/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_thickness_intensity --dataset_name mnist-thickness-intensity --which_dataset val --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/thickness_intensity/t10k-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_thickness_intensity --dataset_name mnist-thickness-intensity --which_dataset test --resolution=32x32

## Thickness slant
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_slant/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant --dataset_name mnist-thickness-slant --which_dataset train --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_slant/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant --dataset_name mnist-thickness-slant --which_dataset val --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_slant/t10k-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_slant --dataset_name mnist-thickness-slant --which_dataset test --resolution=32x32

## MRI mask segmentation
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/evaluations/segmentation/dataset_tool_mri_masks.py --dataset_name $1 --annotation_path $2 --dir_path $3 --dest $4 

## Thickness Slant Dataset 
## Causal Thickness to slant
#srun --ntasks=1 ~/conda3/envs/deepscm/bin/python ../deepscm/main/datasets/morphomnist/create_synth_thickness_slant_data.py --data-dir /dhc/home/wei-cheng.lai/data/MNIST/morphomnist_global -o /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_slant

## Thickness pure
#srun --ntasks=1 ~/conda3/envs/deepscm/bin/python ../deepscm/main/datasets/morphomnist/create_synth_thickness_intensity_data.py --data-dir /dhc/home/wei-cheng.lai/data/MNIST/morphomnist_global -o /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness --with_intensity 0

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness --dataset_name mnist-thickness --which_dataset train --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness --dataset_name mnist-thickness --which_dataset val --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness/t10k-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness --dataset_name mnist-thickness --which_dataset test --resolution=32x32

## Thickness 2
#srun --ntasks=1 ~/conda3/envs/deepscm/bin/python ../deepscm/main/datasets/morphomnist/create_synth_thickness_width_data.py --data-dir /dhc/home/wei-cheng.lai/data/MNIST/morphomnist_global -o /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness2 --with_width 0

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness2/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness2 --dataset_name mnist-thickness --which_dataset train --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness2/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness2 --dataset_name mnist-thickness --which_dataset val --resolution=32x32

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness2/t10k-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness2 --dataset_name mnist-thickness --which_dataset test --resolution=32x32

## Thickness-width
#srun --ntasks=1 ~/conda3/envs/deepscm/bin/python ../deepscm/main/datasets/morphomnist/create_synth_thickness_width_data.py --data-dir /dhc/home/wei-cheng.lai/data/MNIST/morphomnist_global -o /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_width --with_width 1
