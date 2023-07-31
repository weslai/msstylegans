#!/bin/bash
#SBATCH --job-name=deyepacs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --partition=hpcpu # -p
#SBATCH --cpus-per-task=5 # -c
#SBATCH --mem=30gb
#SBATCH --time=25:30:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/data_tool_%j.log # %j is job id

## run dataset tool to create dataset

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH

## MRI UKBiobank
## Source1
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/ --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1 --dataset_name ukb --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/extract_source1.csv --which_dataset train --resolution 256x256
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/ --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1 --dataset_name ukb --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/extract_source1.csv --which_dataset val --resolution 256x256
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/ --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source1 --dataset_name ukb --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/extract_source1.csv --which_dataset test --resolution 256x256
## Source2
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/ --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2 --dataset_name ukb --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/extract_source2.csv --which_dataset train --resolution 256x256
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/ --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2 --dataset_name ukb --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/extract_source2.csv --which_dataset val --resolution 256x256
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/ --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/source2 --dataset_name ukb --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/extract_source2.csv --which_dataset test --resolution 256x256

### UKBiobank
## Retinal
## source1
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1 --dataset_name retinal --which_dataset train --resolution 256x256 --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/phenotype_source0.csv

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1 --dataset_name retinal --which_dataset val --resolution 256x256 --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/phenotype_source0.csv

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source1 --dataset_name retinal --which_dataset test --resolution 256x256 --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/phenotype_source0.csv
## source2
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2 --dataset_name retinal --which_dataset train --resolution 256x256 --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/phenotype_source1.csv

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2 --dataset_name retinal --which_dataset val --resolution 256x256 --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/phenotype_source1.csv

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2 --dataset_name retinal --which_dataset test --resolution 256x256 --annotation_path /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/phenotype_source1.csv

### UKBiobank
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/utils.py
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/derived/imaging/retinal_fundus --dest /dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/multisources/source2 --dataset_name retinal --which_dataset train --resolution 256x256

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2 --dataset_name ukb --which_dataset val --resolution 256x256

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped --dest /dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/T1_3T_coronal_mni_linear_freesurfer_resolution256_source2 --dataset_name ukb --which_dataset test --resolution 256x256

### Adni
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/adni_t1_mprage --dest /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 --dataset_name adni --which_dataset train --resolution 256x256

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/adni_t1_mprage --dest /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 --dataset_name adni --which_dataset val --resolution 256x256

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/adni_t1_mprage --dest /dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256 --dataset_name adni --which_dataset test --resolution 256x256


## Thickness Slant Dataset 
## Causal Thickness to slant
#srun --ntasks=1 ~/conda3/envs/deepscm/bin/python ../deepscm/main/datasets/morphomnist/create_synth_thickness_slant_data.py --data-dir /dhc/home/wei-cheng.lai/data/MNIST/morphomnist_global -o /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_slant

## Thickness Intensity Slant dataset
#srun --ntasks=1 ~/conda3/envs/deepscm/bin/python ../deepscm/main/datasets/morphomnist/create_synth_thickness_intensity_slant_data.py --data-dir /dhc/home/wei-cheng.lai/data/MNIST/morphomnist_global -o /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_intensity_slant

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_intensity_slant_new/train-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant --dataset_name mnist-thickness-intensity-slant --which_dataset train --resolution 32x32 --annotation_path None

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/thickness_intensity_slant_new/t10k-images-idx3-ubyte.gz --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant --dataset_name mnist-thickness-intensity-slant --which_dataset test --resolution 32x32 --annotation_path None

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_split_source.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/ --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/ --dataset_name mnist-thickness-intensity-slant --which_dataset train --resolution 32x32 --annotation_path /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/source1.json

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_split_source.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/ --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/ --dataset_name mnist-thickness-intensity-slant --which_dataset val --resolution 32x32 --annotation_path /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/source1.json

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_split_source.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/testset/ --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source1/ --dataset_name mnist-thickness-intensity-slant --which_dataset test --resolution 32x32 --annotation_path /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/testset/source1.json

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_split_source.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/ --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/ --dataset_name mnist-thickness-intensity-slant --which_dataset train --resolution 32x32 --annotation_path /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/source2.json

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_split_source.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/ --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/ --dataset_name mnist-thickness-intensity-slant --which_dataset val --resolution 32x32 --annotation_path /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/trainset/source2.json

#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_split_source.py --source /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/testset/ --dest /dhc/groups/fglippert/MorphoMNIST/mnist_instances/source2/ --dataset_name mnist-thickness-intensity-slant --which_dataset test --resolution 32x32 --annotation_path /dhc/groups/fglippert/MorphoMNIST/mnist_instances/mnist_thickness_intensity_slant/testset/source2.json


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

###########################################################################
## Extra sources

## Kaggle Eyepacs
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_extrasources.py --source /dhc/dsets/diabetic_retinopathy --dest /dhc/groups/fglippert/kaggle_eyepacs_diabetic --dataset_name retinal --annotation_path /dhc/dsets/diabetic_retinopathy/trainLabels.csv.zip --which_dataset train --resolution 256x256
#srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_extrasources.py --source /dhc/dsets/diabetic_retinopathy --dest /dhc/groups/fglippert/kaggle_eyepacs_diabetic --dataset_name retinal --annotation_path /dhc/dsets/diabetic_retinopathy/trainLabels.csv.zip --which_dataset val --resolution 256x256
srun --ntasks=1 ~/conda3/envs/stylegan3_test/bin/python ../stylegan3/dataset_tool_extrasources.py --source /dhc/dsets/diabetic_retinopathy --dest /dhc/groups/fglippert/kaggle_eyepacs_diabetic --dataset_name retinal --annotation_path /dhc/home/wei-cheng.lai/data/diabetic_retinopathy/testLabels.csv.zip --which_dataset test --resolution 256x256

