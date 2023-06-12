#### Dataset tool
#sbatch run_dataset_tool.sh

## Train StyleGAN3
#sbatch run_gan_mri.sh

## Train multisource morpho
#sbatch run_multisource_gan_morpho.sh

## Train multisource ukb
#sbatch run_multisource_gan_mri.sh

## Train single source 
### As baseline
## morpho
#sbatch run_gan_morpho.sh

## UKB
#sbatch run_gan_mri_single.sh

### Generation ###
sbatch run_image_generation.sh

### Stratified FIDs
#sbatch run_multisource_stratified_fid.sh

### General FIDs
#sbatch run_general_fid.sh
#sbatch run_general_fid_morpho.sh
