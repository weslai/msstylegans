#echo "Hello there"
#sleep 7000
#echo "oops! now we will start"

#### Dataset tool
#sbatch run_dataset_tool.sh

## Train real Multisource retinal/mri
#sbatch run_multisource_gan_real.sh

## Train multisource morpho
#sbatch run_multisource_gan_morpho.sh

## Train multisource ukb
#sbatch run_multisource_gan_mri.sh

## Train multisource retinal
#sbatch run_multisource_gan_retinal.sh

## Train single source 
### As baseline
## morpho
#sbatch run_gan_morpho.sh

## UKB
#sbatch run_gan_mri_single.sh
## Retinal
#sbatch run_gan_retinal_single.sh

### Generation ###
#sbatch run_image_generation.sh
#sbatch run_negpost_generation.sh

### Stratified FIDs
sbatch run_multisource_stratified_fid.sh

### General FIDs
#sbatch run_general_fid.sh
#sbatch run_general_fid_morpho.sh

### Heatmaps
#sbatch run_heatmap.sh
