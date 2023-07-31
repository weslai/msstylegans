#!/bin/bash
#SBATCH --job-name=heatmap
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wei-cheng.lai@hpi.de
#SBATCH --nodes=1 # num nodes
#SBATCH --ntasks-per-node=1  #run once the next srun per node
#SBATCH --partition=vcpu # -p
#SBATCH --cpus-per-task=6 # -c
#SBATCH --mem=20gb
#SBATCH --gpus=0
#SBATCH --time=02:00:00
#SBATCH --output=/dhc/home/wei-cheng.lai/experiments/logs/multisources/heatmaps_%j.log # %j is job id

PYTHONPATH=$PYTHONPATH:~/projects/msstylegans/stylegan3/
export PYTHONPATH
echo $PYTHONPATH
echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

## Heatmap visualization
## Morpho-MNIST
## Thickness-Intensity-Slant
## Multi-Sources
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/heatmap_visual.py --dataset mnist-thickness-intensity-slant --ms_path /dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_half_norm/fids/ms_stratified_fid.csv --source1_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source1_norm/fids/ms_stratified_fid.csv --source2_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source2_norm/fids/ms_stratified_fid.csv --group_by c1 --title multi-source --goal ms --save_path /dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_half_norm/fid_heatmaps/heatmap

## Source1
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/heatmap_visual.py --dataset mnist-thickness-intensity-slant --ms_path /dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_half_norm/fids/ms_stratified_fid.csv --source1_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source1_norm/fids/ms_stratified_fid.csv --source2_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source2_norm/fids/ms_stratified_fid.csv --group_by c1 --title source1 --goal source1 --save_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source1_norm/fid_heatmaps/heatmap

## Source2
srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/heatmap_visual.py --dataset mnist-thickness-intensity-slant --ms_path /dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_half_norm/fids/ms_stratified_fid.csv --source1_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source1_norm/fids/ms_stratified_fid.csv --source2_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source2_norm/fids/ms_stratified_fid.csv --group_by c1 --title source2 --goal source2 --save_path /dhc/home/wei-cheng.lai/experiments/singlesource/morpho/plots/source2_norm/fid_heatmaps/heatmap

## Retinal
## Multi-Sources
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/heatmap_visual.py --dataset retinal --ms_path /dhc/home/wei-cheng.lai/experiments/multisources/retinal/fids/ms_stratified_fid.csv --source1_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/fids/source1/ms_stratified_fid.csv --source2_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/fids/source2/ms_stratified_fid.csv --group_by c1 --title multi-source --goal ms --save_path /dhc/home/wei-cheng.lai/experiments/multisources/retinal/plots/fid_heatmaps/heatmap
## Source1
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/heatmap_visual.py --dataset retinal --ms_path /dhc/home/wei-cheng.lai/experiments/multisources/retinal/fids/ms_stratified_fid.csv --source1_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/fids/source1/ms_stratified_fid.csv --source2_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/fids/source2/ms_stratified_fid.csv --group_by c1 --title source1 --goal source1 --save_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/plots/source1/fid_heatmaps/heatmap
## Source2
#srun ~/conda3/envs/stylegan3_test/bin/python ~/projects/msstylegans/stylegan3/visualizers/heatmap_visual.py --dataset retinal --ms_path /dhc/home/wei-cheng.lai/experiments/multisources/retinal/fids/ms_stratified_fid.csv --source1_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/fids/source1/ms_stratified_fid.csv --source2_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/fids/source2/ms_stratified_fid.csv --group_by c1 --title source2 --goal source2 --save_path /dhc/home/wei-cheng.lai/experiments/singlesource/retinal/plots/source2/fid_heatmaps/heatmap
