## MSSG: Multi-Source StyleGAN

Multi-Source StyleGAN is a method to generate images with different available covariates from multiple heterogeneous data sources. This repository provides code to train your own Multi-Source StyleGAN.
We work on the official [StyleGAN3 model](https://github.com/NVlabs/stylegan3) and this work is accepted by MIDL 2024.

![Teaser image](./docs/mri_age_59.0.pdf)

**Heterogeneous Medical Data Integration with Multi-Source StyleGAN**<br>
Wei-Cheng Lai, Matthias Kirchler, Hadya Yassin, Jana Fehr, Alexander Rakowski, Hampus Olsson, Ludger Starke, Jason M. Millward, Sonia Waiczies, Christoph Lippert<br>

Abstract: *Conditional deep generative models have emerged as powerful tools for generating realistic images enabling fine-grained control over latent factors. 
In the medical domain, data scarcity and the need to integrate information from diverse sources present challenges for existing generative models, often resulting in low-quality image generation and poor controllability. 
To address these two issues, we propose Multi-Source StyleGAN (MSSG). MSSG learns jointly from multiple heterogeneous data sources with different available covariates and can generate new images controlling all covariates together, thereby overcoming both data scarcity and heterogeneity.
We validate our method on semi-synthetic data of hand-written digit images with varying morphological features and in controlled multi-source simulations on retinal fundus images and brain magnetic resonance images. Finally, we apply MSSG in a real-world setting of brain MRI from different sources. Our proposed algorithm offers a promising direction for unbiased data generation from disparate sources.*


### Installation
1. Clone the official StyleGAN3 repository: 
`git clone https://github.com/NVlabs/stylegan3.git`

2. Clone this repository:
`git clone https://github.com/weslai/msstylegans.git`

3. Replace the official codes with our codes, which have the same file names. The other files should stay unchanged. The files that should be replaced are `dataset_tool.py`, `train.py` in the directory `stylegan3`, `dataset.py`, `loss.py`, `networks_stylegan2.py`, `networks_stylegan3.py` and `training_loop.py` in the directory `stylegan3/training`.

### Structure of the repository
* `eval_regr` contains codes to train prediction models with ResNet backbones for the evaluation of the controlability.
* `plots` contains codes to generate the plots used for the manuscript.
* `stylegan3` is the main folder to train MSSG and generate synthetic images. Furthermore, it contains also the evaluation metrics.
* `mri_prep.py` and `retina_prep.py` are used to prepare the datasets from UKBiobank.

## Create Dataset
1. Use `mri_prep.py` and `retina_prep.py` to prepare MRI and Retina data from UKBiobank.

2. To create datasets, use `dataset_tool.py` or `dataset_tool_extrasources.py` in the directory `stylegan3`.

3. For Morpho-MNISTs, use the function `open_mnist` from `dataset_tool.py` to create datasets.

4. For MRI datasets, use the function `open_ukb` from `dataset_tool.py` to create UKBiobank datasets and the function `open_adni` from `dataset_tool_extrasources.py` to create ADNI datasets.

5. For retinal datasets, use the function `open_ukb_retinal` from `dataset_tool.py`to create UKBiobank datasets.

## Train MSSG models
1. To train MSSG models, one needs to specify the covariates from each datasets and first trains a latent model to sample the covariates. For example, sampling brain grey matter and ventricle volumes given ages. `latent_mle_ms.py` provides code to train a latent model and use it to sample covariates for semi-multisource dataset in UKBiobank and `latent_dist_morphomnist.py` provides code for the same purpose with Morpho-MNISTs. `latent_mle_real_ms.py` provides code for the multi-source datasets, for example UKBiobank and ADNI.

2. With latent models, one can train a MSSG model with `train.py`, `train_ms.py` and `train_real_ms.py` in the directory `stylegan3`. `train.py` runs codes to train the baseline single-source models. `train_ms.py` runs to train the proposed multi-source models in the semi-multisource scenarios. `train_real_ms.py` runs to train the proposed multi-source models in the real multi-source scenario. In this paper, we used UKBiobank and ADNI for the real multi-source scenario.

3. `training_loop.py`, `training_loop_ms.py` and `training_loop_real_ms.py` in the directory `stylegan3/training` will be called to train the model. In these files, latent models are used to sample covariates, which are fitted into the generator and discriminator.

4. For this work, we specified the loss function for each covariates. For continuous variables, we used MSE loss. For the binary variables, we then used binary cross entropy loss. These are specified in `loss.py`, `loss_ms.py` and `loss_real_ms.py`.

***Clarify the name of files: files with the name `_ms.py` are used in the semi-multisource scenario and files with the name `_real_ms.py` are used in the real multisource scenario.***

## Evaluation metrics
In this work, we had two types of evaluations, image quality and controllability. Typical Frechet Inception Distance (FID) and Kernel Inception Distance (KID) are used. Furthermore, we included the pairwise metrics, Learned Perceptual Image Patch Similarity (LPIPS),Structual Similarity Index Measure (SSIM) and Peak Signal-To-Noise Ratio (PSNR). Furthermore, we proposed a new metric, strata prediction score, to measure the controllability of continuous covariates. 

* The files `eval_general_fids.py`, `eval_general_fids_ms.py` in the directory `stylegan3/evaluations` is used to evaluate FID or KID with the test set in the semi-multisource scenario. One can specify this with `--metric`.

* The file `eval_general_lpips.py`in the directory `stylegan3/evaluations` is used to evaluate LPIPS, SSIM or PSNR with the test set in the semi-multisource scenario. One can specify this with `--metric`.

* The file `eval_general_jointfids_ms.py` in the directory `stylegan3/evaluations` is used to evaluate FID or KID with the test set in the real multisource scenario. One can specify this with `--metric`.

* The file `eval_strata_mse.py` in the directoy `stylegan3/evaluations` is used to evaluate the strata-based controllability of continuous variables with the MSE loss. For this evaluation, one needs (three) prediction models for evaluating synthetic images.

* The file `eval_general_mse_real_ms.py` is used to evaluate the controllability. However, this is not a strata-based metric, but take the average of the MSE loss from the whole test set.

## Visualization 

1. `visualizer_covs.py` in the directory `stylegan3/visualizers` is used to plot the Figure 1. from the manuscript. With the interpolated covariates along axis, it shows that our model is able to control images with latent covariates from separate datasources in the semi-multisource scenario.

2. `visualizer_covs_real_ms.py` in the directory `stylegan3/visualizers` is used to plot Figure 10, 11 and 12. With this, one is able to plot synthetic images in the real multisource scenario.

3. `heatmap_visual_regr_mse.py` in the directory `stylegan3/visualizers` is used to generate Figure 3. This is the heatmap of the strata prediction scores.



## Citation
