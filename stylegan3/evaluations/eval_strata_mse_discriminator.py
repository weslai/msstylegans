import os, sys
from typing import Tuple
import click
import json
import numpy as np
import pandas as pd
import torch
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans")
import dnnlib

### --- Own --- ###
from eval_utils import calc_mean_scores_disc
from utils import load_generator, generate_images, load_regression_model, load_discriminator
from eval_strata_mse import parse_vec2, load_dataset
# --------------------------------------------------------------------------------------
def calculate_loss(
    data_name: str,
    strata_idxs: list,
    covariates: dict,
    labels1: np.ndarray,
    labels2: np.ndarray,
    dataset1: torch.utils.data.Dataset,
    dataset2: torch.utils.data.Dataset,
    source_gan: str,
    Gen,
    Disc,
    regr_model,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float, float],
    rotate: float
):
    scores = [] ## mse, mae
    strata_predictions = []
    c1_min, c1_max = covariates["c1_min"], covariates["c1_max"]
    c2_min, c2_max = covariates["c2_min"], covariates["c2_max"]
    c3_min, c3_max = covariates["c3_min"], covariates["c3_max"]
    strata_hist = covariates["strata_hist"]
    for key, value in covariates["cov"].items():
        cov = value

    ## strata loop
    for stra_c1 in strata_idxs:
        if stra_c1 == 0:
            cur_c1 = (c1_min, strata_hist["c1"][stra_c1])
        elif stra_c1 == 1:
            cur_c1 = (strata_hist["c1"][stra_c1-1], strata_hist["c1"][stra_c1])
        else:
            cur_c1 = (strata_hist["c1"][stra_c1-1], c1_max)
        for stra_c2 in strata_idxs:
            if stra_c2 == 0:
                cur_c2 = (c2_min, strata_hist["c2"][stra_c2])
            elif stra_c2 == 1:
                cur_c2 = (strata_hist["c2"][stra_c2-1], strata_hist["c2"][stra_c2])
            else:
                cur_c2 = (strata_hist["c2"][stra_c2-1], c2_max)
            for stra_c3 in strata_idxs:
                if stra_c3 == 0:
                    cur_c3 = (c3_min, strata_hist["c3"][stra_c3])
                elif stra_c3 == 1:
                    cur_c3 = (strata_hist["c3"][stra_c3-1], strata_hist["c3"][stra_c3])
                else:
                    cur_c3 = (strata_hist["c3"][stra_c3-1], c3_max)
                real_imgs = []
                gen_imgs = []
                cov_labels = []
                ## get samples from datasets (idxs)
                idxs1 = np.where((labels1[:,0] >= cur_c1[0]) & (labels1[:,0] < cur_c1[1]) & \
                                (labels1[:,1] >= cur_c2[0]) & (labels1[:,1] < cur_c2[1]) & \
                                (labels1[:,2] >= cur_c3[0]) & (labels1[:,2] < cur_c3[1]))[0]
                idxs2 = np.where((labels2[:,0] >= cur_c1[0]) & (labels2[:,0] < cur_c1[1]) & \
                                (labels2[:,1] >= cur_c2[0]) & (labels2[:,1] < cur_c2[1]) & \
                                (labels2[:,2] >= cur_c3[0]) & (labels2[:,2] < cur_c3[1]))[0]
                num_real = len(idxs1) + len(idxs2)
                if num_real >= 5:
                    ## get samples from GANs
                    for _ in range(num_samples // batch_size):
                        z = torch.randn(batch_size, Gen.z_dim).to(device)
                        ### 
                        source1_c = []
                        source2_c = []
                        source1_img = []
                        source2_img = []
                        if len(idxs1) != 0:
                            for idx in np.random.choice(idxs1, batch_size//2):
                                source1_c.append(dataset1.get_norm_label(idx))
                                source1_img.append(torch.tensor(dataset1[idx][0]))
                        if len(idxs1) == 0:
                            for idx in np.random.choice(idxs2, batch_size):
                                source2_c.append(dataset2.get_norm_label(idx))
                                source2_img.append(torch.tensor(dataset2[idx][0]))
                        else:
                            for idx in np.random.choice(idxs2, batch_size//2):
                                source2_c.append(dataset2.get_norm_label(idx))
                                source2_img.append(torch.tensor(dataset2[idx][0]))
                        all_c = source1_c + source2_c
                        all_img = source1_img + source2_img
                        l = torch.from_numpy(np.stack(all_c, axis=0)).to(device)
                        imgs = torch.stack(all_img, dim=0).repeat([1,1,1,1]).to(device)
                        if source_gan != "multi":
                            if data_name == "mnist-thickness-intensity-slant":
                                if source_gan == "source1":
                                    list_idx = [0, 1] + [i for i in range(3, 13)]
                                elif source_gan == "source2":
                                    list_idx = [0, 2] + [i for i in range(3, 13)]
                                gen_l = l[:, list_idx]
                            else:
                                if source_gan == "source1":
                                    gen_l = l[:, :2]
                                elif source_gan == "source2":
                                    gen_l = l[:, [0, 2]]
                        batch_imgs = generate_images(Gen, z, gen_l if source_gan != "multi" else l, 
                                                    truncation_psi, 
                                                    noise_mode, translate, rotate).permute(0,3,1,2)
                        if batch_imgs.shape[1] == 3:
                            mu=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            batch_imgs = batch_imgs.div(255).cpu().detach()
                            imgs = imgs.div(255).cpu().detach()
                            for i in range(len(mu)):
                                batch_imgs[:, i, :, :] = (batch_imgs[:, i, :, :] - mu[i]) / std[i]
                                imgs[:, i, :, :] = (imgs[:, i, :, :] - mu[i]) / std[i]
                        else:
                            batch_imgs = batch_imgs.div(255).cpu().detach()
                            imgs = imgs.div(255).cpu().detach()
                        gen_imgs.append(batch_imgs)
                        real_imgs.append(imgs)
                        cov_labels.append(l[:, cov].reshape(-1,1))
                    gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                    real_imgs = torch.cat(real_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                    cov_labels = torch.cat(cov_labels, dim=0).to(device)
                    ### within strata, calculate mae, mse
                    gen_mse, gen_mae, gen_corr, predict_df = calc_mean_scores_disc(gen_imgs, real_imgs, 
                                                                                   cov_labels, cov, 
                                                                                   regr_model, Disc,
                                                                                   batch_size=64)
                    print(f"strata: {cur_c1}, {cur_c2}, {cur_c3}, MSE: {gen_mse}, MAE: {gen_mae}, CORR: {gen_corr}")
                    ### save the evaluation analysis to a json file
                    scores.append(np.array([num_real, cur_c1[0], cur_c1[1], cur_c2[0], cur_c2[1],
                        cur_c3[0], cur_c3[1], gen_mse, gen_mae, gen_corr]))
                    strata_predictions.append(predict_df)
    scores = np.stack(scores, axis=0)
    return scores, strata_predictions
# --------------------------------------------------------------------------------------
def run_stratified_mse_discriminator(opts):
    dataset = opts.dataset
    assert dataset == "ukb" or dataset == "retinal" or dataset == "mnist-thickness-intensity-slant"
    num_bins = 3
    ## config
    network_pkl = opts.network_pkl
    metric_jsonl = opts.metric_jsonl
    regr_model = opts.regr_model
    cov = opts.cov
    data_path1 = opts.data_path1
    data_path2 = opts.data_path2
    source_gan = opts.source_gan
    num_samples = opts.num_samples
    truncation_psi = opts.truncation_psi
    noise_mode = opts.noise_mode
    translate = opts.translate
    rotate = opts.rotate
    outdir = opts.outdir

    ds1, ds2 = load_dataset(dataset, data_path1, data_path2)
    labels1 = ds1._load_raw_labels() ## (c1, c2, c3) ## ground truth, c1 fixed
    labels2 = ds2._load_raw_labels() ## (c1, c2, c3) ## ground truth, c1 fixed

    labels_all = np.concatenate([labels1, labels2], axis=0) ## (c1, c2, c3)
    c1_all, c2_all, c3_all = labels_all[:,0], labels_all[:,1], labels_all[:,2]
    c1_min, c1_max = np.min(c1_all), np.max(c1_all)
    c2_min, c2_max = np.min(c2_all), np.max(c2_all)
    c3_min, c3_max = np.min(c3_all), np.max(c3_all)
    c1_hist = [np.quantile(c1_all, 1/num_bins), np.quantile(c1_all, 2/num_bins)]
    c2_hist = [np.quantile(c2_all, 1/num_bins), np.quantile(c2_all, 2/num_bins)]
    c3_hist = [np.quantile(c3_all, 1/num_bins), np.quantile(c3_all, 2/num_bins)]
    strata_hist = {"c1": c1_hist, "c2": c2_hist, "c3": c3_hist} ## define strata

    ### cov
    if cov in ["thickness", "age"]:
        cov_idx = 0
    elif cov in ["intensity", "ventricle", "diastolic"]:
        cov_idx = 1
    elif cov in ["slant", "grey_matter", "spherical"]:
        cov_idx = 2
    else:
        raise ValueError(f"covariate {cov} not found")

    covariates_info = dict(
        c1_min = c1_min, c1_max = c1_max,
        c2_min = c2_min, c2_max = c2_max,
        c3_min = c3_min, c3_max = c3_max,
        strata_hist = strata_hist,
        cov = {cov: cov_idx}
    )
    strata_distance = 1 ## distance between strata
    strata_idxs = [i for i in np.arange(0, num_bins, strata_distance)]
    # Load the network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_gen = 64
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    Disc = load_discriminator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    regr_model = load_regression_model(regr_model).to(device)
    ### within strata (c1, c2, c3), calculate MSE, MAE
    scores, predictions_df = calculate_loss(
        data_name = dataset,
        strata_idxs=strata_idxs,
        covariates=covariates_info,
        labels1=labels1,
        labels2=labels2,
        dataset1=ds1,
        dataset2=ds2,
        source_gan=source_gan,
        Gen=Gen,
        Disc=Disc,
        regr_model=regr_model,
        num_samples=num_samples,
        batch_size=batch_gen,
        device=device,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        translate=translate,
        rotate=rotate
    )
    scores_df = pd.DataFrame(scores, columns=["num_samples",
                                            "c1_min", "c1_max", 
                                            "c2_min", "c2_max", 
                                            "c3_min", "c3_max", 
                                            "gen_mse", "gen_mae", "gen_corr"])
    scores_df.to_csv(os.path.join(outdir, f"stratified_loss_{cov}_disc.csv"), index=False)
    for i in range(len(predictions_df)):
        predictions_df[i].to_csv(os.path.join(outdir, f"stratified_predictions_{cov}_stra{i}_disc.csv"),
                                 index=False)

## --------------------------- ##
@click.command()
@click.option('--network_specific', 'network_pkl', help='Network pickle filepath', default=None, required=False)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None, required=False)
@click.option('--regr_model', 'regr_model', help='Regression model', type=str, required=True)
@click.option('--cov', 'cov', help='Covariate', type=str, required=True)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb',
                                                         'retinal', None]),
              default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=True)
@click.option('--source-gan', 'source_gan', type=click.Choice(["source1", "source2", "multi"]), help='which source of GAN', default="multi", required=True)
@click.option('--num-samples', 'num_samples', type=int, help='Number of samples to generate', default=10000, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), 
              default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
              default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
              show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    assert opts.network_pkl is not None or opts.metric_jsonl is not None
    assert opts.regr_model is not None

    os.makedirs(opts.outdir, exist_ok=True)
    config_dict = {
        "gen_specific": opts.network_pkl,
        "gen": opts.metric_jsonl,
        "regr_model": opts.regr_model,
        "covariate": opts.cov,
        "dataset": opts.dataset, 
        "data_path1": opts.data_path1, "data_path2": opts.data_path2,
        "num_samples": opts.num_samples, "out_dir": opts.outdir}
    with open(os.path.join(opts.outdir, "strata_mse_config.json"), "w") as f:
        json.dump(config_dict, f)

    run_stratified_mse_discriminator(opts)
if __name__ == "__main__":
    main()