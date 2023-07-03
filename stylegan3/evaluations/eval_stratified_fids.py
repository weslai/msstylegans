from typing import Tuple, Union
import os
import torch
import numpy as np
import pandas as pd
import json
import click
### -------------------
### --- Own ---
### -------------------
from eval_utils import calc_fid_score
from utils import load_generator, generate_images
from eval_dataset import MorphoMNISTDataset_causal, MorphoMNISTDataset_causal_single
from eval_dataset import UKBiobankMRIDataset2D, UKBiobankMRIDataset2D_single
from eval_dataset import UKBiobankRetinalDataset2D, UKBiobankRetinalDataset2D_single

# --------------------------------------------------------------------------------------
def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')
# --------------------------------------------------------------------------------------

@click.command()
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', required=True)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb',
                                                         'retinal', None]),
              default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=True)
@click.option('--source-gan', 'source_gan', type=click.Choice(["single", "multi"]), help='which source of GAN', default="multi", required=True)
@click.option('--num-samples', 'num_samples', type=int, help='Number of samples to generate', default=10000, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.8, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), 
              default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
              default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
              show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
### define strata with histogram
## get datasets
def run_stratified_fid(
    metric_jsonl: str,
    dataset: str,
    data_path1: str,
    data_path2: str,
    source_gan: str,
    num_samples: int,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float,float],
    rotate: float,
    outdir: str
):
    os.makedirs(outdir, exist_ok=True)
    config_dict = {"gen": metric_jsonl, "dataset": dataset, "data_path1": data_path1, "data_path2": data_path2,
                    "num_samples": num_samples, "out_dir": outdir}
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config_dict, f)
    assert dataset == "ukb" or dataset == "retinal" or dataset == "mnist-thickness-intensity-slant"
    num_bins = 3
    if dataset == "ukb":
        ds1 = UKBiobankMRIDataset2D(data_name=dataset, 
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds2 = UKBiobankMRIDataset2D(data_name=dataset, 
                                    path=data_path2, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds_gen = UKBiobankMRIDataset2D_single(data_name=dataset,
                                    path=data_path1,
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
    elif dataset == "retinal":
        ds1 = UKBiobankRetinalDataset2D(data_name=dataset, 
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds2 = UKBiobankRetinalDataset2D(data_name=dataset, 
                                    path=data_path2, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds_gen = UKBiobankRetinalDataset2D_single(data_name=dataset,
                                    path=data_path1,
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
    elif dataset == "mnist-thickness-intensity-slant":
        ds1 = MorphoMNISTDataset_causal(data_name=dataset, 
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds2 = MorphoMNISTDataset_causal(data_name=dataset, 
                                    path=data_path2, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds_gen = MorphoMNISTDataset_causal_single(data_name=dataset,
                                    path=data_path1,
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
    labels1 = ds1._load_raw_labels() ## (c1, c2, c3) ## ground truth, c1 fixed
    labels2 = ds2._load_raw_labels() ## (c1, c2, c3) ## ground truth, c1 fixed
    label_gen = ds_gen._load_raw_labels() ## (c1, c2) ## only c1 and c2

    labels_all = np.concatenate([labels1, labels2], axis=0) ## (c1, c2, c3)
    c1_all, c2_all, c3_all = labels_all[:,0], labels_all[:,1], labels_all[:,2]
    c1_hist = np.histogram(c1_all, bins=num_bins)
    c2_hist = np.histogram(c2_all, bins=num_bins)
    c3_hist = np.histogram(c3_all, bins=num_bins)
    strata_hist = {"c1": c1_hist, "c2": c2_hist, "c3": c3_hist} ## define strata

    strata_distance = 1 ## distance between strata
    strata_idxs = [i for i in np.arange(0, num_bins, strata_distance)]
    # Load the network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_gen = 4
    Gen = load_generator(
        network_pkl=None,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    ### within strata (c1, c2, c3), calculate FID
    scores = []
    for stra_c1 in strata_idxs:
        cur_c1 = (strata_hist["c1"][1][stra_c1], strata_hist["c1"][1][stra_c1+1])
        for stra_c2 in strata_idxs:
            cur_c2 = (strata_hist["c2"][1][stra_c2], strata_hist["c2"][1][stra_c2+1])
            for stra_c3 in strata_idxs:
                cur_c3 = (strata_hist["c3"][1][stra_c3], strata_hist["c3"][1][stra_c3+1])
                real_imgs = []
                gen_imgs = []
                ## get samples from datasets
                idxs1 = np.where((labels1[:,0] >= cur_c1[0]) & (labels1[:,0] < cur_c1[1]) & \
                                (labels1[:,1] >= cur_c2[0]) & (labels1[:,1] < cur_c2[1]) & \
                                (labels1[:,2] >= cur_c3[0]) & (labels1[:,2] < cur_c3[1]))[0]
                idxs2 = np.where((labels2[:,0] >= cur_c1[0]) & (labels2[:,0] < cur_c1[1]) & \
                                (labels2[:,1] >= cur_c2[0]) & (labels2[:,1] < cur_c2[1]) & \
                                (labels2[:,2] >= cur_c3[0]) & (labels2[:,2] < cur_c3[1]))[0]
                if len(idxs1) + len(idxs2) > 10:
                    for idx in idxs1:
                        real_imgs.append(torch.tensor(ds1[idx][0]))
                    for idx in idxs2:
                        real_imgs.append(torch.tensor(ds2[idx][0]))
                    if dataset in ["ukb", "mnist-thickness-intensity-slant"]:
                        real_imgs = torch.stack(real_imgs, dim=0).repeat([1,3,1,1]).to(device) ## (batch_size, channel (3), pixel, pixel)
                    elif dataset == "retinal":
                        real_imgs = torch.stack(real_imgs, dim=0).repeat([1,1,1,1]).to(device)
                    ## get samples from GANs
                    for _ in range(num_samples // batch_gen):
                        z = torch.randn(batch_gen, Gen.z_dim).to(device)
                        if source_gan == "multi":
                            source1_c = [ds1.get_norm_label(idx) for idx in np.random.choice(idxs1, batch_gen//2)]
                            source2_c = [ds2.get_norm_label(idx) for idx in np.random.choice(idxs2, batch_gen//2)]
                            all_c = source1_c + source2_c
                        else: ## single source
                            all_c = [ds_gen.get_norm_label(idx) for idx in np.random.choice(idxs1, batch_gen)]
                        l = torch.from_numpy(np.stack(all_c, axis=0)).to(device)
                        ### Randomized latent codes
                        # c1 = torch.tensor(np.random.uniform(cur_c1[0], cur_c1[1], size=(batch_gen, 1)))
                        # c1 = (c1 - ds1.model["age_mu"]) / ds1.model["age_std"]
                        # c2 = torch.tensor(np.random.uniform(cur_c2[0], cur_c2[1], size=(batch_gen, 1)))
                        # c2 = (c2 - ds1.model["brain_mu"]) / ds1.model["brain_std"]
                        # c3 = torch.tensor(np.random.uniform(cur_c3[0], cur_c3[1], size=(batch_gen, 1)))
                        # c3 = (c3 - ds2.model["ventricle_mu"]) / ds2.model["ventricle_std"]
                        # l = torch.cat([c1, c2, c3], dim=1).to(device)
                        batch_imgs = generate_images(Gen, z, l, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
                        gen_imgs.append(batch_imgs)
                    if dataset in ["ukb", "mnist-thickness-intensity-slant"]:
                        gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,3,1,1]).to(device)## (batch_size, channel (3), pixel, pixel)
                    elif dataset == "retinal":
                        gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,1,1,1]).to(device)
                    ### within strata, calculate FID
                    fid_score = calc_fid_score(real_imgs, gen_imgs, batch_size=64)
                    print(f"strata: {cur_c1}, {cur_c2}, {cur_c3}, FID: {fid_score}")
                    ### save the evaluation analysis to a json file
                    scores.append(np.array([cur_c1[0], cur_c1[1], cur_c2[0], cur_c2[1],
                                cur_c3[0], cur_c3[1], fid_score.cpu().detach().numpy()]))
    scores = np.stack(scores, axis=0)
    scores_df = pd.DataFrame(scores, columns=["c1_min", "c1_max", "c2_min", "c2_max", "c3_min", "c3_max", "fid_score"])
    scores_df.to_csv(os.path.join(outdir, "ms_stratified_fid.csv"), index=False)

if __name__ == "__main__":
    run_stratified_fid()