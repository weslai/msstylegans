from typing import Tuple, Union
import os, sys
import torch
import numpy as np
import pandas as pd
import json
import click
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans")
### -------------------
### --- Own ---
### -------------------
from eval_utils import calc_lpips_score, calc_psnr_score, calc_ssim_score
from utils import load_generator, generate_images
from eval_dataset import ConcatDataset
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
@click.option('--network_specific', 'network_pkl', help='Network pickle filepath', default=None, required=False)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None, required=False)
@click.option('--metric', 'metric', type=click.Choice(['psnr', 'ssim', 'lpips']), default='lpips', show_default=True)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb', 'retinal', None]), 
              default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', default=None, required=False)
@click.option('--source-gan', 'source_gan', type=click.Choice(["single", "multi"]), help='which source of GAN', default="multi", required=True)
@click.option('--which-source', 'which_source', type=click.Choice(["first", "second"]), help='which source of single source datasets', default="first", required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']),
              default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
              default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
              show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
### define strata with histogram
## get datasets
def run_general_metric(
    network_pkl: str,
    metric_jsonl: str,
    metric: str,
    dataset: str,
    data_path1: str,
    data_path2: str,
    source_gan: str,
    which_source: str,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float,float],
    rotate: float,
    outdir: str
):
    if network_pkl is not None:
        assert metric_jsonl is None
    else:
        assert metric_jsonl is not None
    if source_gan == "multi":
        assert data_path2 is not None
    os.makedirs(outdir, exist_ok=True)
    config_dict = {
        "gen_specific": network_pkl, "gen": metric_jsonl, "metric": metric,
        "dataset": dataset, "data_path1": data_path1, "data_path2": data_path2,
        "source_gan": source_gan, "which_source": which_source,
        "truncation_psi": truncation_psi, "noise_mode": noise_mode,
        "out_dir": outdir}
    with open(os.path.join(outdir, "generalfid_config.json"), "w") as f:
        json.dump(config_dict, f)
    if dataset == "mnist-thickness-intensity-slant":
        if data_path2 is not None and source_gan == "multi":
            dataset2 = "mnist-thickness-intensity-slant"
            ## seed is as the common convariate (c1)
            ds1 = MorphoMNISTDataset_causal(data_name=dataset,
                                            path=data_path1,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False,
                                            include_numbers=True)
            ds2 = MorphoMNISTDataset_causal(data_name=dataset2,
                                            path=data_path2,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False,
                                            include_numbers=True)
            concat_ds = ConcatDataset(ds1, ds2)
            labels = ds1._load_raw_labels()
            labels2 = ds2._load_raw_labels()
        elif data_path2 is not None and source_gan == "single":
            dataset2 = "mnist-thickness-intensity-slant"
            ## seed is as the common convariate (c1)
            ds1 = MorphoMNISTDataset_causal_single(data_name=dataset,
                                            path=data_path1,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False,
                                            include_numbers=True)
            ds2 = MorphoMNISTDataset_causal_single(data_name=dataset2,
                                            path=data_path2,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False,
                                            include_numbers=True)
            concat_ds = ConcatDataset(ds1, ds2)
            labels = ds1._load_raw_labels()
            labels2 = ds2._load_raw_labels()
        else:
            ds1 = MorphoMNISTDataset_causal_single(data_name=dataset,
                                            path=data_path1,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False,
                                            include_numbers=True)
            labels = ds1._load_raw_labels()
    elif dataset == "ukb":
        if data_path2 is not None and source_gan == "multi":
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
            concat_ds = ConcatDataset(ds1, ds2)
            labels = ds1._load_raw_labels()
            labels2 = ds2._load_raw_labels()
        elif data_path2 is not None and source_gan == "single":
            ds1 = UKBiobankMRIDataset2D_single(data_name=dataset,
                                        path=data_path1, 
                                        mode="test", 
                                        use_labels=True,
                                        xflip=False)
            ds2 = UKBiobankMRIDataset2D_single(data_name=dataset, 
                                        path=data_path2, 
                                        mode="test", 
                                        use_labels=True,
                                        xflip=False)
            concat_ds = ConcatDataset(ds1, ds2)
            labels = ds1._load_raw_labels()
            labels2 = ds2._load_raw_labels()
        else:
            ds1 = UKBiobankMRIDataset2D_single(data_name=dataset,
                                                path=data_path1, 
                                                mode="test", 
                                                use_labels=True,
                                                xflip=False)
            labels = ds1._load_raw_labels()
    elif dataset == "retinal":
        if data_path2 is not None and source_gan == "multi":
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
            concat_ds = ConcatDataset(ds1, ds2)
            labels = ds1._load_raw_labels()
            labels2 = ds2._load_raw_labels()
        elif data_path2 is not None and source_gan == "single":
            ds1 = UKBiobankRetinalDataset2D_single(data_name=dataset,
                                            path=data_path1,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False)
            ds2 = UKBiobankRetinalDataset2D_single(data_name=dataset,
                                            path=data_path2,
                                            mode="test",
                                            use_labels=True,
                                            xflip=False)
            concat_ds = ConcatDataset(ds1, ds2)
            labels = ds1._load_raw_labels()
            labels2 = ds2._load_raw_labels()
        else:
            ds1 = UKBiobankRetinalDataset2D_single(data_name=dataset,
                                        path=data_path1, 
                                        mode="test", 
                                        use_labels=True,
                                        xflip=False)
            labels = ds1._load_raw_labels()
    else:
        raise NotImplementedError
    # Load the network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_gen = 4
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    real_imgs = []
    gen_imgs = []
    labels = []
    ### get the real images
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    if data_path2 is not None: ## either multi (test on two datasets) but single source GAN or with multi-source GAN
        ## real images
        for source1, source2 in torch.utils.data.DataLoader(concat_ds, batch_size=batch_gen, shuffle=True, **data_loader_kwargs):
            img1, _labels1 = source1[0], source1[1]
            img2, _labels2 = source2[0], source2[1]
            for img, label in zip([img1, img2], [_labels1, _labels2]):
                if img.shape[1] != 1:
                    img = torch.tensor(img)
                real_imgs.append(img)
                labels.append(label.cpu().detach().numpy())
        labels = np.concatenate(labels, axis=0)
        ## fake images
        if source_gan == "multi":
            for num, l in enumerate(labels):
                ## (c1, c2, c3)
                if num // 2 == 0:
                    label1_norm = ds1._normalise_labels(l[0].reshape(-1, 1), l[1].reshape(-1, 1),
                                                        l[2].reshape(-1, 1))
                    if dataset == "mnist-thickness-intensity-slant":
                        label1_norm = np.concatenate([label1_norm, l[3:].reshape(1, -1)], axis=-1)
                    ll = torch.from_numpy(label1_norm).to(device)
                else:
                    label2_norm = ds2._normalise_labels(l[0].reshape(-1, 1), l[1].reshape(-1, 1),
                                                        l[2].reshape(-1, 1))
                    if dataset == "mnist-thickness-intensity-slant":
                        label2_norm = np.concatenate([label2_norm, l[3:].reshape(1, -1)], axis=-1)
                
                    ll = torch.from_numpy(label2_norm).to(device)
                z = torch.randn(1, Gen.z_dim).to(device)
                batch_imgs = generate_images(Gen, z, ll, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
                gen_imgs.append(batch_imgs)
        else: ## single
            ## single ## (c1, c2)
            for num, l in enumerate(labels):
                if which_source == "first":
                    label1_norm = ds1._normalise_labels(l[0].reshape(-1, 1), l[1].reshape(-1, 1))
                else: ## second
                    if dataset == "ukb":
                        label1_norm = ds1._normalise_labels(l[0].reshape(-1, 1), grey_matter=l[1].reshape(-1, 1))
                    elif dataset == "retinal":
                        label1_norm = ds1._normalise_labels(l[0].reshape(-1, 1), spherical_power_left=l[1].reshape(-1, 1))
                    else:
                        label1_norm = ds1._normalise_labels(l[0].reshape(-1, 1), slant=l[1].reshape(-1, 1))
                if dataset == "mnist-thickness-intensity-slant":
                    label1_norm = np.concatenate([label1_norm, l[2:].reshape(1, -1)], axis=-1)
                ll = torch.from_numpy(label1_norm).to(device)
                z = torch.randn(1, Gen.z_dim).to(device)
                batch_imgs = generate_images(Gen, z, ll, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
                gen_imgs.append(batch_imgs)
    
    if dataset != "retinal":
        if metric == "lpips":
            real_imgs = torch.cat(real_imgs, dim=0).repeat([1, 3, 1, 1]).to(device) / 255 ## (batch_size, channel (3), pixel, pixel)
            gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1, 3, 1, 1]).to(device) / 255 ## (batch_size, channel (3), pixel, pixel)
        else:
            real_imgs = torch.cat(real_imgs, dim=0).to(device) / 255 if metric == "ssim" else torch.cat(real_imgs, dim=0).to(device)
            gen_imgs = torch.cat(gen_imgs, dim=0).to(device) / 255 if metric == "ssim" else torch.cat(gen_imgs, dim=0).to(device)
    else:
        if metric in ["lpips", "ssim"]:
            real_imgs = torch.cat(real_imgs, dim=0).to(device) / 255
            gen_imgs = torch.cat(gen_imgs, dim=0).to(device) / 255
        else:
            real_imgs = torch.cat(real_imgs, dim=0).to(device)
            gen_imgs = torch.cat(gen_imgs, dim=0).to(device)
    print(f"Real images: {real_imgs.shape}")
    print(f"Generated images: {gen_imgs.shape}")
    ### calculate Inception
    if metric == "lpips":
        lpips_score = calc_lpips_score(gen_imgs, real_imgs, batch_size=128)
        lpips_mean = lpips_score[0]
        lpips_std = lpips_score[1]
        print(f"IPIPS: {lpips_mean} +- {lpips_std}")
    elif metric == "psnr":
        psnr_score = calc_psnr_score(gen_imgs, real_imgs, batch_size=128)
        psnr_mean = psnr_score[0]
        psnr_std = psnr_score[1]
        print(f"PSNR: {psnr_mean} +- {psnr_std}")
    elif metric == "ssim":
        ssim_score = calc_ssim_score(gen_imgs, real_imgs, batch_size=128)
        ssim_mean = ssim_score[0]
        ssim_std = ssim_score[1]
        print(f"SSIM: {ssim_mean} +- {ssim_std}")
    ### save the evaluation analysis to a json file
    result = dict(
            mean=lpips_mean if metric == "lpips" else psnr_mean if metric == "psnr" else ssim_mean,
            std=lpips_std if metric == "lpips" else psnr_std if metric == "psnr" else ssim_std,
            metric=metric,
            dataset=dataset,
            network=metric_jsonl,
            data_path1=data_path1,
            data_path2=data_path2 if data_path2 is not None else "None"
    )
    result_df = pd.DataFrame.from_dict(result, orient="index").T
    if data_path2 is not None and source_gan == "multi":
        result_df.to_csv(os.path.join(outdir, f"ms_general_{metric}.csv"), index=False)
    elif data_path2 is not None and source_gan == "single":
        result_df.to_csv(os.path.join(outdir, f"single_with_doubledists_general_{metric}.csv"), index=False)
    else:
        result_df.to_csv(os.path.join(outdir, f"general_{metric}.csv"), index=False)

if __name__ == "__main__":
    run_general_metric()