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
from eval_utils import calc_inception_score
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
@click.option('--test_on', help='Test on the real images or the generated images', type=click.Choice(['real', 'gen']), 
              default='real', show_default=True)
@click.option('--network_specific', 'network_pkl', help='Network pickle filepath', default=None, required=False)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None, required=False)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb', 'retinal', None]), 
              default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', default=None, required=False)
@click.option('--source-gan', 'source_gan', type=click.Choice(["single", "multi"]), help='which source of GAN', default="multi", required=True)
@click.option('--num-samples', 'num_samples', type=int, help='Number of samples to generate', default=10000, show_default=True)
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
    test_on: str,
    network_pkl: str,
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
    if network_pkl is not None:
        assert metric_jsonl is None
    else:
        assert metric_jsonl is not None
    if source_gan == "multi":
        assert data_path2 is not None
    os.makedirs(outdir, exist_ok=True)
    config_dict = {
        "test_on": test_on, "gen_specific": network_pkl, "gen": metric_jsonl,
        "dataset": dataset, "data_path1": data_path1, "data_path2": data_path2,
        "source_gan": source_gan, "num_samples": num_samples,
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
    if test_on == "gen":
        Gen = load_generator(
            network_pkl=network_pkl,
            metric_jsonl=metric_jsonl,
            use_cuda=True
        )
        gen_imgs = []
    real_imgs = []
    ### get the real images
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    if data_path2 is not None: ## either multi (test on two datasets) but single source GAN or with multi-source GAN
        ## real images
        for source1, source2 in torch.utils.data.DataLoader(concat_ds, batch_size=batch_gen, shuffle=True, **data_loader_kwargs):
            img1, _labels1 = source1[0], source1[1]
            img2, _labels2 = source2[0], source2[1]
            for img in [img1, img2]:
                if img.shape[1] != 1:
                    img = torch.tensor(img)
                real_imgs.append(img)
        ## fake images
        if test_on == "gen":
            if source_gan == "multi":
                for _ in range(num_samples // (batch_gen * 2)):
                    batch_labels = []
                    z = torch.randn(batch_gen * 2, Gen.z_dim).to(device)
                    for _i in range(batch_gen):
                        ## (c1, c2, c3)
                        source_c1, source_c2 = concat_ds[np.random.randint(len(concat_ds))]
                        label1, label2 = source_c1[1], source_c2[1]
                        label1_norm = ds1._normalise_labels(label1[0].reshape(-1, 1), label1[1].reshape(-1, 1),
                                                            label1[2].reshape(-1, 1))
                        label2_norm = ds2._normalise_labels(label2[0].reshape(-1, 1), label2[1].reshape(-1, 1),
                                                            label2[2].reshape(-1, 1))
                        if dataset == "mnist-thickness-intensity-slant":
                            label1_norm = np.concatenate([label1_norm, label1[3:].reshape(1, -1)], axis=-1)
                            label2_norm = np.concatenate([label2_norm, label2[3:].reshape(1, -1)], axis=-1)
                        label1_norm = torch.from_numpy(label1_norm).to(device)
                        label2_norm = torch.from_numpy(label2_norm).to(device)
                        l = torch.cat([label1_norm, label2_norm], dim=0)
                        batch_labels.append(l)
                    ll = torch.cat(batch_labels, dim=0).to(device)
                    batch_imgs = generate_images(Gen, z, ll, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
                    gen_imgs.append(batch_imgs)
            else: ## single
                ## single ## (c1, c2)
                for _ in range(num_samples // batch_gen):
                    z = torch.randn(batch_gen, Gen.z_dim).to(device)
                    l = [ds1.get_norm_label(np.random.randint(len(ds1))) for _ in range(batch_gen)]
                    l = torch.from_numpy(np.stack(l, axis=0)).to(device)
                    batch_imgs = generate_images(Gen, z, l, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
                    gen_imgs.append(batch_imgs)
    else: ## single
        for imgs, _labels in torch.utils.data.DataLoader(ds1, batch_size=batch_gen, shuffle=True, **data_loader_kwargs):
            if imgs.shape[1] == 1:
                imgs = torch.tensor(imgs)
            real_imgs.append(imgs)
        ## fake images
        if test_on == "gen":
            for _ in range(num_samples // batch_gen):
                z = torch.randn(batch_gen, Gen.z_dim).to(device)
                c = [ds1.get_norm_label(np.random.randint(len(ds1))) for _ in range(batch_gen)]
                c = torch.from_numpy(np.stack(c, axis=0)).to(device)
                batch_imgs = generate_images(Gen, z, c, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
                gen_imgs.append(batch_imgs)
    if dataset != "retinal":
        real_imgs = torch.cat(real_imgs, dim=0).repeat([1, 3, 1, 1]).to(device) ## (batch_size, channel (3), pixel, pixel)
        if test_on == "gen":
            gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1, 3, 1, 1]).to(device) ## (batch_size, channel (3), pixel, pixel)
    else:
        real_imgs = torch.cat(real_imgs, dim=0).to(device)
        if test_on == "gen":
            gen_imgs = torch.cat(gen_imgs, dim=0).to(device)
    print(f"Real images: {real_imgs.shape}")
    if test_on == "gen":
        print(f"Generated images: {gen_imgs.shape}")
    ### calculate Inception
    inception_score = calc_inception_score(real_imgs if test_on == "real" else gen_imgs, batch_size=64)
    inception_mean = inception_score[0].cpu().detach().numpy()
    inception_std = inception_score[1].cpu().detach().numpy()
    print(f"IC: {inception_mean} +- {inception_std} for {num_samples} samples")
    ### save the evaluation analysis to a json file
    result = dict(inception_mean=inception_mean,
            inception_std = inception_std,
            test_on = test_on,
            num_samples=num_samples,
            dataset=dataset,
            network=metric_jsonl,
            data_path1=data_path1,
            data_path2=data_path2 if data_path2 is not None else "None"
    )
    result_df = pd.DataFrame.from_dict(result, orient="index").T
    if test_on == "real":
        result_df.to_csv(os.path.join(outdir, "real_general_inception.csv"), index=False)
    else:
        if data_path2 is not None and source_gan == "multi":
            result_df.to_csv(os.path.join(outdir, "ms_general_inception.csv"), index=False)
        elif data_path2 is not None and source_gan == "single":
            result_df.to_csv(os.path.join(outdir, "single_with_doubledists_general_inception.csv"), index=False)
        else:
            result_df.to_csv(os.path.join(outdir, "general_inception.csv"), index=False)

if __name__ == "__main__":
    run_general_metric()