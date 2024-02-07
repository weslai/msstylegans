from typing import Tuple, Union
import os, sys
import torch
import numpy as np
import pandas as pd
import json
import click
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans")
import dnnlib
### -------------------
### --- Own ---
### -------------------
from eval_utils import calc_fid_score, calc_kid_score
from utils import load_generator, generate_images
from latent_mle_real_ms import SourceSampling
from eval_general_mse_real_ms import load_dataset
from eval_dataset import ConcatDataset

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
def run_general_fid(opts):
    ## config
    dataset = opts.dataset 
    network_pkl = opts.network_pkl
    metric_jsonl = opts.metric_jsonl
    metric = opts.metric
    data_source = opts.data_source
    data_path1 = opts.data_path1
    data_path2 = opts.data_path2
    data_path3 = opts.data_path3
    source_gan = opts.source_gan
    num_samples = opts.num_samples
    truncation_psi = opts.truncation_psi
    noise_mode = opts.noise_mode
    translate = opts.translate
    rotate = opts.rotate
    outdir = opts.outdir

    ds = load_dataset(dataset, source_gan=data_source, data_path1=data_path1, data_path2=data_path2, data_path3=data_path3)
    target_dataset0, target_dataset1 = ds[0], ds[1]
    target_dataset0._load_raw_labels()
    target_dataset1._load_raw_labels()
    dataset1 = ds[0]
    dataset2 = ds[1] if type(ds) == tuple else None
    dataset3 = None
    concat_ds = ConcatDataset(target_dataset0, target_dataset1)
    # if source_gan == "single":
    #     target_dataset = ds
    #     target_dataset._load_raw_labels()
    # else:
    #     target_dataset = ds[0] if dataset in ["ukb", "retinal"] else ds[1] if dataset in ["adni", "rfmid"] else ds[2]
    #     target_dataset._load_raw_labels()
    #     dataset1 = ds[0]
    #     dataset2 = ds[1] if type(ds) == tuple else None
    #     dataset3 = ds[2] if opts.data_path3 is not None else None
    
    ## load latent models
    sampler_sources = {}
    if source_gan == "multi_mri":
        sampler_source1 = SourceSampling("ukb", label_path=os.path.join(data_path1, "trainset"))
        sampler_source1.get_graph()
        sampler_source1.get_causal_model()
        sampler_source2 = SourceSampling("adni", label_path=os.path.join(data_path2, "trainset"))
        sampler_source2.get_graph()
        sampler_source2.get_causal_model()
        sampler_sources["ukb"] = sampler_source1
        sampler_sources["adni"] = sampler_source2
        if data_path3 is not None:
            sampler_source3 = SourceSampling("nacc", label_path=os.path.join(data_path3, "trainset"))
            sampler_source3.get_graph()
            sampler_source3.get_causal_model()
            sampler_sources["nacc"] = sampler_source3
    elif source_gan == "multi_retina":
        sampler_source1 = SourceSampling("retinal", label_path=os.path.join(data_path1, "trainset"))
        sampler_source1.get_graph()
        sampler_source1.get_causal_model()
        sampler_source2 = SourceSampling("rfmid", label_path=os.path.join(data_path2, "trainset"))
        sampler_source2.get_graph()
        sampler_source2.get_causal_model()
        sampler_sources["retinal"] = sampler_source1
        sampler_sources["rfmid"] = sampler_source2
        if data_path3 is not None:
            sampler_source3 = SourceSampling("eyepacs", label_path=os.path.join(data_path3, "trainset"))
            sampler_source3.get_graph()
            sampler_source3.get_causal_model()
            sampler_sources["eyepacs"] = sampler_source3
    else:
        if source_gan != "single":
            raise ValueError(f"source gan {source_gan} not found")
    cov_dict = {}
    if source_gan != "single":
        if dataset == "ukb" or dataset == "adni" or dataset == "nacc":
            if dataset3 is not None:
                dataset1._load_raw_labels()
                dataset2._load_raw_labels()
                dataset3._load_raw_labels()
                cov_dict["age_max"] = max(dataset1.model["age_max"], dataset2.model["age_max"], 
                                        dataset3.model["age_max"])
                cov_dict["age_min"] = min(dataset1.model["age_min"], dataset2.model["age_min"], 
                                        dataset3.model["age_min"])
            else:
                dataset1._load_raw_labels()
                dataset2._load_raw_labels()
                cov_dict["age_max"] = max(dataset1.model["age_max"], dataset2.model["age_max"])
                cov_dict["age_min"] = min(dataset1.model["age_min"], dataset2.model["age_min"])
    
    # Load the network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_gen = 4
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    ### get the real images
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    real_imgs = []
    gen_imgs = []
    ## real images
    for source in torch.utils.data.DataLoader(concat_ds, batch_size=batch_gen, shuffle=True, **data_loader_kwargs):
        img0, img1 = source[0][0], source[1][0]
        # img, _labels = source[0], source[1]
        if img0.shape[1] != 1:
            img0 = torch.tensor(img0)
            img1 = torch.tensor(img1)
        real_imgs.append(img0)
        real_imgs.append(img1)
    ## fake images
    for _ in range(num_samples // batch_gen):
        batch_labels = []
        if source_gan != "single":
            batch_labels1 = []
        z = torch.randn(batch_gen, Gen.z_dim).to(device)
        if source_gan == "single":
            for idx in np.random.choice(range(len(concat_ds)), batch_gen):
                if dataset in ["ukb", "retinal"]:
                    batch_labels.append(concat_ds.datasets[0].get_norm_label(idx))
                elif dataset in ["adni", "rfmid"]:
                    batch_labels.append(concat_ds.datasets[1].get_norm_label(idx))
        else:
            for idx in np.random.choice(range(len(concat_ds)), batch_gen//2):
                batch_labels.append(concat_ds.datasets[0].get_norm_label(idx))
                batch_labels1.append(concat_ds.datasets[1].get_norm_label(idx))
        l = torch.from_numpy(np.stack(batch_labels, axis=0)).to(device)
        if source_gan != "single":
            l1 = torch.from_numpy(np.stack(batch_labels1, axis=0)).to(device)
        
        if source_gan == "multi_mri":
            if len(sampler_sources) == 3: ## three sources
                if dataset == "ukb":
                    age = l[:, 0] * torch.tensor(dataset1.model["age_max"] - dataset1.model["age_min"]) + torch.tensor(dataset1.model["age_min"])
                    gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                    gen_c4 = sampler_sources["nacc"].sampling_given_age(age.cpu().detach(), normalize=True)
                    c_source = torch.tensor([1, 0, 0] * age.shape[0]).reshape(age.shape[0], -1)
                    age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                    all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), l[:, 1:].cpu().detach(), 
                                        gen_c3[:, 1:], gen_c4[:, 1:], c_source], dim=1).to(device)
                elif dataset == "adni":
                    age = l[:, 0] * torch.tensor(dataset2.model["age_max"] - dataset2.model["age_min"]) + torch.tensor(dataset2.model["age_min"])
                    gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                    gen_c4 = sampler_sources["nacc"].sampling_given_age(age.cpu().detach(), normalize=True)
                    c_source = torch.tensor([0, 1, 0] * age.shape[0]).reshape(age.shape[0], -1)
                    age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                    all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                        l[:, 1:].cpu().detach(), gen_c4[:, 1:], c_source], dim=1).to(device)
            elif len(sampler_sources) == 2: ## two sources
                # if dataset == "ukb":
                age = l[:, 0] * torch.tensor(dataset1.model["age_max"] - dataset1.model["age_min"]) + torch.tensor(dataset1.model["age_min"])
                gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                c_source = torch.tensor([0] * age.shape[0]).reshape(age.shape[0], -1)
                age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), l[:, 1:].cpu().detach(), 
                                    gen_c3[:, 1:], c_source], dim=1).to(device)
                # elif dataset == "adni":
                age = l1[:, 0] * torch.tensor(dataset2.model["age_max"] - dataset2.model["age_min"]) + torch.tensor(dataset2.model["age_min"])
                gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                c_source = torch.tensor([1] * age.shape[0]).reshape(age.shape[0], -1)
                age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                all_c1 = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                    l1[:, 1:].cpu().detach(), c_source], dim=1).to(device)
                all_c = torch.cat([all_c, all_c1], dim=0)
        elif source_gan == "multi_retina":
            if len(sampler_sources) == 3:
                if dataset == "retinal":
                    gen_c3 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                    gen_c4 = sampler_sources["eyepacs"].sampling_model(l.shape[0])[-1]
                    c_source = torch.tensor([1, 0, 0] * l.shape[0]).reshape(l.shape[0], -1)
                    all_c = torch.cat([l.cpu().detach(), gen_c3, gen_c4, c_source], dim=1).to(device)
                elif dataset == "rfmid":
                    gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                    gen_c4 = sampler_sources["eyepacs"].sampling_model(l.shape[0])[-1]
                    c_source = torch.tensor([0, 1, 0] * l.shape[0]).reshape(l.shape[0], -1)
                    all_c = torch.cat([gen_c2, l.cpu().detach(), gen_c4, c_source], dim=1).to(device)
                elif dataset == "eyepacs":
                    gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                    gen_c3 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                    c_source = torch.tensor([0, 0, 1] * l.shape[0]).reshape(l.shape[0], -1)
                    all_c = torch.cat([gen_c2, gen_c3, l.cpu().detach(), c_source], dim=1).to(device)
            elif len(sampler_sources) == 2:
                # if dataset == "retinal":
                gen_c3 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                c_source = torch.tensor([0] * l.shape[0]).reshape(l.shape[0], -1)
                all_c = torch.cat([l.cpu().detach(), gen_c3, c_source], dim=1).to(device)
                # elif dataset == "rfmid":
                gen_c2 = sampler_sources["retinal"].sample_normalize(l1.shape[0])
                c_source = torch.tensor([1] * l1.shape[0]).reshape(l1.shape[0], -1)
                all_c1 = torch.cat([gen_c2, l1.cpu().detach(), c_source], dim=1).to(device)
                all_c = torch.cat([all_c, all_c1], dim=0)
        batch_imgs = generate_images(Gen, z, l[:, :Gen.c_dim] if source_gan == "single" else all_c, 
                                    truncation_psi, 
                                    noise_mode, translate, rotate).permute(0,3,1,2)
        gen_imgs.append(batch_imgs)
    if dataset not in ["retinal", "rfmid", "eyepacs"]:
        real_imgs = torch.cat(real_imgs, dim=0).repeat([1, 3, 1, 1]).to(device) ## (batch_size, channel (3), pixel, pixel)
        gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,3,1,1]).to(device) ## (batch_size, channel (3), pixel, pixel)
    else:
        real_imgs = torch.cat(real_imgs, dim=0).to(device)
        gen_imgs = torch.cat(gen_imgs, dim=0).to(device)
    print(f"Real images: {real_imgs.shape}")
    print(f"Generated images: {gen_imgs.shape}")
    ### calculate evaluation metrics
    if metric == "fid":
        fid_score = calc_fid_score(real_imgs, gen_imgs, batch_size=64).cpu().detach().numpy()
        print(f"FID: {fid_score} for {num_samples} samples")
        ### save the evaluation analysis to a json file
        result = dict(
            fid=fid_score,
            num_samples=num_samples,
            dataset=dataset,
            network=metric_jsonl,
            data_path1=data_path1,
            data_path2=data_path2 if data_path2 is not None else "None",
            data_path3 = data_path3 if data_path3 is not None else "None"
        )
    elif metric == "kid":
        kids = calc_kid_score(real_imgs, gen_imgs, batch_size=64)
        kid_mean = kids[0].cpu().detach().numpy()
        kid_std = kids[1].cpu().detach().numpy()
        print(f"KID: {kid_mean} and {kid_std} for {num_samples} samples")
        ### save the evaluation analysis to a json file
        result = dict(kid_mean = kid_mean,
            kid_std = kid_std,
            num_samples=num_samples,
            dataset=dataset,
            network=metric_jsonl,
            data_path1=data_path1,
            data_path2=data_path2 if data_path2 is not None else "None",
            data_path3 = data_path3 if data_path3 is not None else "None"
        )
    result_df = pd.DataFrame.from_dict(result, orient="index").T
    result_df.to_csv(os.path.join(outdir, f"general_fid_{source_gan}_{dataset}.csv"), index=False)
# --------------------------------------------------------------------------------------
@click.command()
@click.option('--network_specific', 'network_pkl', help='Network pickle filepath', default=None, required=False)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None, required=False)
@click.option('--metric', 'metric', type=click.Choice(['fid', 'kid']), default='fid', show_default=True)
@click.option('--dataset', 'dataset', type=click.Choice(['ukb', 'retinal', 'adni', 'nacc', 'eyepacs', 'rfmid',
                                                         None]), default=None, show_default=True)
@click.option('--data-source', 'data_source', type=click.Choice(["multi_mri", "multi_retina", "single",
                                                            None]), default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=False)
@click.option('--data-path3', 'data_path3', type=str, help='Path to the data source 3', required=False)

@click.option('--source-gan', 'source_gan', type=click.Choice(["single", "multi_mri", "multi_retina"]), help='which source of GAN', default="single", required=True)
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

    os.makedirs(opts.outdir, exist_ok=True)
    config_dict = {
        "gen_specific": opts.network_pkl,
        "gen": opts.metric_jsonl,
        "metric": opts.metric,
        "dataset": opts.dataset,
        "data_source": opts.data_source,
        "data_path1": opts.data_path1, "data_path2": opts.data_path2,
        "data_path3": opts.data_path3, "source_gan": opts.source_gan,
        "num_samples": opts.num_samples, 
        "truncation_psi": opts.truncation_psi, "noise_mode": opts.noise_mode,
        "out_dir": opts.outdir}
    with open(os.path.join(opts.outdir, "general_fid_config.json"), "w") as f:
        json.dump(config_dict, f)
    
    run_general_fid(opts)
## get datasets
    
if __name__ == "__main__":
    main()