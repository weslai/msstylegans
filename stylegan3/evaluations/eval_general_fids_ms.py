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
from eval_dataset import ConcatDataset
from eval_dataset import UKBiobankMRIDataset2D
from eval_dataset import UKBiobankRetinalDataset2D
from training.dataset_real_ms import AdniMRIDataset2D, KaggleEyepacsDataset
from latent_mle_real_ms import SourceSampling

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
@click.option('--dataset', 'dataset', type=click.Choice(['mri', 
                                                         'retinal', None]),
              default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', default=None, required=False)
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
def run_general_fid(
    network_pkl: str,
    metric_jsonl: str,
    dataset: str,
    data_path1: str,
    data_path2: str,
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
    os.makedirs(outdir, exist_ok=True)
    config_dict = {
        "gen_specific": network_pkl, "gen": metric_jsonl, "dataset": dataset, 
        "data_path1": data_path1, "data_path2": data_path2,
        "num_samples": num_samples, "out_dir": outdir}
    with open(os.path.join(outdir, "generalfid_config.json"), "w") as f:
        json.dump(config_dict, f)

    if dataset == "mri":
        ds1 = UKBiobankMRIDataset2D(data_name="ukb", 
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds2 = AdniMRIDataset2D(data_name="adni", 
                                    path=data_path2, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        concat_ds = ConcatDataset(ds1, ds2)
        sampler1 = SourceSampling(dataset="ukb", label_path=ds1._path)
        sampler2 = SourceSampling(dataset="adni", label_path=ds2._path)
        df = sampler1.get_graph()
        df2 = sampler2.get_graph()
        model = sampler1.get_causal_model()
        model2 = sampler2.get_causal_model()

    elif dataset == "retinal":
        ds1 = UKBiobankRetinalDataset2D(data_name="retinal",
                                    path=data_path1,
                                    mode="test",
                                    use_labels=True,
                                    xflip=False)
        ds2 = KaggleEyepacsDataset(data_name="eyepacs",
                                    path=data_path2,
                                    mode="test",
                                    use_labels=True,
                                    xflip=False)
        concat_ds = ConcatDataset(ds1, ds2)
        sampler1 = SourceSampling(dataset="retinal", label_path=ds1._path)
        sampler2 = SourceSampling(dataset="eyepacs", label_path=ds2._path)
        df = sampler1.get_graph()
        df2 = sampler2.get_graph()
        model = sampler1.get_causal_model()
        model2 = sampler2.get_causal_model()
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
    ### get the real images
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    real_imgs = []
    gen_imgs = []
    ## real images
    for source1, source2 in torch.utils.data.DataLoader(concat_ds, batch_size=batch_gen, shuffle=True, **data_loader_kwargs):
        img1, _labels1 = source1[0], source1[1]
        img2, _labels2 = source2[0], source2[1]
        for img in [img1, img2]:
            if img.shape[1] != 1:
                img = torch.tensor(img)
            real_imgs.append(img)
    ## fake images
    for _ in range(num_samples // (batch_gen * 2)):
        batch_labels = []
        z = torch.randn(batch_gen * 2, Gen.z_dim).to(device)
        for _i in range(batch_gen):
            ## (c1, c2, c3)
            source_c1, source_c2 = concat_ds[np.random.randint(len(concat_ds))]
            label1, label2 = source_c1[1], source_c2[1]
            label1_norm = ds1._normalise_labels(label1[0].reshape(-1, 1), label1[1].reshape(-1, 1),
                                                label1[2].reshape(-1, 1)).reshape(1, -1)
            if dataset == "mri":
                age2 = label2[0].reshape(-1, 1) * (ds2.model["age_max"] - ds2.model["age_min"]) + ds2.model["age_min"]
                label1_hidd = sampler2.sampling_given_age(label1[0].reshape(-1, 1), normalize=True) ## age, cdr
                label2_hidd = sampler1.sampling_given_age(age2, normalize=True) ## age, vols
                label1_norm = torch.cat([torch.tensor(label1_norm), label1_hidd[0, 1:].reshape(1, -1), torch.tensor([0]).reshape(-1, 1)], dim=1)
                label2_norm = torch.cat([label2_hidd, torch.tensor(label2[1:].reshape(1, -1)), torch.tensor([1]).reshape(-1, 1)], dim=1)
            elif dataset == "retinal": ## ukb + eyepacs
                label1_hidd = sampler2.sampling_model(1)[-1] ## discease
                label2_hidd = sampler1.sample_normalize(1) ## age, bp, spherical
                label1_norm = torch.cat([torch.tensor(label1_norm), label1_hidd, torch.tensor([0]).reshape(-1, 1)], dim=1)
                label2_norm = torch.cat([label2_hidd, torch.tensor(label2.reshape(1, -1)), torch.tensor([1]).reshape(-1, 1)], dim=1)
            l = torch.cat([label1_norm, label2_norm], dim=0).to(device)
            batch_labels.append(l)
        ll = torch.cat(batch_labels, dim=0).to(device)
        batch_imgs = generate_images(Gen, z, ll, truncation_psi, noise_mode, translate, rotate).permute(0,3,1,2)
        gen_imgs.append(batch_imgs)
    if dataset != "retinal":
        real_imgs = torch.cat(real_imgs, dim=0).repeat([1, 3, 1, 1]).to(device) ## (batch_size, channel (3), pixel, pixel)
        gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,3,1,1]).to(device) ## (batch_size, channel (3), pixel, pixel)
    else:
        real_imgs = torch.cat(real_imgs, dim=0).to(device)
        gen_imgs = torch.cat(gen_imgs, dim=0).to(device)
    print(f"Real images: {real_imgs.shape}")
    print(f"Generated images: {gen_imgs.shape}")
    ### calculate FID
    fid_score = calc_fid_score(real_imgs, gen_imgs, batch_size=64).cpu().detach().numpy()
    print(f"FID: {fid_score} for {num_samples} samples")
    ### save the evaluation analysis to a json file
    result = dict(fid=fid_score, num_samples=num_samples,
                  dataset=dataset,
                  network=metric_jsonl,
                  data_path1=data_path1,
                  data_path2=data_path2 if data_path2 is not None else "None"
                  )
    result_df = pd.DataFrame.from_dict(result, orient="index").T
    result_df.to_csv(os.path.join(outdir, "ms_general_fid.csv"), index=False)
    
if __name__ == "__main__":
    run_general_fid()