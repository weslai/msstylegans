### This file is used to run the visualizer for changing two covariates
### c1 is fixed, c2 and c3 are changed
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import click
from typing import List, Tuple, Union

from utils import load_generator, generate_images
from training.dataset import UKBiobankMRIDataset2D, MorphoMNISTDataset_causal, UKBiobankRetinalDataset
from visualizers.visual_two_covs import plot_two_covs_images

# --------------------------------------------------------------------------------------
def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges
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

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m
#----------------------------------------------------------------------------
def get_covs(dataset, source):
    if dataset == "retinal":
        if source == "source1":
            COVS = {"c1": "age", "c2": "diastolic blood pressure"}
        elif source == "source2":
            COVS = {"c1": "age", "c2": "spherical power"}
    elif dataset == "mri":
        if source == "source1":
            COVS = {"c1": "age", "c2": "ventricle"}
        elif source == "source2":
            COVS = {"c1": "age", "c2": "grey matter"}
    if dataset == "morphomnist":
        if source == "source1":
            COVS = {"c1": "thickness", "c2": "intensity"}
        elif source == "source2":
            COVS = {"c1": "thickness", "c2": "slant"}
    return COVS

#----------------------------------------------------------------------------

@click.command()
@click.option('--network_pkl', 'network_pkl', help='Network pickle filename', default=None)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', required=True)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity', 'mnist-thickness-slant',
                                                         'mnist-thickness-intensity-slant',
                                                         'ukb', 'retinal', None]),
              default=None, show_default=True)
@click.option('--data-path', 'data_path', type=str, help='Path to the data source', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.5, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), 
              default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
              default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
              show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def run_visualizer_two_covs_singlesource(
    network_pkl: str,
    metric_jsonl: str,
    dataset: str,
    data_path: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float,float],
    rotate: float,
    outdir: str
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    """
    device = torch.device('cuda')
    num_labels = 5
    # Load the network.
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )

    os.makedirs(outdir, exist_ok=True)
    ## load the testset
    ## import sampler
    if dataset == "ukb":
        ds = UKBiobankMRIDataset2D(data_name=dataset,
                                    path=data_path,
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
    elif dataset == "retinal":
        ds = UKBiobankRetinalDataset(data_name=dataset,
                                    path=data_path,
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
    elif dataset in ["mnist-thickness-intensity", "mnist-thickness-slant", "mnist-thickness-intensity-slant"]:
        ## seed is as the common convariate (c1)
        ds = MorphoMNISTDataset_causal(data_name=dataset,
                                        path=data_path,
                                        mode="test",
                                        use_labels=True,
                                        xflip=False)
    which_source = ds.which_source
    if dataset == "ukb":
        dataset = "mri"
    elif dataset == "mnist-thickness-intensity-slant":
        dataset = "morphomnist"
        c3_fix = F.one_hot(torch.tensor(np.array([3] * num_labels).reshape(-1, 1), dtype=torch.long),
            num_classes=10).squeeze(1).to(device)
    ## norm labels
    labels = ds._load_raw_labels() ## (c1, c2)
    labels_min, labels_max = labels.min(axis=0), labels.max(axis=0)
    c1_range = np.linspace(labels_min[0], labels_max[0], num=num_labels).reshape(-1, 1)
    c2_range = np.linspace(labels_min[1], labels_max[1], num=num_labels).reshape(-1, 1)
    c = torch.tensor(np.concatenate([c1_range, c2_range], axis=1)).to(device)
    
    ## labels
    if dataset == "mri":
        c1_orig = c1_range * (ds.model["age_max"] - ds.model["age_min"]) + ds.model["age_min"]
        if which_source == "source1":
            c2_orig = np.exp(c2_range * ds.model["ventricle_std"] + ds.model["ventricle_mu"])
        elif which_source == "source2":
            c2_orig = np.exp(c2_range * ds.model["grey_matter_std"] + ds.model["grey_matter_mu"])
    elif dataset == "retinal":
        c1_orig = c1_range * (ds.model["age_max"] - ds.model["age_min"]) + ds.model["age_min"]
        if which_source == "source1":
            c2_orig = np.exp(c2_range * ds.model["diastolic_bp_std"] + ds.model["diastolic_bp_mu"])
        elif which_source == "source2":
            c2_orig = np.exp(c2_range * ds.model["spherical_power_left_std"] + ds.model["spherical_power_left_mu"]) - 1e2
    elif dataset == "morphomnist":
        c1_orig = c1_range * ds.model["thickness_std"] + ds.model["thickness_mu"]
        if which_source == "source1":
            c2_orig = c2_range * ds.model["intensity_std"] + ds.model["intensity_mu"]
        elif which_source == "source2":
            c2_orig = c2_range * ds.model["slant_std"] + ds.model["slant_mu"]
    
    if dataset == "morphomnist":
        c = torch.cat([c, c3_fix], dim=1)
    
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        gen_images = []
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, Gen.z_dim)).to(device)
        for x in range(c.shape[0]):
            for y in range(c.shape[0]):
                l = torch.tensor((c1_range[x], c2_range[y])).reshape(1, -1).to(device)
                img = generate_images(Gen, z, l, truncation_psi, noise_mode, translate, rotate)
                gen_images.append(img)
        imgs = torch.cat(gen_images, dim=0)
        c1_name = get_covs(dataset, which_source)["c1"]
        c2_name = get_covs(dataset, which_source)["c2"]
        plot_two_covs_images(imgs, c1_orig, c2_orig, dataset_name=dataset,
                            c2_name=c1_name, c3_name=c2_name,
                            save_path=f'{outdir}/seed{seed:04d}.png'
                            )

## --- run ---
if __name__ == "__main__":
    run_visualizer_two_covs_singlesource()
## --- end of run ---