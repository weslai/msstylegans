### This file is used to run the visualizer for changing two covariates
### c1 is fixed, c2 and c3 are changed
import os
import re
import numpy as np
import torch
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
def get_covs(dataset):
    if dataset == "retinal":
        COVS = {"c1": "age", "c2": "diastolic bp", "c3": "spherical power left"}
    elif dataset == "ukb":
        pass
    elif dataset == "mnist-thickness-intensity-slant":
        COVS = {"c1": "thickness", "c2": "intensity", "c3": "slant"}
    return COVS
#----------------------------------------------------------------------------

@click.command()
@click.option('--network_pkl', 'network_pkl', help='Network pickle filename', default=None)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None)
@click.option('--group-by', 'group_by', type=str, default="c1", show_default=True)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb', 
                                                         'retinal', None]), default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.8, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), 
              default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
              default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
              show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def run_visualizer_two_covs(
    network_pkl: str,
    metric_jsonl: str,
    group_by: str,
    dataset: str,
    data_path1: str,
    data_path2: str,
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

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """
    device = torch.device('cuda')
    num_labels = 3
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
    elif dataset == "retinal":
        ds1 = UKBiobankRetinalDataset(data_name=dataset,
                                      path=data_path1,
                                      mode="test",
                                      use_labels=True,
                                      xflip=False)
        ds2 = UKBiobankRetinalDataset(data_name=dataset,
                                      path=data_path2,
                                      mode="test",
                                      use_labels=True,
                                      xflip=False)

    elif dataset == "mnist-thickness-intensity-slant":
        dataset2 = "mnist-thickness-intensity-slant"
        ## seed is as the common convariate (c1)
        ds1 = MorphoMNISTDataset_causal(data_name=dataset,
                                        path=data_path1,
                                        mode="test",
                                        use_labels=True,
                                        xflip=False)
        ds2 = MorphoMNISTDataset_causal(data_name=dataset2,
                                        path=data_path2,
                                        mode="test",
                                        use_labels=True,
                                        xflip=False)
    labels1 = ds1._load_raw_labels() ## (c1, c2)
    labels1_min, labels1_max = labels1.min(axis=0), labels1.max(axis=0)
    labels2 = ds2._load_raw_labels() ## (c1, c3)
    labels2_min, labels2_max = labels2.min(axis=0), labels2.max(axis=0)
    c1_min, c1_max = min(labels1_min[0], labels2_min[0]), max(labels1_max[0], labels2_max[0]) ## c1 fixed
    if group_by == "c1":
        c1 = np.mean([c1_min, c1_max])
        c1_range = np.array([c1] * num_labels).reshape(-1, 1)
        c2_range = np.linspace(labels1_min[1]-3, labels1_max[1]+3, num=num_labels).reshape(-1, 1)
        c3_range = np.linspace(labels2_min[1]-3, labels2_max[1]+3, num=num_labels).reshape(-1, 1)
    elif group_by == "c2":
        c2_min, c2_max = labels1_min[1], labels1_max[1]
        c2 = np.mean([c2_min, c2_max]) + 0.5
        c1_range = np.linspace(c1_min-2, c1_max+2, num=num_labels).reshape(-1, 1)
        c2_range = np.array([c2] * num_labels).reshape(-1, 1)
        c3_range = np.linspace(labels2_min[1]-2, labels2_max[1]+2, num=num_labels).reshape(-1, 1)
    elif group_by == "c3":
        c3_min, c3_max = labels2_min[1], labels2_max[1]
        c3 = np.mean([c3_min, c3_max])
        c1_range = np.linspace(c1_min-2, c1_max+2, num=num_labels).reshape(-1, 1)
        c2_range = np.linspace(labels1_min[1]-2, labels1_max[1]+2, num=num_labels).reshape(-1, 1)
        c3_range = np.array([c3] * num_labels).reshape(-1, 1)
    c = torch.tensor(np.concatenate([c1_range, c2_range, c3_range], axis=1)).to(device)
    
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        gen_images = []
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, Gen.z_dim)).to(device)
        for x in range(c.shape[0]):
            for y in range(c.shape[0]):
                if group_by == "c1":
                    l = torch.tensor((c1_range[0], c2_range[x], c3_range[y])).reshape(1, -1).to(device)
                elif group_by == "c2":
                    l = torch.tensor((c1_range[x], c2_range[0], c3_range[y])).reshape(1, -1).to(device)
                elif group_by == "c3":
                    l = torch.tensor((c1_range[x], c2_range[y], c3_range[0])).reshape(1, -1).to(device)
                img = generate_images(Gen, z, l, truncation_psi, noise_mode, translate, rotate)
                gen_images.append(img)
        imgs = torch.cat(gen_images, dim=0)
        if group_by == "c1":
            y_range = c2_range
            y_name = get_covs(dataset)["c2"]
            x_range = c3_range
            x_name = get_covs(dataset)["c3"]
        elif group_by == "c2":
            y_range = c1_range
            y_name = get_covs(dataset)["c1"]
            x_range = c3_range
            x_name = get_covs(dataset)["c3"]
        elif group_by == "c3":
            y_range = c1_range
            y_name = get_covs(dataset)["c1"]
            x_range = c2_range
            x_name = get_covs(dataset)["c2"]
        plot_two_covs_images(imgs, y_range, x_range, dataset_name=dataset,
                             c2_name=y_name, c3_name=x_name,
                            save_path=f'{outdir}/seed{seed:04d}.png',
                            single_source=False)

## --- run ---
if __name__ == "__main__":
    run_visualizer_two_covs()
## --- end of run ---