### This file is used to run the visualizer for plotting two covariates (most negatives and most positives)
### c1 does not matter, c2 and c3 are changed
import os
import re
import numpy as np
import torch
import click
from typing import List, Tuple, Union

from utils import load_generator, generate_images
from evaluations.eval_dataset import UKBiobankMRIDataset2D, MorphoMNISTDataset_causal
from evaluations.eval_dataset import UKBiobankRetinalDataset2D
from visualizers.visual_two_covs import plot_negpos_images

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
        COVS = {"c1": "age", "c2": "diastolic blood pressure", "c3": "spherical power"}
    elif dataset == "ukb":
        COVS = {"c1": "age", "c2": "ventricle", "c3": "grey matter"}
    elif dataset == "mnist-thickness-intensity-slant":
        COVS = {"c1": "thickness", "c2": "intensity", "c3": "slant"}
    return COVS
#----------------------------------------------------------------------------

@click.command()
@click.option('--network_pkl', 'network_pkl', help='Network pickle filename', default=None)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', required=True)
@click.option('--group-by', 'group_by', type=str, default="c1", show_default=True)
@click.option('--quantile', 'quantile', type=float, default=0.90, show_default=True)
@click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb', 
                                                         'retinal', None]),
              default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=True)
@click.option('--num-plots', 'num_plots', type=int, help='Number of plots', default=1, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.9, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), 
              default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
              default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
              show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def run_visualizer_neg_pos_covs(
    network_pkl: str,
    metric_jsonl: str,
    group_by: str,
    quantile: float,
    dataset: str,
    data_path1: str,
    data_path2: str,
    num_plots: int,
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
    num_images = 5
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
    labels1 = ds1._load_raw_labels() ## (c1, c2, c3)
    labels2 = ds2._load_raw_labels() ## (c1, c2, c3)
    len_labels1 = labels1.shape[0]
    all_labels = np.concatenate([labels1, labels2], axis=0)

    min_sort_criteria = np.quantile(all_labels, 1 - quantile, axis=0)
    max_sort_criteria = np.quantile(all_labels, quantile, axis=0)
    if group_by == "c1": ## c2, c3
        min_all_labels_idxs = np.where((all_labels[:, 1] <= min_sort_criteria[1]) & (all_labels[:, 2] <= min_sort_criteria[2]))[0][:num_images]
        max_labels_1_idxs = np.where((all_labels[:, 1] <= min_sort_criteria[1]) & (all_labels[:, 2] >= max_sort_criteria[2]))[0][:num_images]
        max_labels_2_idxs = np.where((all_labels[:, 1] >= max_sort_criteria[1]) & (all_labels[:, 2] <= min_sort_criteria[2]))[0][:num_images]
        max_all_labels_idxs = np.where((all_labels[:, 1] >= max_sort_criteria[1]) & (all_labels[:, 2] >= max_sort_criteria[2]))[0][:num_images]
    elif group_by == "c2": ## c1, c3
        min_all_labels_idxs = np.where((all_labels[:, 0] <= min_sort_criteria[0]) & (all_labels[:, 2] <= min_sort_criteria[2]))[0][:num_images]
        max_labels_1_idxs = np.where((all_labels[:, 0] <= min_sort_criteria[0]) & (all_labels[:, 2] >= max_sort_criteria[2]))[0][:num_images]
        max_labels_2_idxs = np.where((all_labels[:, 0] >= max_sort_criteria[0]) & (all_labels[:, 2] <= min_sort_criteria[2]))[0][:num_images]
        max_all_labels_idxs = np.where((all_labels[:, 0] >= max_sort_criteria[0]) & (all_labels[:, 2] >= max_sort_criteria[2]))[0][:num_images]
    elif group_by == "c3": ## c1, c2
        min_all_labels_idxs = np.where((all_labels[:, 0] <= min_sort_criteria[0]) & (all_labels[:, 1] <= min_sort_criteria[1]))[0][:num_images]
        max_labels_1_idxs = np.where((all_labels[:, 0] <= min_sort_criteria[0]) & (all_labels[:, 1] >= max_sort_criteria[1]))[0][:num_images]
        max_labels_2_idxs = np.where((all_labels[:, 0] >= max_sort_criteria[0]) & (all_labels[:, 1] <= min_sort_criteria[1]))[0][:num_images]
        max_all_labels_idxs = np.where((all_labels[:, 0] >= max_sort_criteria[0]) & (all_labels[:, 1] >= max_sort_criteria[1]))[0][:num_images]
    
    # Generate images.
    for i in range(num_plots):
        real_dict = {}
        gen_dict = {}
        labels_dict = {}
        num = 0
        for labels in [min_all_labels_idxs, max_labels_1_idxs, max_labels_2_idxs, max_all_labels_idxs]:
            real_images = []
            gen_images = []
            real_labels = []
            for idx in labels:
                # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                seed = np.random.randint(0, 100000)
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, Gen.z_dim)).to(device)
                if idx < len_labels1:
                    real_img, real_label = ds1[idx]
                    real_label_norm = ds1.get_norm_label(idx)
                else:
                    real_img, real_label = ds2[idx-len_labels1]
                    real_label_norm = ds2.get_norm_label(idx-len_labels1)
                l = torch.tensor(real_label_norm).reshape(1, -1).to(device)
                # if which == 0:
                #     l[0, 1], l[0, 2] = l[0, 1] - 0.5, l[0, 2] - 0.5
                # elif which == 1:
                #     l[0, 1], l[0, 2] = l[0, 1] - 0.5, l[0, 2] + 0.5
                # elif which == 2:
                #     l[0, 1], l[0, 2] = l[0, 1] + 0.5, l[0, 2] - 0.5
                # elif which == 3:
                #     l[0, 1], l[0, 2] = l[0, 1] + 0.5, l[0, 2] + 0.5
                img = generate_images(Gen, z, l, truncation_psi, noise_mode, translate, rotate)
                real_img = torch.tensor(real_img.transpose(1, 2, 0)).unsqueeze(0)
                real_images.append(real_img)
                gen_images.append(img)
                real_labels.append(real_label.reshape(1, -1))
            real_imgs = torch.cat(real_images, dim=0)
            imgs = torch.cat(gen_images, dim=0)
            real_l_concat = np.concatenate(real_labels, axis=0)
            real_dict[str(num)] = real_imgs
            gen_dict[str(num)] = imgs
            labels_dict[str(num)] = real_l_concat
            num += 1
            if group_by == "c1":
                y_name = get_covs(dataset)["c2"]
                x_name = get_covs(dataset)["c3"]
            elif group_by == "c2":
                y_name = get_covs(dataset)["c1"]
                x_name = get_covs(dataset)["c3"]
            elif group_by == "c3":
                y_name = get_covs(dataset)["c1"]
                x_name = get_covs(dataset)["c2"]
        plot_negpos_images(real_images=real_dict,
                        gen_images=gen_dict,
                        labels=labels_dict,
                        dataset_name=dataset,
                        c2_name=y_name,
                        c3_name=x_name,
                        save_path=f'{outdir}/seed{i}.png',
                        single_source=False
        )

## --- run ---
if __name__ == "__main__":
    run_visualizer_neg_pos_covs()
## --- end of run ---