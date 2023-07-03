### -------------------
### --- Third-Party ---
### -------------------
import os
from typing import Tuple
import pandas as pd
import numpy as np
import random
import seaborn as sns
import torch
import dnnlib
import legacy

### -----------
### --- Own ---
### -----------
from evaluations.eval_utils import get_k_lowest_checkpoints

## pairplot
def plot_pairplots(df, diag_kind: str = "hist", save_path: str = None):
    """
        df: (pd.DataFrame) with columns names (add type as a column)
    """
    # grid = sns.PairGrid(data=df, hue="type")
    grid = sns.pairplot(data=df, hue="type", diag_kind=diag_kind, size=2)
    grid.savefig(save_path)
    grid.savefig(save_path.replace(".png", ".pdf"))
    return grid


### --- UKB Data preparation ---
### --- Datasets split ---
def annotations_split(
    annotation_file,
    save_path: str = None
):
    """
    This is for multi-source datasets preparation
    Here first for ukb, 
    we create in one source and split it into two sources
    the same id should be only in a single source
    """
    # read the annotation file
    df = pd.read_csv(annotation_file)
    # get the unique ids
    filepaths = df['filepath_MNIlin']
    filepaths_dict = {} ## id: [filepaths]
    ids = []
    for filepath in filepaths:
        id = filepath.split('/')[-3].split('_')[0]
        if id not in ids:
            ids.append(id)
            filepaths_dict[id] = [filepath]
        else:
            filepaths_dict[id].append(filepath)
    assert len(ids) == len(filepaths_dict)
    # shuffle the ids
    random.shuffle(ids)
    first_ids_list = ids[:len(ids)//2]
    # split the annotation file
    first_df, second_df = None, None
    for key, value in filepaths_dict.items(): ## key is id, value is filepaths with lists
        if key in first_ids_list:
            if first_df is None:
                first_df = df[df['filepath_MNIlin'].isin(value)]
            else:
                first_df = pd.concat([first_df, df[df['filepath_MNIlin'].isin(value)]], ignore_index=True)
        else:
            if second_df is None:
                second_df = df[df['filepath_MNIlin'].isin(value)]
            else:
                second_df = pd.concat([second_df, df[df['filepath_MNIlin'].isin(value)]], ignore_index=True)
    # save the annotation file
    if save_path is not None:
        first_df.to_csv(os.path.join(save_path, 'ukb_linear_freesurfer_first_annotation.csv'), index=False)
        second_df.to_csv(os.path.join(save_path, 'ukb_linear_freesurfer_second_annotation.csv'), index=False)
    return first_df, second_df

### GeneraterLoader ###
### Load pre-trained Model for causal-stylegan3
def load_generator(
    metric_jsonl: str,
    network_pkl: str = None,
    use_cuda: bool = True
):
    if network_pkl is None:
        assert metric_jsonl is not None
        pkl, _, _ = get_k_lowest_checkpoints(metric_jsonl, k=1)
        network_pkl = os.path.join(os.path.dirname(metric_jsonl), pkl[0])
    else:
        print("loading networks from '%s' ..." % network_pkl)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    return G

def load_discriminator(
    metric_jsonl: str,
    network_pkl: str = None,
    use_cuda: bool = True
):
    if network_pkl is None:
        assert metric_jsonl is not None
        pkl, _, _ = get_k_lowest_checkpoints(metric_jsonl, k=1)
        network_pkl = os.path.join(os.path.dirname(metric_jsonl), pkl[0])
    else:
        print("loading networks from '%s' ..." % network_pkl)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    with dnnlib.util.open_url(network_pkl) as f:
        D = legacy.load_network_pkl(f)["D"].to(device)
    return D

### -------------------- ###
### FOR IMAGE GENERATION ###
### -------------------- ###
def generate_images(
    generator, z, c, 
    truncation_psi, noise_mode, translate, rotate
):
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
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
    if hasattr(generator.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        generator.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = generator(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img