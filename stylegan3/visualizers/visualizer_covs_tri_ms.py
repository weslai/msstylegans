### This file is used to run the visualizer for changing two covariates
### c1 is fixed, c2, c3 and c4 are changed for example
import os
import re
import numpy as np
import torch
import click
from typing import List, Tuple, Union

from utils import load_generator, generate_images
from training.dataset_real_ms import UKBiobankMRIDataset2D, UKBiobankRetinalDataset, AdniMRIDataset2D, KaggleEyepacsDataset
from training.dataset_real_ms import NACCMRIDataset2D, RFMiDDataset
# from latent_mle_real_ms import SourceSampling
from visualizers.visual_two_covs import plot_covs_images_threesources

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
def parse_str(s: Union[str, List]) -> List[str]:
    '''Parse a comma separated list of numbers or ranges and return a list of strings.

    Example: '1,2,5-10' returns [c1, c2, c5, c6, c7, c8, c9, c10]
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
    for item in ranges:
        ranges[ranges.index(item)] = "c" + str(item)
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
        COVS = {"c1": "age", "c2": "diastolic blood pressure", "c3": "spherical power", 
                "c4": "diabetic retinopathy"}
    elif dataset == "mri":
        COVS = {"c1_1": "age", "c1_2": "age", "c2": "ventricle", "c3": "grey matter", 
                "c4": "clinical dementia rating", "c5": "apoe4"}
    return COVS
#----------------------------------------------------------------------------

@click.command()
@click.option('--network_pkl', 'network_pkl', help='Network pickle filename', default=None)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None)
@click.option('--variables', 'variables', type=parse_str, help='List of conditional variables', 
                                                default="2,4", show_default=True)
@click.option('--group-by', 'group_by', type=str, help='Group by a conditional variables')
@click.option('--num_subplots', 'num_subplots', type=int, help='Number of subplots', default=4, show_default=True)
@click.option('--source', 'source', type=click.Choice(["0", "1"]))
@click.option('--dataset', 'dataset', type=click.Choice(['mri', 'retinal', None]), 
                                                default=None, show_default=True)
@click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
@click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=True)
@click.option('--data-path3', 'data_path3', type=str, help='Path to the data source 3', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.9, show_default=True)
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
    variables: List[str],
    group_by: str,
    num_subplots: int,
    source: int,
    dataset: str,
    data_path1: str,
    data_path2: str,
    data_path3: str,
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
    ##
    print(f"variables: {variables}")
    if "c11" in variables:
        variables.remove("c11")
        variables.append("c1_1")
        variables = variables[::-1]
    if "c12" in variables:
        variables.remove("c12")
        variables.append("c1_2")
        variables = variables[::-1]
    if "c13" in variables:
        variables.remove("c13")
        variables.append("c1_3")
        variables = variables[::-1]
    assert group_by not in variables
    if group_by in ["c4", "c5"] and dataset == "mri":
        num_subplots = 3
    else:
        num_subplots = num_subplots

    if dataset == "mri":
        all_vars = ["c1_1", "c1_2", "c1_3", "c2", "c3", "c4", "c5"]
        assert group_by in all_vars
        for variable in variables:
            assert variable in all_vars
            if variable in all_vars:
                all_vars.remove(variable)
            if variable == "c1_1":
                all_vars.remove("c1_2")
                all_vars.remove("c1_3")
            elif variable == "c1_2":
                all_vars.remove("c1_1")
                all_vars.remove("c1_3")
            elif variable == "c1_3":
                all_vars.remove("c1_1")
                all_vars.remove("c1_2")
        if "c1_1" in all_vars:
            all_vars.remove("c1_2")
            all_vars.remove("c1_3")
        elif "c1_2" in all_vars:
            all_vars.remove("c1_1")
            all_vars.remove("c1_3")
        elif "c1_3" in all_vars:
            all_vars.remove("c1_1")
            all_vars.remove("c1_2")
    elif dataset == "retinal":
        all_vars = ["c1", "c2", "c3", "c4", "c5"]
        assert group_by in all_vars
        for variable in variables:
            assert variable in all_vars
            if variable in all_vars:
                all_vars.remove(variable)
    all_vars.remove(group_by)

    device = torch.device('cuda')
    num_labels = 5
    nrows = num_labels
    # Load the network.
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    os.makedirs(outdir, exist_ok=True)
    ## load the testset
    if dataset == "mri":
        ds1 = UKBiobankMRIDataset2D(data_name="ukb", ## UKB c = (age, greymatter, ventricle)
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds2 = AdniMRIDataset2D(data_name="adni",  ## ADNI c = (age, cdr)
                                path=data_path2, 
                                mode="test", 
                                use_labels=True,
                                xflip=False)
        ds3 = NACCMRIDataset2D(data_name="nacc",  ## NACC c = (age, apoe4, cdr)
                                path=data_path3, 
                                mode="test", 
                                use_labels=True,
                                xflip=False)
    elif dataset == "retinal":
        ds1 = UKBiobankRetinalDataset(data_name=dataset,
                                      path=data_path1,
                                      mode="test",
                                      use_labels=True,
                                      xflip=False)
        ds2 = KaggleEyepacsDataset(data_name=dataset,
                                    path=data_path2,
                                    mode="test",
                                    use_labels=True,
                                    xflip=False)
        ds3 = RFMiDDataset(data_name=dataset,
                            path=data_path3,
                            mode="test",
                            use_labels=True,
                            xflip=False)
    ## norm labels
    labels1 = ds1._load_raw_labels() ## (c1, c2, c3)
    labels1_min, labels1_max = labels1.min(axis=0), labels1.max(axis=0)
    labels2 = ds2._load_raw_labels() ## (c1, c4 (cdr))
    labels2_min, labels2_max = labels2.min(axis=0), labels2.max(axis=0)
    labels3 = ds3._load_raw_labels() ## (c1, c5 (apoe), c4 (cdr))
    labels3_min, labels3_max = labels3.min(axis=0), labels3.max(axis=0)
    
    ## add group_by into variables
    variables.append(group_by) ## last variable is group_by
    c_vars = {}
    c_vars_orig = {}
    for n, variable in enumerate(variables):
        if n == len(variables) - 1: ## group by variable
            num_labels = num_subplots
        if variable == "c1_1": ## age
            v = np.linspace(labels1_min[0], labels1_max[0], num=num_labels).reshape(-1, 1)
            v_orig = v * (ds1.model["age_max"] - ds1.model["age_min"]) + ds1.model["age_min"]
        elif variable == "c1_2": ## age
            v = np.linspace(labels2_min[0], labels2_max[0], num=num_labels).reshape(-1, 1)
            v_orig = v * (ds2.model["age_max"] - ds2.model["age_min"]) + ds2.model["age_min"]
        elif variable == "c1_3": ## age
            v = np.linspace(labels3_min[0], labels3_max[0], num=num_labels).reshape(-1, 1)
            v_orig = v * (ds3.model["age_max"] - ds3.model["age_min"]) + ds3.model["age_min"]
        elif variable == "c1": ## age retinal
            v = np.linspace(labels1_min[0], labels1_max[0], num=num_labels).reshape(-1, 1)
            v_orig = v * (ds1.model["age_max"] - ds1.model["age_min"]) + ds1.model["age_min"]
        elif variable == "c2":
            v = np.linspace(labels1_min[1], labels1_max[1], num=num_labels).reshape(-1, 1)
            if dataset == "mri":
                v_orig = np.exp(v * ds1.model["ventricle_std"] + ds1.model["ventricle_mu"])
            elif dataset == "retinal":
                v_orig = np.exp(v * ds1.model["diastolic_bp_std"] + ds1.model["diastolic_bp_mu"])
        elif variable == "c3":
            v = np.linspace(labels1_min[2], labels1_max[2], num=num_labels).reshape(-1, 1)
            if dataset == "mri":
                v_orig = np.exp(v * ds1.model["grey_matter_std"] + ds1.model["grey_matter_mu"])
            elif dataset == "retinal":
                v_orig = np.exp(v * ds1.model["spherical_power_left_std"] + ds1.model["spherical_power_left_mu"]) - 1e2
        elif variable == "c4":
            if dataset == "retinal":
                v = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
                v_orig = v
            elif dataset == "mri":
                v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                v_orig = np.array([0, 0.5, 1]).reshape(-1, 1)
        elif variable == "c5":
            if dataset == "mri":
                v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                v_orig = np.array([0, 1, 2]).reshape(-1, 1)
            elif dataset == "retinal":
                v = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                v_orig = np.array([0, 1]).reshape(-1, 1)
        c_vars[variable] = v
        c_vars_orig[variable] = v_orig
    c_fix = {}
    c_fix_orig = {}
    for remain_var in all_vars:
        if remain_var == "c1_1":
            c_mean = np.mean([labels1_min[0], labels1_max[0]]) ## age
            c_mean_orig = c_mean * (ds1.model["age_max"] - ds1.model["age_min"]) + ds1.model["age_min"]
        elif remain_var == "c1_2":
            c_mean = np.mean([labels2_min[0], labels2_max[0]]) ## age
            c_mean_orig = c_mean * (ds2.model["age_max"] - ds2.model["age_min"]) + ds2.model["age_min"]
        elif remain_var == "c1_3":
            c_mean = np.mean([labels3_min[0], labels3_max[0]]) ## age
            c_mean_orig = c_mean * (ds3.model["age_max"] - ds3.model["age_min"]) + ds3.model["age_min"]
        elif remain_var == "c1": ## age retinal
            c_mean = np.mean([labels1_min[0], labels1_max[0]])
            c_mean_orig = c_mean * (ds1.model["age_max"] - ds1.model["age_min"]) + ds1.model["age_min"]
        elif remain_var == "c2": ## ventricle or diastolic bp
            c_mean = np.mean([labels1_min[1], labels1_max[1]])
            if dataset == "mri":
                c_mean_orig = np.exp(c_mean * ds1.model["ventricle_std"] + ds1.model["ventricle_mu"])
            elif dataset == "retinal":
                c_mean_orig = np.exp(c_mean * ds1.model["diastolic_bp_std"] + ds1.model["diastolic_bp_mu"])
        elif remain_var == "c3": ## grey matter or spherical power
            c_mean = np.mean([labels1_min[2], labels1_max[2]])
            if dataset == "mri":
                c_mean_orig = np.exp(c_mean * ds1.model["grey_matter_std"] + ds1.model["grey_matter_mu"])
            elif dataset == "retinal":
                c_mean_orig = np.exp(c_mean * ds1.model["spherical_power_left_std"] + ds1.model["spherical_power_left_mu"]) - 1e2
        elif remain_var == "c4": ## cdr or diabetic retinopathy
            if dataset == "retinal":
                c_mean = np.random.choice([0, 1, 2, 3, 4])
                c_mean_orig = c_mean
            elif dataset == "mri":
                choice = np.random.choice([0, 1, 2])
                c_mean = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[choice].reshape(1, -1)
                c_mean_orig = np.array([0, 0.5, 1])[choice]
        elif remain_var == "c5": ## apoe4
            if dataset == "mri":
                choice = np.random.choice([0, 1, 2])
                c_mean = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[choice].reshape(1, -1)
                c_mean_orig = np.array([0, 1, 2])[choice]
            elif dataset == "retinal":
                c_mean = np.random.choice([0, 1, 3, 4]) ## only choose the first two classes
                c_mean = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])[c_mean].reshape(1, -1)
                c_mean_orig = [c_mean[0][0], c_mean[0][1]]
        c_fix[remain_var] = c_mean
        c_fix_orig[remain_var] = c_mean_orig

    if ("c4" in variables[:-1]) or ("c5" in variables[:-1]):
        if dataset == "mri":
            ncols = 3
        elif dataset == "retinal":
            ncols = 5
    else:
        ncols = num_labels
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        imgs = {}
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, Gen.z_dim)).to(device)
        for sub in range(num_subplots):
            gen_images = []
            for x in range(nrows):
                for y in range(ncols):
                    c_order = {}
                    if dataset == "mri":
                        for i, variable in enumerate(variables):
                            if i == len(variables) - 1:
                                if variable.startswith("c1"):
                                    c_order[1] = c_vars[variable][sub]
                                else:
                                    number = int(variable.split("c")[-1])
                                    c_order[number] = c_vars[variable][sub]
                            else:
                                if variable.startswith("c1"): ## c1...
                                    c_order[1] = c_vars[variable][x] if i == 0 else c_vars[variable][y]
                                else: ## not c1...
                                    number = int(variable.split("c")[-1])
                                    c_order[number] = c_vars[variable][x] if i == 0 else c_vars[variable][y]
                        for i, variable in enumerate(all_vars):
                            if variable.startswith("c1"):
                                if 1 not in c_order.keys():
                                    c_order[1] = np.array([c_fix[variable]])
                            else:
                                number = int(variable.split("c")[-1])
                                if number not in c_order.keys():
                                    c_order[number] = np.array([c_fix[variable]])
                    elif dataset == "retinal":
                        for i, variable in enumerate(variables):
                            if i == len(variables) - 1:
                                number = int(variable.split("c")[-1])
                                c_order[number] = c_vars[variable][sub]
                            else:
                                number = int(variable.split("c")[-1])
                                c_order[number] = c_vars[variable][x] if i == 0 else c_vars[variable][y]
                        for i, variable in enumerate(all_vars):
                            number = int(variable.split("c")[-1])
                            if number not in c_order.keys():
                                c_order[number] = np.array([c_fix[variable]])
                    if source == "0":
                        c_order[6] = np.array([1, 0, 0])
                    elif source == "1":
                        c_order[6] = np.array([0, 1, 0])
                    elif source == "2":
                        c_order[6] = np.array([0, 0, 1])
                    
                    cond_l = np.concatenate([c_order[key].reshape(1, -1) for key in sorted(c_order.keys())], axis=1)
                    l = torch.tensor(cond_l).reshape(1, -1).to(device)
                    img = generate_images(Gen, z, l, truncation_psi, noise_mode, translate, rotate)
                    gen_images.append(img)
            imgs[sub] = torch.cat(gen_images, dim=0)
        
        y_range, x_range = c_vars_orig[variables[0]], c_vars_orig[variables[1]]
        y_name, x_name = get_covs(dataset)[variables[0]], get_covs(dataset)[variables[1]]
        group_by_name = get_covs(dataset)[variables[-1]]
        group_by_range = c_vars_orig[variables[-1]]
        fix_name = f"s{source}_{all_vars[0]}={c_fix_orig[all_vars[0]]:.2f}_{all_vars[1]}={c_fix_orig[all_vars[1]]:.2f}"
        plot_covs_images_threesources(imgs, y_range, x_range, group_by_range, dataset_name=dataset,
                             c2_name=y_name, c3_name=x_name, group_name=group_by_name,
                             save_path=f'{outdir}/{fix_name}_seed{seed:04d}.png')

## --- run ---
if __name__ == "__main__":
    run_visualizer_two_covs()
## --- end of run ---