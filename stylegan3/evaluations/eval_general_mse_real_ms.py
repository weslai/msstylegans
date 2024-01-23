import os, sys
from typing import Tuple, Union
import click
import json
import numpy as np
import pandas as pd
import torch
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans")
import dnnlib

### --- Own --- ###
from utils import load_generator, load_regression_model, load_single_source_regression_model
from eval_strata_loss_utils import eval_loss_real_ms
from eval_dataset import UKBiobankMRIDataset2D
from eval_dataset import UKBiobankRetinalDataset2D
from eval_dataset import AdniMRIDataset2D, NACCMRIDataset2D
from eval_dataset import KaggleEyepacsDataset, RFMiDDataset
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
def load_dataset(
    dataset: str,
    source_gan: str,
    data_path1: str,
    data_path2: str = None,
    data_path3: str = None
):
    if source_gan == "single":
        if dataset == "ukb":
            ds1 = UKBiobankMRIDataset2D(data_name=dataset, 
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
        elif dataset == "adni":
            ds1 = AdniMRIDataset2D(data_name=dataset,
                                path=data_path1,
                                mode="test",
                                use_labels=True,
                                xflip=False)
        elif dataset == "nacc":
            ds1 = NACCMRIDataset2D(data_name=dataset,
                                path=data_path1,
                                mode="test",
                                use_labels=True,
                                xflip=False)
        elif dataset == "eyepacs":
            ds1 = KaggleEyepacsDataset(data_name=dataset,
                                path=data_path1,
                                mode="test",
                                use_labels=True,
                                xflip=False)
        elif dataset == "rfmid":
            ds1 = RFMiDDataset(data_name=dataset,
                                path=data_path1,
                                mode="test",
                                use_labels=True,
                                xflip=False)
    elif source_gan == "multi_mri":
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
        if data_path3 is not None:
            ds3 = NACCMRIDataset2D(data_name="nacc",
                                path=data_path3,
                                mode="test",
                                use_labels=True,
                                xflip=False)
    elif source_gan == "multi_retina":
        ds1 = UKBiobankRetinalDataset2D(data_name="retinal",
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds2 = RFMiDDataset(data_name="rfmid",
                            path=data_path2,
                            mode="test",
                            use_labels=True,
                            xflip=False)
        if data_path3 is not None:
            ds3 = KaggleEyepacsDataset(data_name="eyepacs",
                            path=data_path3,
                            mode="test",
                            use_labels=True,
                            xflip=False)

    else:
        raise ValueError(f"dataset {dataset} not found")
    return ds1 if source_gan == "single" else (ds1, ds2) if data_path3 is None else (ds1, ds2, ds3)

# --------------------------------------------------------------------------------------
def run_general_mse(opts):
    dataset = opts.dataset 
    num_bins = 3
    ## config
    network_pkl = opts.network_pkl
    metric_jsonl = opts.metric_jsonl
    regr_model0 = opts.regr_model0
    which_model0 = opts.which_model0
    task0 = opts.task0
    num_classes0 = opts.num_classes0
    regr_model1 = opts.regr_model1
    which_model1 = opts.which_model1
    task1 = opts.task1
    num_classes1 = opts.num_classes1
    regr_model2 = opts.regr_model2
    which_model2 = opts.which_model2
    task2 = opts.task2
    num_classes2 = opts.num_classes2
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

    ds = load_dataset(dataset, source_gan=source_gan, data_path1=data_path1, data_path2=data_path2, data_path3=data_path3)
    if source_gan == "single":
        labels_all = ds._load_raw_labels()
    else:
        labels_all = ds[0]._load_raw_labels() if dataset in ["ukb", "retinal"] else ds[1]._load_raw_labels() if dataset in ["adni", "rfmid"] else ds[2]._load_raw_labels()
    if dataset in ["ukb", "retinal", "adni", "rfmid"]:
        c1_all, c2_all, c3_all = labels_all[:,0], labels_all[:,1], labels_all[:,2]
        c1_min, c1_max = np.min(c1_all), np.max(c1_all)
        c2_min, c2_max = np.min(c2_all), np.max(c2_all)
        c3_min, c3_max = np.min(c3_all), np.max(c3_all)
        c1_hist = [np.quantile(c1_all, 1/num_bins), np.quantile(c1_all, 2/num_bins)]
        c2_hist = [np.quantile(c2_all, 1/num_bins), np.quantile(c2_all, 2/num_bins)]
        c3_hist = [np.quantile(c3_all, 1/num_bins), np.quantile(c3_all, 2/num_bins)]
        strata_hist = {"c1": c1_hist, "c2": c2_hist, "c3": c3_hist} ## define strata
    elif dataset == "nacc":
        c1_all, c2_all = labels_all[:,0], labels_all[:,1]
        c1_min, c1_max = np.min(c1_all), np.max(c1_all)
        c2_min, c2_max = np.min(c2_all), np.max(c2_all)
        c3_min, c3_max = None, None
        c1_hist = [np.quantile(c1_all, 1/num_bins), np.quantile(c1_all, 2/num_bins)]
        c2_hist = [np.quantile(c2_all, 1/num_bins), np.quantile(c2_all, 2/num_bins)]
        strata_hist = {"c1": c1_hist, "c2": c2_hist}
    else: ## eyepacs
        c1_all = labels_all[:,0]
        c1_min, c1_max = np.min(c1_all), np.max(c1_all)
        c2_min, c2_max = None, None
        c3_min, c3_max = None, None
        c1_hist = [np.quantile(c1_all, 1/num_bins), np.quantile(c1_all, 2/num_bins)]
        strata_hist = {"c1": c1_hist}
    ### cov
    if dataset == "ukb":
        cov_dict = {"age": 0, "ventricle": 1, "grey_matter": 2}
    elif dataset == "retinal":
        cov_dict = {"age": 0, "cataract": 1, "spherical": 2}
    elif dataset == "adni":
        cov_dict = {"age": 0, "left_hippocampus": 1, "right_hippocampus": 2}
    elif dataset == "nacc":
        cov_dict = {"age": 0, "apoe4": 1}
    elif dataset == "eyepacs":
        cov_dict = {"level": 0}
    elif dataset == "rfmid":
        cov_dict = {"disease_risk": 0, "MH": 1, "TSLN": 2}

    covariates_info = dict(
        c1_min = c1_min, c1_max = c1_max,
        c2_min = c2_min, c2_max = c2_max,
        c3_min = c3_min, c3_max = c3_max,
        strata_hist = strata_hist,
        cov = cov_dict
    )
    # Load the network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_gen = 64
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    if dataset in ["ukb", "retinal"]:
        regr_model0 = load_regression_model(regr_model0, which_model=which_model0, task=task0, ncls=num_classes0).to(device)
        if regr_model1 is not None:
            regr_model1 = load_regression_model(regr_model1, which_model=which_model1, task=task1, ncls=num_classes1).to(device)
        if regr_model2 is not None:
            regr_model2 = load_regression_model(regr_model2, which_model=which_model2, task=task2, ncls=num_classes2).to(device)
    elif dataset in ["adni", "nacc", "eyepacs", "rfmid"]:
        regr_model0 = load_single_source_regression_model(regr_model0, which_model=which_model0, task=task0, ncls=num_classes0).to(device)
        if regr_model1 is not None:
            regr_model1 = load_single_source_regression_model(regr_model1, which_model=which_model1, task=task1, ncls=num_classes1).to(device)
        if regr_model2 is not None:
            regr_model2 = load_single_source_regression_model(regr_model2, which_model=which_model2, task=task2, ncls=num_classes2).to(device)

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
        
    ## --------------------------- ##
    ### within strata (c1, c2, c3), calculate MSE, MAE
    scores, predictions_dict = eval_loss_real_ms(
        data_name = dataset,
        covariates=covariates_info,
        source_gan=source_gan,
        Gen=Gen,
        num_samples=num_samples,
        batch_size=batch_gen,
        device=device,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        translate=translate,
        rotate=rotate,
        regr_model0=regr_model0,
        regr_model1=regr_model1,
        regr_model2=regr_model2,
        dataset1=ds[0] if source_gan != "single" else ds,
        dataset2=ds[1] if data_path2 is not None else None,
        dataset3=ds[2] if data_path3 is not None else None,
        sampler_sources=sampler_sources
    )
    for key, value in covariates_info["cov"].items():
        if key in ["cataract", "apoe4", "level", "disease_risk", "MH", "TSLN"]:
            scores_df = pd.DataFrame(scores[key], columns=["accuracy", "precision", "recall", 
                                                "f1", "corr"])
        else:
            scores_df = pd.DataFrame(scores[key], columns=["mse", "mae", "corr"])
        scores_df.to_csv(os.path.join(outdir, f"general_test_loss_{key}_{dataset}.csv"), index=False)
        for i in range(len(predictions_dict[key])):
            predictions_dict[key][i].to_csv(os.path.join(outdir, f"general_test_predictions_{key}_stra{i}.csv"),
                                    index=False)

## --------------------------- ##
@click.command()
@click.option('--network_specific', 'network_pkl', help='Network pickle filepath', default=None, required=False)
@click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None, required=False)
@click.option('--regr_model0', 'regr_model0', help='Regression model', type=str, required=True)
@click.option('--which_model0', 'which_model0', help='Which model is trained', type=str, required=True)
@click.option('--task0', 'task0', help='Task', type=str, required=True)
@click.option('--ncls0', 'num_classes0', help='Number of classes', type=int, required=False, default=None)
@click.option('--regr_model1', 'regr_model1', help='Regression model', type=str, required=False)
@click.option('--which_model1', 'which_model1', help='Which model is trained', type=str, required=False)
@click.option('--task1', 'task1', help='Task', type=str, required=False)
@click.option('--ncls1', 'num_classes1', help='Number of classes', type=int, required=False, default=None)
@click.option('--regr_model2', 'regr_model2', help='Regression model', type=str, required=False)
@click.option('--which_model2', 'which_model2', help='Which model is trained', type=str, required=False)
@click.option('--task2', 'task2', help='Task', type=str, required=False)
@click.option('--ncls2', 'num_classes2', help='Number of classes', type=int, required=False, default=None)
@click.option('--dataset', 'dataset', type=click.Choice(['ukb', 'retinal', 'adni', 'nacc', 'eyepacs', 'rfmid',
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
    assert opts.regr_model0 is not None

    os.makedirs(opts.outdir, exist_ok=True)
    config_dict = {
        "gen_specific": opts.network_pkl,
        "gen": opts.metric_jsonl,
        "regr_model0": opts.regr_model0,
        "which_model0": opts.which_model0,
        "task0": opts.task0,
        "ncls0": opts.num_classes0,
        "regr_model1": opts.regr_model1,
        "which_model1": opts.which_model1,
        "task1": opts.task1,
        "ncls1": opts.num_classes1,
        "regr_model2": opts.regr_model2,
        "which_model2": opts.which_model2,
        "task2": opts.task2,
        "ncls2": opts.num_classes2,
        "dataset": opts.dataset, 
        "data_path1": opts.data_path1, "data_path2": opts.data_path2,
        "data_path3": opts.data_path3, "source_gan": opts.source_gan,
        "num_samples": opts.num_samples, "out_dir": opts.outdir}
    with open(os.path.join(opts.outdir, "strata_mse_config.json"), "w") as f:
        json.dump(config_dict, f)

    run_general_mse(opts)
if __name__ == "__main__":
    main()