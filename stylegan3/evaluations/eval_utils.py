### -------------------
### --- Third Party ---
### -------------------
import os, glob
import time
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import json
import dnnlib
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from metrics.metric_main import is_valid_metric, _metric_dict
### -----------
### --- Own ---
### -----------
from metrics.metric_utils_general import MetricOptionsGeneral
from metrics.metric_utils_stratified import MetricOptions


def get_k_lowest_checkpoints(metric_jsonl, k=5):
    """
        Get k lowest checkpoints from a jsonl file.
        return checkpoint paths, fid scores, index.
    Args:
        metric_jsonl (str): path to jsonl file
        k (int): number of lowest checkpoints
    """
    df_fid = pd.read_json(metric_jsonl, lines=True, orient="records")
    fid_scores = df_fid["results"].values
    fid_scores = np.array([fid_scores[i]["fid50k_full"] for i in range(len(fid_scores))])
    k_idxs = np.argsort(fid_scores)[:k]
    k_paths = df_fid["snapshot_pkl"].values[k_idxs]
    return k_paths, fid_scores[k_idxs], k_idxs

def delete_checkpoints(result_path: str, best_checkpoints: np.ndarray):
    """
        Delete checkpoints.
        in order to save memory of the disks.
    Args:
        result_path (str): path to the result folder
        best_checkpoints (np.ndarry): array of checkpoint paths (not to delete) (just the output(k_paths) from get_k_lowest_checkpoints)
    """
    if not result_path.split("/")[0]:
        print("Absolute path given")
    else:
        result_path = "/" + result_path
    all_ckpts_path = os.path.join(result_path, "*.pkl")
    all_ckpts = glob.glob(all_ckpts_path, recursive=False)
    abs_path_best_checkpoints = list(result_path + "/" + best_checkpoints)
    ckpt_to_delete = [item for item in all_ckpts if item not in abs_path_best_checkpoints]
    assert len(ckpt_to_delete) == len(all_ckpts) - len(abs_path_best_checkpoints)
    for checkpoint_path in ckpt_to_delete:
        # Removing the checkpoint file from the directory
        print("Removing ", checkpoint_path)
        os.remove(checkpoint_path)

### --------------------------------------------- ###
### Calculate the metric for general and stratified Intervene ###
### --------------------------------------------- ###
def calc_general_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = MetricOptionsGeneral(**kwargs)
    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

def calc_stratified_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = MetricOptions(**kwargs)
    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

### ------------------------------------------- ###
### save the evaluation analysis to a json file ###
### ------------------------------------------- ###
def save_eval_report(result_dict, save_path, save_name, snapshot_pkl=None):
    """
    Mostly copy from metric_main.report_metric
    """
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, save_path)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    # Create output directory.
    print('Creating output directory...')
    assert save_path is not None
    os.makedirs(save_path, exist_ok=True)
    if os.path.isdir(save_path):
        with open(os.path.join(save_path, f'metric-{metric}-{save_name}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

### ------------------------------------------- ###
### Argument wrapper for the evaluation script ###
### ------------------------------------------- ###
def calc_fid_score(
    img_dist1,
    img_dist2,
    batch_size=64
):
    """
    Calculate FID score for two image distributions.
    Args:
        img_dist1 (torch.Tensor): image distribution 1
        img_dist2 (torch.Tensor): image distribution 2
        batch_size (int): batch size
    """
    _ = torch.manual_seed(64)
    fid = FrechetInceptionDistance(feature=2048).to(device=img_dist1.device)
    for i in range(0, len(img_dist1), batch_size):
        fid.update(img_dist1[i:min(i+batch_size, len(img_dist1))], real=True)
        fid.update(img_dist2[i:min(i+batch_size, len(img_dist1))], real=False)
    for i in range(len(img_dist1), len(img_dist2), batch_size):
        fid.update(img_dist2[i:min(i+batch_size, len(img_dist2))], real=False)
    fid_score = fid.compute()
    return fid_score


def calc_mean_scores(
    img_dist,
    true_labels,
    regr_model,
    batch_size=64
):
    _ = torch.manual_seed(64)
    num_samples = img_dist.shape[0]
    predictions = []
    for i in range(0, num_samples, batch_size):
        img_batch = img_dist[i:min(i+batch_size, num_samples)]
        score = regr_model(img_batch.float()).cpu().detach()
        predictions.append(score)
    predictions = torch.concat(predictions, axis=0).to(device=true_labels.device)

    mse = MeanSquaredError().to(device=true_labels.device)
    mse_score = mse(predictions, true_labels).cpu().detach().numpy()
    mae = MeanAbsoluteError().to(device=true_labels.device)
    mae_score = mae(predictions, true_labels).cpu().detach().numpy()

    predictions_df = pd.DataFrame(predictions.numpy())
    true_labels_df = pd.DataFrame(true_labels.numpy())
    corr, _ = stats.pearsonr(predictions_df.values.flatten(), true_labels_df.values.flatten())

    return mse_score, mae_score, corr