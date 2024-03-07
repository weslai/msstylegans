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
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import BinarySpecificity, BinaryRecall

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

def calc_kid_score(
    img_dist1,
    img_dist2,
    batch_size=64
):
    """
    Calculate KID score for two image distributions.
    Args:
        img_dist1 (torch.Tensor): image distribution 1
        img_dist2 (torch.Tensor): image distribution 2
        batch_size (int): batch size
    """
    _ = torch.manual_seed(64)
    kid = KernelInceptionDistance(feature=2048, subset_size=batch_size).to(device=img_dist1.device)
    for i in range(0, len(img_dist1), batch_size):
        kid.update(img_dist1[i:min(i+batch_size, len(img_dist1))], real=True)
        kid.update(img_dist2[i:min(i+batch_size, len(img_dist1))], real=False)
    for i in range(len(img_dist1), len(img_dist2), batch_size):
        kid.update(img_dist2[i:min(i+batch_size, len(img_dist2))], real=False)
    kid_score = kid.compute()
    return kid_score

def cal_inception_score(
    img_dist,
    batch_size=64
):
    """
    Calculate inception score for generated image distributions.
    Args:
        img_dist (torch.Tensor): generated image distribution
        batch_size (int): batch size
    """
    _ = torch.manual_seed(64)
    inception = InceptionScore().to(device=img_dist.device)
    for i in range(0, len(img_dist), batch_size):
        inception.update(img_dist[i:min(i+batch_size, len(img_dist))])
    inception_score = inception.compute()
    return inception_score

def cal_lpips_score(
    img_dist1,
    img_dist2,
    batch_size=64
):
    """
    Calculate LPIPS score for two image distributions.
    This must be a pair of images.
    Args:
        img_dist1 (torch.tensor): generated image distribution [0, 1]
        img_dist2 (torch.tensor): ground truth image distribution [0, 1]
        batch_size (int, optional): _description_. Defaults to 64.
    """
    _ = torch.manual_seed(64)
    assert img_dist1.shape == img_dist2.shape
    scores = []
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device=img_dist1.device)
    for i in range(0, len(img_dist1), batch_size):
        img1 = img_dist1[i:min(i+batch_size, len(img_dist1))]
        img2 = img_dist2[i:min(i+batch_size, len(img_dist2))]
        score = lpips(img1, img2)
        scores.append(score)
    return (torch.mean(torch.cat(scores)).cpu().detach().numpy(), torch.std(torch.cat(scores)).cpu().detach().numpy())

def calc_mean_scores(
    genimg_dist,
    realimg_dist,
    true_labels,
    regr_model,
    covariate,
    batch_size=64
):
    """
    This func is to calculate the mean scores of the predictions of real images and the prediction of synthetic images 
    from the regression model. After this, the mae and mse scores are calculated between the two predictions.
    """
    _ = torch.manual_seed(64)
    assert len(genimg_dist) == len(realimg_dist) == len(true_labels)
    num_samples = genimg_dist.shape[0]
    gen_predictions = []
    real_predictions = []
    for i in range(0, num_samples, batch_size):
        genimg_batch = genimg_dist[i:min(i+batch_size, num_samples)]
        realimg_batch = realimg_dist[i:min(i+batch_size, num_samples)]
        gen_score = regr_model(genimg_batch.float()).cpu().detach()
        real_score = regr_model(realimg_batch.float()).cpu().detach()
        gen_predictions.append(gen_score)
        real_predictions.append(real_score)
    gen_predictions = torch.concat(gen_predictions, axis=0).to(device=true_labels.device)
    real_predictions = torch.concat(real_predictions, axis=0).to(device=true_labels.device)

    # if covariate in ["cataract", "disease_risk", "MH", "TSLN", "apoe4", "level"]:
    if covariate in ["disease_risk", "MH", "TSLN", "apoe4", "level"]:
        if covariate != "apoe4":
            accuracy = Accuracy(task="binary").to(device=true_labels.device)
            precision = Precision(task="binary").to(device=true_labels.device)
            recall = Recall(task="binary").to(device=true_labels.device)
            f1 = F1Score(task="binary").to(device=true_labels.device)
            binaryRecall = BinaryRecall().to(device=true_labels.device)
            binarySpecificity = BinarySpecificity().to(device=true_labels.device)
            gen_predictions = torch.sigmoid(gen_predictions)
            gen_predictions = torch.where(gen_predictions > 0.5, 1, 0).reshape(-1, 1)
            real_predictions = torch.sigmoid(real_predictions)
            real_predictions = torch.where(real_predictions > 0.5, 1, 0).reshape(-1, 1)
            specificity_score = binarySpecificity(gen_predictions, true_labels).cpu().detach().numpy()
            sensitivity_score = binaryRecall(gen_predictions, true_labels).cpu().detach().numpy()
            balanced_accuracy_score = (specificity_score + sensitivity_score) / 2
            gen_corr = np.corrcoef(gen_predictions.flatten().cpu().detach().numpy(), 
                                   true_labels.flatten().cpu().detach().numpy())[0, 1]
        else:
            ncls = 3
            accuracy = Accuracy(task="multiclass", num_classes=ncls).to(device=true_labels.device)
            precision = Precision(task="multiclass", num_classes=ncls).to(device=true_labels.device)
            recall = Recall(task="multiclass", num_classes=ncls).to(device=true_labels.device)
            f1 = F1Score(task="multiclass", num_classes=ncls).to(device=true_labels.device)
            gen_predictions = torch.softmax(gen_predictions, dim=1)
            gen_predictions = torch.argmax(gen_predictions, dim=1)
            real_predictions = torch.softmax(real_predictions, dim=1)
            real_predictions = torch.argmax(real_predictions, dim=1)
        accuracy_score = accuracy(gen_predictions, true_labels).cpu().detach().numpy()
        precision_score = precision(gen_predictions, true_labels).cpu().detach().numpy()
        recall_score = recall(gen_predictions, true_labels).cpu().detach().numpy()
        f1_score = f1(gen_predictions, true_labels).cpu().detach().numpy()
        # accuracy_score = accuracy(gen_predictions, real_predictions).cpu().detach().numpy()
        # precision_score = precision(gen_predictions, real_predictions).cpu().detach().numpy()
        # recall_score = recall(gen_predictions, real_predictions).cpu().detach().numpy()
        # f1_score = f1(gen_predictions, real_predictions).cpu().detach().numpy()
        
    else:
        mse = MeanSquaredError().to(device=true_labels.device)
        mse_score = mse(gen_predictions, real_predictions).cpu().detach().numpy()
        mae = MeanAbsoluteError().to(device=true_labels.device)
        mae_score = mae(gen_predictions, real_predictions).cpu().detach().numpy()

    gen_predictions = gen_predictions.cpu().detach().numpy()
    real_predictions = real_predictions.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()
    gen_predictions_df = pd.DataFrame(gen_predictions, columns=["gen_predict"])
    real_predictions_df = pd.DataFrame(real_predictions, columns=["real_predict"])
    true_labels_df = pd.DataFrame(true_labels, columns=["labels"])
    if covariate not in ["cataract", "disease_risk", "MH", "TSLN", "apoe4", "level"]:
        corr, _ = stats.pearsonr(gen_predictions_df.values.flatten(), real_predictions_df.values.flatten())
    elif covariate == "cataract":
        corr, _ = stats.pearsonr(gen_predictions_df.values.flatten(), true_labels_df.values.flatten())
    predictions_df = pd.concat([gen_predictions_df, real_predictions_df, true_labels_df], axis=1)
    # if covariate in ["cataract", "disease_risk", "MH", "TSLN", "apoe4", "level"]:
    if covariate in ["disease_risk", "MH", "TSLN", "apoe4", "level"]:
        return (accuracy_score, precision_score), (recall_score, f1_score), (gen_corr, balanced_accuracy_score), predictions_df
    else:
        return mse_score, mae_score, corr, predictions_df

def calc_mean_scores_disc(
    genimg_dist,
    realimg_dist,
    true_labels,
    covariance,
    regr_model,
    discriminator = None,
    batch_size=64
):
    """
    This func is to calculate the mean scores of the predictions of real images from the regression model 
    and the predictions of generated images from the discriminator.

    """
    _ = torch.manual_seed(64)
    assert len(genimg_dist) == len(realimg_dist) == len(true_labels)
    num_samples = genimg_dist.shape[0]
    gen_predictions = []
    real_predictions = []
    for i in range(0, num_samples, batch_size):
        genimg_batch = genimg_dist[i:min(i+batch_size, num_samples)]
        realimg_batch = realimg_dist[i:min(i+batch_size, num_samples)]
        labels = true_labels[i:min(i+batch_size, num_samples)]

        gen_score = discriminator(genimg_batch.float(), labels).cpu().detach()[:, covariance + 1]
        real_score = regr_model(realimg_batch.float()).cpu().detach()
        gen_predictions.append(gen_score.reshape(-1, 1))
        real_predictions.append(real_score)
    gen_predictions = torch.concat(gen_predictions, axis=0).to(device=true_labels.device)
    real_predictions = torch.concat(real_predictions, axis=0).to(device=true_labels.device)

    mse = MeanSquaredError().to(device=true_labels.device)
    mse_score = mse(gen_predictions, real_predictions).cpu().detach().numpy()
    mae = MeanAbsoluteError().to(device=true_labels.device)
    mae_score = mae(gen_predictions, real_predictions).cpu().detach().numpy()

    gen_predictions = gen_predictions.cpu().detach().numpy()
    real_predictions = real_predictions.cpu().detach().numpy()
    gen_predictions_df = pd.DataFrame(gen_predictions, columns=["gen_predict"])
    real_predictions_df = pd.DataFrame(real_predictions, columns=["real_predict"])
    corr, _ = stats.pearsonr(gen_predictions_df.values.flatten(), real_predictions_df.values.flatten())

    predictions_df = pd.concat([gen_predictions_df, real_predictions_df], axis=1)
    return mse_score, mae_score, corr, predictions_df

def calc_prediction_disc(
    disc_gen_img_dist,
    disc_real_img_dist,
    genimg_dist,
    realimg_dist,
    true_labels,
    covariance,
    regr_model,
    discriminator = None,
    batch_size=64
):
    """
    This func is to calculate the mean scores of the predictions of real images and synthetic images from the regression model 
    and the same predictions from the discriminator.

    This is to check whether the discriminator from GANs can be used as a regression model or
    at least predict similar results as the regression model.
    """
    _ = torch.manual_seed(64)
    assert len(genimg_dist) == len(realimg_dist) == len(true_labels)
    num_samples = genimg_dist.shape[0]
    gen_discriminations = []
    real_discriminations = []
    gen_predictions = []
    real_predictions = []
    for i in range(0, num_samples, batch_size):
        disc_genimg_batch = disc_gen_img_dist[i:min(i+batch_size, num_samples)]
        disc_realimg_batch = disc_real_img_dist[i:min(i+batch_size, num_samples)]
        genimg_batch = genimg_dist[i:min(i+batch_size, num_samples)]
        realimg_batch = realimg_dist[i:min(i+batch_size, num_samples)]
        labels = true_labels[i:min(i+batch_size, num_samples)]
        ## gen scores
        gen_disc = discriminator(disc_genimg_batch.float(), labels).cpu().detach()[:, covariance + 1]
        gen_pred = regr_model(genimg_batch.float()).cpu().detach()
        ## real scores
        real_disc = discriminator(disc_realimg_batch.float(), labels).cpu().detach()[:, covariance + 1]
        real_pred = regr_model(realimg_batch.float()).cpu().detach()
        
        gen_discriminations.append(gen_disc.reshape(-1, 1))
        gen_predictions.append(gen_pred)
        real_discriminations.append(real_disc.reshape(-1, 1))
        real_predictions.append(real_pred)

    gen_discriminations = torch.concat(gen_discriminations, axis=0).to(device=true_labels.device)
    gen_predictions = torch.concat(gen_predictions, axis=0).to(device=true_labels.device)
    real_discriminations = torch.concat(real_discriminations, axis=0).to(device=true_labels.device)
    real_predictions = torch.concat(real_predictions, axis=0).to(device=true_labels.device)
    ## real scores
    mse = MeanSquaredError().to(device=true_labels.device)
    mse_real = mse(real_discriminations, real_predictions).cpu().detach().numpy()
    mae = MeanAbsoluteError().to(device=true_labels.device)
    mae_real = mae(real_discriminations, real_predictions).cpu().detach().numpy()
    ## gen scores
    mse = MeanSquaredError().to(device=true_labels.device)
    mse_gen = mse(gen_discriminations, gen_predictions ).cpu().detach().numpy()
    mae = MeanAbsoluteError().to(device=true_labels.device)
    mae_gen = mae(gen_discriminations, gen_predictions).cpu().detach().numpy()

    real_discriminations = real_discriminations.cpu().detach().numpy()
    real_predictions = real_predictions.cpu().detach().numpy()
    gen_discriminations = gen_discriminations.cpu().detach().numpy()
    gen_predictions = gen_predictions.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()

    real_discriminations_df = pd.DataFrame(real_discriminations, columns=["real_disc"])
    real_predictions_df = pd.DataFrame(real_predictions, columns=["real_predict"])
    gen_discriminations_df = pd.DataFrame(gen_discriminations, columns=["gen_disc"])
    gen_predictions_df = pd.DataFrame(gen_predictions, columns=["gen_predict"])
    true_labels_df = pd.DataFrame(true_labels, columns=["labels"])

    real_corr, _ = stats.pearsonr(real_discriminations_df.values.flatten(),
                             real_predictions_df.values.flatten())
    gen_corr, _ = stats.pearsonr(gen_discriminations_df.values.flatten(),
                             gen_predictions_df.values.flatten())

    scores_df = pd.concat([real_discriminations_df, real_predictions_df, 
                           gen_discriminations_df, gen_predictions_df,
                           true_labels_df], axis=1)
    return (mae_real, mse_real), (mae_gen, mse_gen), (real_corr, gen_corr), scores_df ## (real, gen), scores_df