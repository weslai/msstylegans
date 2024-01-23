import numpy as np
import torch
from typing import Tuple
from utils import generate_images
from evaluations.eval_utils import calc_mean_scores

def eval_loss_real_ms(
    data_name: str,
    covariates: dict,
    source_gan: str,
    Gen,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float, float],
    rotate: float,
    regr_model0,
    regr_model1 = None,
    regr_model2 = None,
    dataset1: torch.utils.data.Dataset = None,
    dataset2: torch.utils.data.Dataset = None,
    dataset3: torch.utils.data.Dataset = None,
    sampler_sources = None
):
    if source_gan == "single":
        target_dataset = dataset1
    else:
        target_dataset = dataset1 if data_name in ["ukb", "retinal"] else dataset2 if data_name in ["adni", "rfmid"] else dataset3
    cov_dict = {}
    if source_gan != "single":
        if data_name == "ukb" or data_name == "adni" or data_name == "nacc":
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
    scores_dict = {}
    strata_predictions_dict = {}
    regr_ml = {}
    for key, value in covariates["cov"].items():
        scores_dict[key] = [] ## mse, mae
        strata_predictions_dict[key] = []
        if value == 0:
            regr_ml[key] = regr_model0
        elif value == 1:
            regr_ml[key] = regr_model1
        elif value == 2:
            regr_ml[key] = regr_model2
    #### 
    if len(covariates["cov"]) == 3: ## [ukb, retinal, adni, rfmid]
        gen_imgs = []
        real_imgs = []
        cov_labels = []
        ## get samples from GANs
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, Gen.z_dim).to(device)
            source1_c = []
            source1_img = []
            for idx in np.random.choice(range(len(target_dataset)), batch_size):
                source1_c.append(target_dataset.get_norm_label(idx))
                source1_img.append(torch.tensor(target_dataset[idx][0]))
            l = torch.from_numpy(np.stack(source1_c, axis=0)).to(device)
            imgs = torch.stack(source1_img, dim=0).repeat([1,1,1,1]).to(device)
            if source_gan == "multi_mri":
                if len(sampler_sources) == 3: ## three sources
                    if data_name == "ukb":
                        age = l[:, 0] * torch.tensor(dataset1.model["age_max"] - dataset1.model["age_min"]) + torch.tensor(dataset1.model["age_min"])
                        gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                        gen_c4 = sampler_sources["nacc"].sampling_given_age(age.cpu().detach(), normalize=True)
                        c_source = torch.tensor([1, 0, 0] * age.shape[0]).reshape(age.shape[0], -1)
                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), l[:, 1:].cpu().detach(), 
                                            gen_c3[:, 1:], gen_c4[:, 1:], c_source], dim=1).to(device)
                    elif data_name == "adni":
                        age = l[:, 0] * torch.tensor(dataset2.model["age_max"] - dataset2.model["age_min"]) + torch.tensor(dataset2.model["age_min"])
                        gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                        gen_c4 = sampler_sources["nacc"].sampling_given_age(age.cpu().detach(), normalize=True)
                        c_source = torch.tensor([0, 1, 0] * age.shape[0]).reshape(age.shape[0], -1)
                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                            l[:, 1:].cpu().detach(), gen_c4[:, 1:], c_source], dim=1).to(device)
                elif len(sampler_sources) == 2: ## two sources
                    if data_name == "ukb":
                        age = l[:, 0] * torch.tensor(dataset1.model["age_max"] - dataset1.model["age_min"]) + torch.tensor(dataset1.model["age_min"])
                        gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                        c_source = torch.tensor([0] * age.shape[0]).reshape(age.shape[0], -1)
                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), l[:, 1:].cpu().detach(), 
                                            gen_c3[:, 1:], c_source], dim=1).to(device)
                    elif data_name == "adni":
                        age = l[:, 0] * torch.tensor(dataset2.model["age_max"] - dataset2.model["age_min"]) + torch.tensor(dataset2.model["age_min"])
                        gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                        c_source = torch.tensor([1] * age.shape[0]).reshape(age.shape[0], -1)
                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                            l[:, 1:].cpu().detach(), c_source], dim=1).to(device)
            elif source_gan == "multi_retina":
                if len(sampler_sources) == 3:
                    if data_name == "retinal":
                        gen_c3 = sampler_sources["eyepacs"].sampling_model(l.shape[0])[-1]
                        gen_c4 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                        c_source = torch.tensor([1, 0, 0] * l.shape[0]).reshape(l.shape[0], -1)
                        all_c = torch.cat([l.cpu().detach(), gen_c3, gen_c4, c_source], dim=1).to(device)
                    elif data_name == "rfmid":
                        gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                        gen_c3 = sampler_sources["eyepacs"].sampling_model(l.shape[0])[-1]
                        c_source = torch.tensor([0, 0, 1] * l.shape[0]).reshape(l.shape[0], -1)
                        all_c = torch.cat([gen_c2, gen_c3, l.cpu().detach(), c_source], dim=1).to(device)
                elif len(sampler_sources) == 2:
                    if data_name == "retinal":
                        gen_c3 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                        c_source = torch.tensor([0] * l.shape[0]).reshape(l.shape[0], -1)
                        all_c = torch.cat([l.cpu().detach(), gen_c3, c_source], dim=1).to(device)
                    elif data_name == "rfmid":
                        gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                        c_source = torch.tensor([1] * l.shape[0]).reshape(l.shape[0], -1)
                        all_c = torch.cat([gen_c2, l.cpu().detach(), c_source], dim=1).to(device)
            batch_imgs = generate_images(Gen, z, l[:, :Gen.c_dim] if source_gan == "single" else all_c, 
                                        truncation_psi, 
                                        noise_mode, translate, rotate).permute(0,3,1,2)
            batch_imgs = batch_imgs.div(255).cpu().detach()
            imgs = imgs.div(255).cpu().detach()
            gen_imgs.append(batch_imgs)
            real_imgs.append(imgs)
            cov_labels.append(l) ## all covariates
    elif len(covariates["cov"]) == 2: ## [nacc]
        real_imgs = []
        gen_imgs = []
        cov_labels = []
        ## get samples from GANs
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, Gen.z_dim).to(device)
            source1_c = []
            source1_img = []
            for idx in np.random.choice(range(len(target_dataset)), batch_size):
                source1_c.append(target_dataset.get_norm_label(idx)) ## normalized labels
                source1_img.append(torch.tensor(target_dataset[idx][0]))
            l = torch.from_numpy(np.stack(source1_c, axis=0)).to(device)
            imgs = torch.stack(source1_img, dim=0).repeat([1,1,1,1]).to(device)
            if source_gan == "multi_mri":
                if len(sampler_sources) == 3:
                    if data_name == "nacc":
                        age = l[:, 0] * torch.tensor(dataset3.model["age_max"] - dataset3.model["age_min"]) + torch.tensor(dataset3.model["age_min"])
                        gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                        gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                        c_source = torch.tensor([0, 0, 1] * age.shape[0]).reshape(age.shape[0], -1)
                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                            gen_c3[:, 1:], l[:, 1:].cpu().detach(), c_source], dim=1).to(device)
                    else:
                        raise NotImplementedError
            batch_imgs = generate_images(Gen, z, l if source_gan == "single" else all_c,
                                        truncation_psi, 
                                        noise_mode, translate, rotate).permute(0,3,1,2)
            batch_imgs = batch_imgs.div(255).cpu().detach()
            imgs = imgs.div(255).cpu().detach()
            gen_imgs.append(batch_imgs)
            real_imgs.append(imgs)
            cov_labels.append(l) ## all covariates
    elif len(covariates["cov"]) == 1: ## [eyepacs]
        real_imgs = []
        gen_imgs = []
        cov_labels = []
        ## get samples from GANs
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, Gen.z_dim).to(device)
            source1_c = []
            source1_img = []
            for idx in np.random.choice(range(len(target_dataset)), batch_size):
                source1_c.append(target_dataset.get_norm_label(idx))
                source1_img.append(torch.tensor(target_dataset[idx][0]))
            l = torch.from_numpy(np.stack(source1_c, axis=0)).to(device)
            imgs = torch.stack(source1_img, dim=0).repeat([1,1,1,1]).to(device)
            if source_gan == "multi_retina":
                if data_name == "eyepacs":
                    gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                    gen_c3 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                    c_source = torch.tensor([0, 0, 1] * l.shape[0]).reshape(l.shape[0], -1)
                    all_c = torch.cat([gen_c2, gen_c3, l.cpu().detach(), c_source], dim=1).to(device)
                else:
                    raise NotImplementedError
            batch_imgs = generate_images(Gen, z, l if source_gan == "single" else all_c,
                                        truncation_psi, 
                                        noise_mode, translate, rotate).permute(0,3,1,2)
            batch_imgs = batch_imgs.div(255).cpu().detach()
            imgs = imgs.div(255).cpu().detach()
            gen_imgs.append(batch_imgs)
            real_imgs.append(imgs)
            cov_labels.append(l) ## all covariates
    gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
    real_imgs = torch.cat(real_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
    cov_labels = torch.cat(cov_labels, dim=0).to(device)
    ### within strata, calculate mae, mse
    for key, value in covariates["cov"].items():
        cov = value
        ### separate the covariates and regression models
        if key in ["disease_risk", "MH", "TSLN", "level", "apoe4"]:
            metric0, metric1, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                            cov_labels[:, cov].reshape(-1, 1),
                                                            regr_ml[key],
                                                            covariate=key,
                                                            batch_size=64)
            accuracy, precision = metric0[0], metric0[1]
            recall, f1 = metric1[0], metric1[1]
            print(f"ACC: {accuracy}, PRE: {precision}, CORR: {corr}")
            print(f"REC: {recall}, F1: {f1}")
            scores_dict[key].append(np.array([accuracy, precision, recall, f1, corr])) ##, corr
        else:
            mse_err, mae_err, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                            cov_labels[:, cov].reshape(-1, 1),
                                                            regr_ml[key],
                                                            covariate=key,
                                                            batch_size=64)
            print(f"MSE: {mse_err}, MAE: {mae_err}, CORR: {corr}")
        ### save the evaluation analysis to a json file
            scores_dict[key].append(np.array([mse_err, mae_err, corr])) ##mse, mae, corr
        strata_predictions_dict[key].append(scores_df)
    for key, value in covariates["cov"].items():
        scores_dict[key] = np.stack(scores_dict[key], axis=0)
    return scores_dict, strata_predictions_dict


def strata_eval_real_ms(
    data_name: str,
    strata_idxs: list,
    covariates: dict,
    source_gan: str,
    Gen,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float, float],
    rotate: float,
    regr_model0,
    regr_model1 = None,
    regr_model2 = None,
    labels1: np.ndarray = None,
    dataset1: torch.utils.data.Dataset = None,
    dataset2: torch.utils.data.Dataset = None,
    dataset3: torch.utils.data.Dataset = None,
    sampler_sources = None
):
    if source_gan == "single":
        target_dataset = dataset1
    else:
        target_dataset = dataset1 if data_name in ["ukb", "retinal"] else dataset2 if data_name in ["adni", "rfmid"] else dataset3
    cov_dict = {}
    if source_gan != "single":
        if data_name == "ukb" or data_name == "adni" or data_name == "nacc":
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
    strata_hist = covariates["strata_hist"]
    scores_dict = {}
    strata_predictions_dict = {}
    regr_ml = {}
    for key, value in covariates["cov"].items():
        scores_dict[key] = [] ## mse, mae
        strata_predictions_dict[key] = []
        if value == 0:
            regr_ml[key] = regr_model0
        elif value == 1:
            regr_ml[key] = regr_model1
        elif value == 2:
            regr_ml[key] = regr_model2
    ## loop through all strata
    if len(covariates["cov"]) == 3: ## [ukb, retinal, adni, rfmid]
        c1_min, c1_max = covariates["c1_min"], covariates["c1_max"]
        c2_min, c2_max = covariates["c2_min"], covariates["c2_max"]
        c3_min, c3_max = covariates["c3_min"], covariates["c3_max"]
        for stra_c1 in strata_idxs if data_name != "rfmid" else [0, 1]:
            if stra_c1 == 0:
                cur_c1 = (c1_min, strata_hist["c1"][stra_c1]) if data_name != "rfmid" else (0)
            elif stra_c1 == 1:
                cur_c1 = (strata_hist["c1"][stra_c1-1], strata_hist["c1"][stra_c1]) if data_name != "rfmid" else (1)
            else:
                cur_c1 = (strata_hist["c1"][stra_c1-1], c1_max)
            for stra_c2 in strata_idxs if data_name not in ["retinal", "rfmid"] else [0, 1]:
                if stra_c2 == 0:
                    cur_c2 = (c2_min, strata_hist["c2"][stra_c2]) if data_name not in ["retinal", "rfmid"] else (0)
                elif stra_c2 == 1:
                    cur_c2 = (strata_hist["c2"][stra_c2-1], strata_hist["c2"][stra_c2]) if data_name not in  ["retinal", "rfmid"] else (1)
                else:
                    cur_c2 = (strata_hist["c2"][stra_c2-1], c2_max)
                for stra_c3 in strata_idxs if data_name != "rfmid" else [0, 1]:
                    if stra_c3 == 0:
                        cur_c3 = (c3_min, strata_hist["c3"][stra_c3]) if data_name != "rfmid" else (0)
                    elif stra_c3 == 1:
                        cur_c3 = (strata_hist["c3"][stra_c3-1], strata_hist["c3"][stra_c3]) if data_name != "rfmid" else (1)
                    else:
                        cur_c3 = (strata_hist["c3"][stra_c3-1], c3_max)
                    real_imgs = []
                    gen_imgs = []
                    cov_labels = []
                    if data_name == "retinal":
                        idxs1 = np.where((labels1[:,0] >= cur_c1[0]) & (labels1[:,0] < cur_c1[1]) & \
                                        (labels1[:,1] == cur_c2[0]) & \
                                        (labels1[:,2] >= cur_c3[0]) & (labels1[:,2] < cur_c3[1]))[0]
                    elif data_name == "rfmid":
                        idxs1 = np.where((labels1[:,0] == cur_c1[0]) & \
                                        (labels1[:,1] == cur_c2[0]) & \
                                        (labels1[:,2] == cur_c3[0]))[0]
                    else:
                        idxs1 = np.where((labels1[:,0] >= cur_c1[0]) & (labels1[:,0] < cur_c1[1]) & \
                                        (labels1[:,1] >= cur_c2[0]) & (labels1[:,1] < cur_c2[1]) & \
                                        (labels1[:,2] >= cur_c3[0]) & (labels1[:,2] < cur_c3[1]))[0]
                    num_real = len(idxs1)
                    if num_real >= 5:
                        ## get samples from GANs
                        for _ in range(num_samples // batch_size):
                            z = torch.randn(batch_size, Gen.z_dim).to(device)
                            source1_c = []
                            source1_img = []
                            for idx in np.random.choice(idxs1, batch_size):
                                source1_c.append(target_dataset.get_norm_label(idx))
                                source1_img.append(torch.tensor(target_dataset[idx][0]))
                            l = torch.from_numpy(np.stack(source1_c, axis=0)).to(device)
                            imgs = torch.stack(source1_img, dim=0).repeat([1,1,1,1]).to(device)
                            if source_gan == "multi_mri":
                                if len(sampler_sources) == 3: ## three sources
                                    if data_name == "ukb":
                                        age = l[:, 0] * torch.tensor(dataset1.model["age_max"] - dataset1.model["age_min"]) + torch.tensor(dataset1.model["age_min"])
                                        gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                                        gen_c4 = sampler_sources["nacc"].sampling_given_age(age.cpu().detach(), normalize=True)
                                        c_source = torch.tensor([1, 0, 0] * age.shape[0]).reshape(age.shape[0], -1)
                                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), l[:, 1:].cpu().detach(), 
                                                           gen_c3[:, 1:], gen_c4[:, 1:], c_source], dim=1).to(device)
                                    elif data_name == "adni":
                                        age = l[:, 0] * torch.tensor(dataset2.model["age_max"] - dataset2.model["age_min"]) + torch.tensor(dataset2.model["age_min"])
                                        gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                                        gen_c4 = sampler_sources["nacc"].sampling_given_age(age.cpu().detach(), normalize=True)
                                        c_source = torch.tensor([0, 1, 0] * age.shape[0]).reshape(age.shape[0], -1)
                                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                                           l[:, 1:].cpu().detach(), gen_c4[:, 1:], c_source], dim=1).to(device)
                                elif len(sampler_sources) == 2: ## two sources
                                    if data_name == "ukb":
                                        age = l[:, 0] * torch.tensor(dataset1.model["age_max"] - dataset1.model["age_min"]) + torch.tensor(dataset1.model["age_min"])
                                        gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                                        c_source = torch.tensor([0] * age.shape[0]).reshape(age.shape[0], -1)
                                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), l[:, 1:].cpu().detach(), 
                                                           gen_c3[:, 1:], c_source], dim=1).to(device)
                                    elif data_name == "adni":
                                        age = l[:, 0] * torch.tensor(dataset2.model["age_max"] - dataset2.model["age_min"]) + torch.tensor(dataset2.model["age_min"])
                                        gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                                        c_source = torch.tensor([1] * age.shape[0]).reshape(age.shape[0], -1)
                                        age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                                        all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                                           l[:, 1:].cpu().detach(), c_source], dim=1).to(device)
                            elif source_gan == "multi_retina":
                                if len(sampler_sources) == 3:
                                    if data_name == "retinal":
                                        gen_c3 = sampler_sources["eyepacs"].sampling_model(l.shape[0])[-1]
                                        gen_c4 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                                        c_source = torch.tensor([1, 0, 0] * l.shape[0]).reshape(l.shape[0], -1)
                                        all_c = torch.cat([l.cpu().detach(), gen_c3, gen_c4, c_source], dim=1).to(device)
                                    elif data_name == "rfmid":
                                        gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                                        gen_c3 = sampler_sources["eyepacs"].sampling_model(l.shape[0])[-1]
                                        c_source = torch.tensor([0, 0, 1] * l.shape[0]).reshape(l.shape[0], -1)
                                        all_c = torch.cat([gen_c2, gen_c3, l.cpu().detach(), c_source], dim=1).to(device)
                            batch_imgs = generate_images(Gen, z, l if source_gan == "single" else all_c, 
                                                        truncation_psi, 
                                                        noise_mode, translate, rotate).permute(0,3,1,2)
                            batch_imgs = batch_imgs.div(255).cpu().detach()
                            imgs = imgs.div(255).cpu().detach()
                            gen_imgs.append(batch_imgs)
                            real_imgs.append(imgs)
                            cov_labels.append(l) ## all covariates
                        gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                        real_imgs = torch.cat(real_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                        cov_labels = torch.cat(cov_labels, dim=0).to(device)
                        ### within strata, calculate mae, mse
                        for key, value in covariates["cov"].items():
                            cov = value
                            ### separate the covariates and regression models
                            if key in ["cataract", "disease_risk", "MH", "TSLN"]:
                                metric0, metric1, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                                                cov_labels[:, cov].reshape(-1, 1),
                                                                                regr_ml[key],
                                                                                covariate=key,
                                                                                batch_size=64)
                                accuracy, precision = metric0
                                recall, f1 = metric1
                                print(f"strata: {cur_c1}, {cur_c2}, {cur_c3}, ACC: {accuracy}, PRE: {precision}, CORR: {corr}")
                                print(f"strata: {cur_c1}, {cur_c2}, {cur_c3}, REC: {recall}, F1: {f1}")
                                scores_dict[key].append(np.array([num_real, cur_c1[0], cur_c1[1], cur_c2[0], cur_c2[1],
                                    cur_c3[0], cur_c3[1], accuracy, precision, recall, f1, corr])) ##, corr
                            else:
                                mse_err, mae_err, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                                                cov_labels[:, cov].reshape(-1, 1),
                                                                                regr_ml[key],
                                                                                covariate=key,
                                                                                batch_size=64)
                                print(f"strata: {cur_c1}, {cur_c2}, {cur_c3}, MSE: {mse_err}, MAE: {mae_err}, CORR: {corr}")
                            ### save the evaluation analysis to a json file
                                scores_dict[key].append(np.array([num_real, cur_c1[0], cur_c1[1], cur_c2[0], cur_c2[1],
                                    cur_c3[0], cur_c3[1], mse_err, mae_err, corr])) ##mse, mae, corr
                            strata_predictions_dict[key].append(scores_df)
    elif len(covariates["cov"]) == 2: ## [nacc]
        c1_min, c1_max = covariates["c1_min"], covariates["c1_max"]
        c2_min, c2_max = covariates["c2_min"], covariates["c2_max"]
        for stra_c1 in strata_idxs:
            if stra_c1 == 0:
                cur_c1 = (c1_min, strata_hist["c1"][stra_c1])
            elif stra_c1 == 1:
                cur_c1 = (strata_hist["c1"][stra_c1-1], strata_hist["c1"][stra_c1])
            else:
                cur_c1 = (strata_hist["c1"][stra_c1-1], c1_max)
            for stra_c2 in strata_idxs if data_name != "nacc" else [0, 1, 2]:
                if stra_c2 == 0:
                    cur_c2 = (c2_min, strata_hist["c2"][stra_c2]) if data_name != "nacc" else (0)
                elif stra_c2 == 1:
                    cur_c2 = (strata_hist["c2"][stra_c2-1], strata_hist["c2"][stra_c2]) if data_name != "nacc" else (1)
                else:
                    cur_c2 = (strata_hist["c2"][stra_c2-1], c2_max) if data_name != "nacc" else (2)
                real_imgs = []
                gen_imgs = []
                cov_labels = []
                ## get samples from datasets (idxs)
                idxs1 = np.where((labels1[:,0] >= cur_c1[0]) & (labels1[:,0] < cur_c1[1]) & \
                                (labels1[:,1] == cur_c2[0]))[0]
                num_real = len(idxs1)
                if num_real >= 5:
                    ## get samples from GANs
                    for _ in range(num_samples // batch_size):
                        z = torch.randn(batch_size, Gen.z_dim).to(device)
                        ### 
                        source1_c = []
                        source1_img = []
                        for idx in np.random.choice(idxs1, batch_size):
                            source1_c.append(target_dataset.get_norm_label(idx)) ## normalized labels
                            source1_img.append(torch.tensor(target_dataset[idx][0]))
                        l = torch.from_numpy(np.stack(source1_c, axis=0)).to(device)
                        imgs = torch.stack(source1_img, dim=0).repeat([1,1,1,1]).to(device)

                        if source_gan == "multi_mri":
                            if len(sampler_sources) == 3:
                                if data_name == "nacc":
                                    age = l[:, 0] * torch.tensor(dataset3.model["age_max"] - dataset3.model["age_min"]) + torch.tensor(dataset3.model["age_min"])
                                    gen_c2 = sampler_sources["ukb"].sampling_given_age(age.cpu().detach(), normalize=True)
                                    gen_c3 = sampler_sources["adni"].sampling_given_age(age.cpu().detach(), normalize=True)
                                    c_source = torch.tensor([0, 0, 1] * age.shape[0]).reshape(age.shape[0], -1)
                                    age_norm = (age - cov_dict["age_min"]) / (cov_dict["age_max"] - cov_dict["age_min"])
                                    all_c = torch.cat([age_norm.reshape(-1, 1).cpu().detach(), gen_c2[:, 1:], 
                                                       gen_c3[:, 1:], l[:, 1:].cpu().detach(), c_source], dim=1).to(device)
                                else:
                                    raise NotImplementedError
                        batch_imgs = generate_images(Gen, z, l if source_gan == "single" else all_c,
                                                    truncation_psi, 
                                                    noise_mode, translate, rotate).permute(0,3,1,2)
                        batch_imgs = batch_imgs.div(255).cpu().detach()
                        imgs = imgs.div(255).cpu().detach()
                        gen_imgs.append(batch_imgs)
                        real_imgs.append(imgs)
                        cov_labels.append(l) ## all covariates
                    gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                    real_imgs = torch.cat(real_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                    cov_labels = torch.cat(cov_labels, dim=0).to(device)
                    ### within strata, calculate mae, mse
                    for key, value in covariates["cov"].items():
                        cov = value
                        ### separate the covariates and regression models
                        if key in ["apoe4"]:
                            metric0, metric1, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                                            cov_labels[:, cov].reshape(-1, 1),
                                                                            regr_ml[key],
                                                                            covariate=key,
                                                                            batch_size=64)
                            accuracy, precision = metric0
                            recall, f1 = metric1
                            print(f"strata: {cur_c1}, {cur_c2}, ACC: {accuracy}, PRE: {precision}, CORR: {corr}")
                            print(f"strata: {cur_c1}, {cur_c2}, REC: {recall}, F1: {f1}")
                            scores_dict[key].append(np.array([num_real, cur_c1[0], cur_c1[1], cur_c2[0],
                                                            accuracy, precision, recall, f1, corr])) ##, corr
                        else:
                            mse_err, mae_err, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                                            cov_labels[:, cov].reshape(-1, 1),
                                                                            regr_ml[key],
                                                                            covariate=key,
                                                                            batch_size=64)
                            print(f"strata: {cur_c1}, {cur_c2}, MSE: {mse_err}, MAE: {mae_err}, CORR: {corr}")
                        ### save the evaluation analysis to a json file
                            scores_dict[key].append(np.array([num_real, cur_c1[0], cur_c1[1], cur_c2[0],
                                mse_err, mae_err, corr])) ##mse, mae, corr
                        strata_predictions_dict[key].append(scores_df)
    elif len(covariates["cov"]) == 1: ## [eyepacs]
        c1_min, c1_max = covariates["c1_min"], covariates["c1_max"]
        for stra_c1 in strata_idxs:
            if stra_c1 == 0:
                cur_c1 = (c1_min)
            elif stra_c1 == 1:
                cur_c1 = (1)
            real_imgs = []
            gen_imgs = []
            cov_labels = []
            ## get samples from datasets (idxs)
            idxs1 = np.where(labels1[:,0] == cur_c1[0])[0]
            num_real = len(idxs1)
            if num_real >= 5:
                ## get samples from GANs
                for _ in range(num_samples // batch_size):
                    z = torch.randn(batch_size, Gen.z_dim).to(device)
                    source1_c = []
                    source1_img = []
                    for idx in np.random.choice(idxs1, batch_size):
                        source1_c.append(target_dataset.get_norm_label(idx))
                        source1_img.append(torch.tensor(target_dataset[idx][0]))
                    l = torch.from_numpy(np.stack(source1_c, axis=0)).to(device)
                    imgs = torch.stack(source1_img, dim=0).repeat([1,1,1,1]).to(device)
                    if source_gan == "multi_retina":
                        if data_name == "eyepacs":
                            gen_c2 = sampler_sources["retinal"].sample_normalize(l.shape[0])
                            gen_c4 = sampler_sources["rfmid"].sampling_model(l.shape[0])[-1]
                            c_source = torch.tensor([0, 1, 0] * l.shape[0]).reshape(l.shape[0], -1)
                            all_c = torch.cat([gen_c2, l.cpu().detach(), gen_c4, c_source], dim=1).to(device)
                        else:
                            raise NotImplementedError
                    batch_imgs = generate_images(Gen, z, l if source_gan == "single" else all_c,
                                                truncation_psi, 
                                                noise_mode, translate, rotate).permute(0,3,1,2)
                    
                    batch_imgs = batch_imgs.div(255).cpu().detach()
                    imgs = imgs.div(255).cpu().detach()
                    gen_imgs.append(batch_imgs)
                    real_imgs.append(imgs)
                    cov_labels.append(l) ## all covariates
                gen_imgs = torch.cat(gen_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                real_imgs = torch.cat(real_imgs, dim=0).repeat([1,1,1,1]).to(device)## (batch_size, channel, pixel, pixel)
                cov_labels = torch.cat(cov_labels, dim=0).to(device)
                ### within strata, calculate mae, mse
                for key, value in covariates["cov"].items():
                    cov = value
                    ### separate the covariates and regression models
                    if key in ["level"]:
                        metric0, metric1, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                                        cov_labels[:, cov].reshape(-1, 1),
                                                                        regr_ml[key],
                                                                        covariate=key,
                                                                        batch_size=64)
                        accuracy, precision = metric0
                        recall, f1 = metric1
                        print(f"strata: {cur_c1}, ACC: {accuracy}, PRE: {precision}, CORR: {corr}")
                        print(f"strata: {cur_c1}, REC: {recall}, F1: {f1}")
                        scores_dict[key].append(np.array([num_real, cur_c1[0],
                            accuracy, precision, recall, f1, corr])) ##, corr
                    else:
                        mse_err, mae_err, corr, scores_df = calc_mean_scores(gen_imgs, real_imgs, 
                                                                        cov_labels[:, cov].reshape(-1, 1),
                                                                        regr_ml[key],
                                                                        covariate=key,
                                                                        batch_size=64)
                        print(f"strata: {cur_c1}, MSE: {mse_err}, MAE: {mae_err}, CORR: {corr}")
                    ### save the evaluation analysis to a json file
                        scores_dict[key].append(np.array([num_real, cur_c1[0],
                            mse_err, mae_err, corr])) ##mse, mae, corr
                    strata_predictions_dict[key].append(scores_df)
    for key, value in covariates["cov"].items():
        scores_dict[key] = np.stack(scores_dict[key], axis=0)
    return scores_dict, strata_predictions_dict