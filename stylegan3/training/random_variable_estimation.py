"""
Random Variable Estimation:
Used for Hybrid training with two datasets
To estimate the missing values on the other dataset
"""
### ------------------- ###
### --- Third-Party --- ###
### ------------------- ###
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal, Gamma

### ------------------- ###
### --- Own Package --- ###
### ------------------- ###
from latent_mle import sample_from_gmm_custom

class RandomEstimation:
    """
        Given the labels from two datasets, and find the missing values for each dataset
        Setup for using the MRI datasets (UKB and ADNI)
    Args:
        labels0 (np.ndarray): the labels of the first dataset
        labels1 (np.ndarray): the labels of the second dataset
        causal_models (list): causal models for two datasets
    """
    def __init__(
        self, 
        labels0,
        labels1,
        causal_samplers
    ):
        self.labels0 = labels0
        self.labels1 = labels1
        self.total_num = min(self.labels0.shape[0], self.labels1.shape[0])
        self.causal_samplers = causal_samplers
        self.causal_models = [
            self.causal_samplers[0].get_causal_model(), self.causal_samplers[1].get_causal_model()
        ]
    def random_var_estimate(self, current_labels, for_which_ds: str):
        """
        Args:
            current_labels (np.ndarray): the labels used for the current step
            for_which_ds (str) : ukb or adni
        """
        if for_which_ds == "ukb": ## should give back ventricle, brain, intracranial
            cdr = torch.tensor(self.causal_models[1]["cdr_encoder"].inverse_transform(current_labels[:, 2:5]).reshape(-1, 1))
            ## VOL
            mvn = Normal(
            loc=torch.zeros_like(self.model_find_label0["ventricle_bias"]),
            scale=self.model_find_label0["ventricle_sigma"],
            )
            vent_x = torch.cat([
                current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1),
                cdr, current_labels[:, 5].reshape(-1, 1),
                current_labels[:, 6].reshape(-1, 1)], dim=-1)
            ventricle = vent_x @ self.model_find_label0["ventricle_weights"].T + self.model_find_label0["ventricle_bias"] + mvn.sample((current_labels.shape[0],))

            mvn = Normal(
            loc=torch.zeros_like(self.model_find_label0["brain_bias"]),
            scale=self.model_find_label0["brain_sigma"],
            )
            brain_x = torch.cat([
                current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1),
                cdr, current_labels[:, 7].reshape(-1, 1), 
                current_labels[:, 8].reshape(-1, 1)
            ], dim=-1)
            brain = brain_x @ self.model_find_label0["brain_weights"].T + self.model_find_label0["brain_bias"] + mvn.sample((current_labels.shape[0],))

            mvn = Normal(
            loc=torch.zeros_like(self.model_find_label0["intracranial_bias"]),
            scale=self.model_find_label0["intracranial_sigma"],
            )
            intracranial_x = torch.cat([
                current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1),
                cdr, current_labels[:, 5].reshape(-1, 1), current_labels[:, 6].reshape(-1, 1),
                current_labels[:, 7].reshape(-1, 1), current_labels[:, 8].reshape(-1, 1),
                current_labels[:, 9].reshape(-1, 1), current_labels[:, 10].reshape(-1, 1)
            ], dim=-1)
            intracranial = intracranial_x @ self.model_find_label0["intracranial_weights"].T + self.model_find_label0["intracranial_bias"] + mvn.sample((current_labels.shape[0],))
            return torch.cat([current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1), 
                ventricle, brain, intracranial], dim=-1)

        elif for_which_ds == "adni": ## should give back CDR, VOL
            original_cdr = self.causal_models[1]["cdr_encoder"].inverse_transform(self.labels1[:, 2:5]).reshape(-1)    
            prob_cdr = [np.count_nonzero(original_cdr == i) / original_cdr.shape[0] for i in np.unique(original_cdr)]
            cdr = np.random.choice([0, 0.5, 1], size=(current_labels.shape[0], 1), p=prob_cdr)
            cdr = torch.tensor(self.causal_models[1]["cdr_encoder"].transform(cdr).toarray())
            ## VOL
            mvn = MultivariateNormal(
            loc=torch.zeros_like(self.model_find_label1["ventricle_bias"]),
            covariance_matrix=self.model_find_label1["ventricle_sigma"],
            )
            # vent_x = torch.cat([
            #     current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1),
            #     current_labels[:, 2].reshape(-1, 1), current_labels[:, -1].reshape(-1, 1)
            # ], dim=-1)
            ventricle = current_labels @ self.model_find_label1["ventricle_weights"].T + self.model_find_label1["ventricle_bias"] + mvn.sample((current_labels.shape[0],))

            mvn = MultivariateNormal(
            loc=torch.zeros_like(self.model_find_label1["cortex_bias"]),
            covariance_matrix=self.model_find_label1["cortex_sigma"],
            )
            # cortex_x = torch.cat([
            #     current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1),
            #     current_labels[:, 3].reshape(-1, 1), current_labels[:, -1].reshape(-1, 1)
            # ], dim=-1)
            cortex = current_labels @ self.model_find_label1["cortex_weights"].T + self.model_find_label1["cortex_bias"] + mvn.sample((current_labels.shape[0],))
            ## hippocampus
            # hippo_left = torch.tensor(np.random.choice(self.labels1[:, 9], size=(current_labels.shape[0], 1)))
            # hippo_right = torch.tensor(np.random.choice(self.labels1[:, 10], size=(current_labels.shape[0], 1)))
            mvn = MultivariateNormal(
                loc=torch.zeros_like(self.model_find_label1["hippocampus_bias"]),
                covariance_matrix=self.model_find_label1["hippocampus_sigma"],
            )
            hippocampus = current_labels @ self.model_find_label1["hippocampus_weights"].T + self.model_find_label1["hippocampus_bias"] + mvn.sample((current_labels.shape[0],))
            return torch.cat([current_labels[:, 0].reshape(-1, 1), current_labels[:, 1].reshape(-1, 1), 
                cdr, ventricle, cortex, hippocampus], dim=-1)
        else:
            raise NotImplementedError

    def volume_regression_find_labels1(self):
        ### Assume find ADNI (left/right ventricle (5-7), cortex (7-9), hippocampus(9-11))
        ### Input X would be from UKB (ventricle (2), brain (3), intracranial (4)) [Sex (0), Age (1), Volumes(2:5)]        
        ## shuffle 
        labels0 = self.labels0[torch.randperm(self.labels0.shape[0])][:self.total_num]
        labels1 = self.labels1[torch.randperm(self.labels1.shape[0])][:self.total_num]
        ## ventricle
        y = labels1[:, 5:7]
        lin_regr =self.mv_linear_regression(y, labels0)
        vol_weights = torch.tensor(lin_regr[0], dtype=torch.float)
        vol_bias = torch.tensor(lin_regr[1], dtype=torch.float)
        vol_sigma = torch.tensor(lin_regr[-1], dtype=torch.float)
        self.model_find_label1 = {
            "ventricle_weights": vol_weights,
            "ventricle_bias" : vol_bias,
            "ventricle_sigma" : vol_sigma
        }
        ## cortex
        y = labels1[:, 7:9]
        # x = np.concatenate([
        #     labels0[:, 0].reshape(-1, 1), labels0[:, 1].reshape(-1, 1), 
        #     labels0[:, 3].reshape(-1, 1), labels0[:, -1].reshape(-1, 1)], axis=-1)
        lin_regr =self.mv_linear_regression(y, labels0)
        vol_weights = torch.tensor(lin_regr[0], dtype=torch.float)
        vol_bias = torch.tensor(lin_regr[1], dtype=torch.float)
        vol_sigma = torch.tensor(lin_regr[-1], dtype=torch.float)
        self.model_find_label1.update({
            "cortex_weights": vol_weights,
            "cortex_bias" : vol_bias,
            "cortex_sigma" : vol_sigma
        })
        ## hippocampus
        y = labels1[:, 9:11]
        # x = np.concatenate([
        #     labels0[:, 0].reshape(-1, 1), labels0[:, 1].reshape(-1, 1), 
        #     labels0[:, 3].reshape(-1, 1), labels0[:, -1].reshape(-1, 1)], axis=-1)
        lin_regr =self.mv_linear_regression(y, labels0)
        vol_weights = torch.tensor(lin_regr[0], dtype=torch.float)
        vol_bias = torch.tensor(lin_regr[1], dtype=torch.float)
        vol_sigma = torch.tensor(lin_regr[-1], dtype=torch.float)
        self.model_find_label1.update({
            "hippocampus_weights": vol_weights,
            "hippocampus_bias" : vol_bias,
            "hippocampus_sigma" : vol_sigma
        })


    def volume_regression_find_labels0(self):
        ### Assume find UKB (ventricle (2), brain (3), intracranial (4)) [Sex (0), Age (1), Volumes(2:5)]
        ### from ADNI as Input (X) (left/right ventricle, cortex, hippocampus) [Sex (0), Age (1), CDR(2:5), Volumes(5:11)]
        ### Volumes [left ventricle (5), right ventricle (6), left cortex (7), right cortex (8), hippocampus left (9), hippocampus right (10)]
        ## shuffle
        labels0 = self.labels0[torch.randperm(self.labels0.shape[0])][:self.total_num]
        labels1 = self.labels1[torch.randperm(self.labels1.shape[0])][:self.total_num]

        original_cdr = self.causal_models[1]["cdr_encoder"].inverse_transform(self.labels1[:, 2:5]).reshape(-1, 1)
        ## ventricle
        y = labels0[:, 2].reshape(-1, 1)
        x = np.concatenate([
            labels1[:, 0].reshape(-1, 1), labels1[:, 1].reshape(-1, 1), 
            original_cdr, labels1[:, 5].reshape(-1, 1), labels1[:, 6].reshape(-1, 1)], 
            axis=-1
        )
        lin_regr =self.mv_linear_regression(y, x)
        vol_weights = torch.tensor(lin_regr[0], dtype=torch.float)
        vol_bias = torch.tensor(lin_regr[1], dtype=torch.float)
        vol_sigma = torch.tensor(lin_regr[-1], dtype=torch.float)
        self.model_find_label0 = {
            "ventricle_weights": vol_weights,
            "ventricle_bias" : vol_bias,
            "ventricle_sigma" : vol_sigma
        }
        ## brain
        y = labels0[:, 3].reshape(-1, 1)
        x = np.concatenate([
            labels1[:, 0].reshape(-1, 1), labels1[:, 1].reshape(-1, 1),
            original_cdr, labels1[:, 7].reshape(-1, 1),
            labels1[:, 8].reshape(-1, 1)], 
            axis=-1
        )
        lin_regr =self.mv_linear_regression(y, x)
        vol_weights = torch.tensor(lin_regr[0], dtype=torch.float)
        vol_bias = torch.tensor(lin_regr[1], dtype=torch.float)
        vol_sigma = torch.tensor(lin_regr[-1], dtype=torch.float)
        self.model_find_label0.update({
            "brain_weights": vol_weights,
            "brain_bias" : vol_bias,
            "brain_sigma" : vol_sigma
        })
        ## intracranial
        y = labels0[:, 4].reshape(-1, 1)
        x = np.concatenate([
            labels1[:, 0].reshape(-1, 1), labels1[:, 1].reshape(-1, 1),
            original_cdr, labels1[:, 5].reshape(-1, 1), labels1[:, 6].reshape(-1, 1),
            labels1[:, 7].reshape(-1, 1), labels1[:, 8].reshape(-1, 1),
            labels1[:, 9].reshape(-1, 1), labels1[:, 10].reshape(-1, 1)],
            axis=-1
        )
        lin_regr =self.mv_linear_regression(y, x)
        vol_weights = torch.tensor(lin_regr[0], dtype=torch.float)
        vol_bias = torch.tensor(lin_regr[1], dtype=torch.float)
        vol_sigma = torch.tensor(lin_regr[-1], dtype=torch.float)
        self.model_find_label0.update({
            "intracranial_weights": vol_weights,
            "intracranial_bias" : vol_bias,
            "intracranial_sigma" : vol_sigma
        })

    def mv_linear_regression(self, y, x):
        Sigma = np.cov(y.T, ddof=1)
        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)

        W = np.linalg.solve(x.T @ x, x.T @ y)
        b = W[-1, :]
        W = W[:-1, :].T
        return W, b, Sigma
    
def linear_regression(y, x):
    ### Assume Morpho-MNIST (relationship between thickness and intensity/slant)
    Sigma = np.cov(y.T, ddof=1)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)

    W = np.linalg.solve(x.T @ x, x.T @ y)
    b = W[-1, :]
    W = W[:-1, :].T

    latent_weights = torch.tensor(W, dtype=torch.float)
    latent_bias = torch.tensor(b, dtype=torch.float)
    latent_sigma = torch.tensor(Sigma, dtype=torch.float)
    model_find_label = {
        "weights": latent_weights,
        "bias" : latent_bias,
        "sigma" : latent_sigma
    }
    mvn = Normal(
        loc=torch.zeros_like(model_find_label["bias"]),
        scale=model_find_label["sigma"],
    )
    return model_find_label, mvn

### Thickness and intensity
def intensity_estimation(thickness): ## thickness: (batch_size, 1)
    normal_dist = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.5]))
    normal_sample = normal_dist.sample((thickness.shape[0],)).to(thickness.device)
    intensity = 191 * torch.sigmoid(normal_sample + 2 * thickness - 5) + 64
    return intensity.reshape(-1, 1)

### Thickness and slant
def slant_estimation(thickness): ## thickness: (batch_size, 1)
    normal_dist = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.5]))
    normal_sample = normal_dist.sample((thickness.shape[0],)).to(thickness.device)
    slant = normal_sample + 20 * thickness - 50
    return slant.reshape(-1, 1)

### Thickness and slant
def slant_estimation_new(thickness): ## thickness: (batch_size, 1)
    normal_dist = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([2.0]))
    normal_sample = normal_dist.sample((thickness.shape[0],)).to(thickness.device)
    slant = normal_sample + (20 * thickness - 50) * 5
    return slant.reshape(-1, 1)

### MRI 
### UKB
### Volumes (ventricle, brain, intracranial) condition on age and sex
### ADNI
### volumes (left/right ventricle, cortex and hippocampus) condition on age and sex and CDR
def volumes_estimation(sex, age, causalmodel, cdr=None, normalize=True):
    """
    Volumes estimatio for Sex and Age
    Args:
        sex : 0 and 1 (Binary Variable)
        age : unnormalized age
        causalmodel (_type_): adni or ukb causal model
        cdr : only for adni. Defaults to None. not one hot encoding
        normalize (bool): return normalized volumes Defaults to True.

    Returns:
        _type_: _description_
    """
    assert "vol_gmm" in causalmodel.keys()
    device = sex.device
    if cdr is not None:
        if cdr.shape[-1] != 1:
            cdr = torch.tensor(causalmodel["cdr_encoder"].inverse_transform(
                cdr.cpu().detach().numpy()).reshape(-1, 1)).to(cdr.device)
        x = torch.cat([age, sex, cdr], dim=-1).cpu().detach()
    else:
        x = torch.cat([age, sex], dim=-1).cpu().detach()
    volumes = sample_from_gmm_custom(
        x=x, indices=causalmodel["vol_indices"],
        gmm=causalmodel["vol_gmm"]
    )
    if causalmodel["vol_model"] == "lognormal":
        volumes = torch.exp(volumes)
    if normalize:
        volumes = (volumes - causalmodel["vol_means"]) / causalmodel["vol_std"]
    return volumes.to(device)

### ADNI
### CDR condition on age and sex
def cdr_estimation(sex, age, causalmodel, normalize=True):
    assert "cdr_weights" in causalmodel.keys()
    assert "cdr_bias" in causalmodel.keys()
    device = sex.device
    x = torch.cat([age, sex], dim=-1).cpu().detach()
    cdr_prob = torch.softmax(x @ causalmodel["cdr_weights"].T + causalmodel["cdr_bias"], dim=-1)
    cdr = causalmodel["cdr_classes"][torch.multinomial(cdr_prob, num_samples=1)]
    if normalize:
        cdr = torch.tensor(causalmodel["cdr_encoder"].transform(cdr).toarray())
    return cdr.to(device)
