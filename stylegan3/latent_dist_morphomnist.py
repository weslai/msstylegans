import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import torch
from torch.distributions import Gamma, Normal


## select the dataset
def set_dataset(dataset_name: str, which_source=None):
    if dataset_name == "mnist-thickness-intensity":
        return ["thickness", "intensity"]
    elif dataset_name == "mnist-thickness-slant":
        return ["thickness", "slant"]
    elif dataset_name == "mnist-thickness-intensity-slant":
        if which_source == "source1":
            return ["thickness", "intensity", "label"]
        elif which_source == "source2":
            return ["thickness", "slant", "label"]
    else:
        raise ValueError("Dataset name not found")

### groud truth model (space)
class MorphoSampler:
    def __init__(
        self,
        dataset_name: str,
        use_groud_truth: bool = False,
        label_path: str = None,
        which_source: str = None
    ):
        self.dataset_name = dataset_name
        self.use_ground_truth = use_groud_truth
        self.label_path = label_path
        self.which_source = which_source
        self.vars = set_dataset(self.dataset_name, self.which_source)
        if use_groud_truth:
            self.model = model(self.dataset_name, self.use_ground_truth, self.which_source)
        else:
            assert self.label_path is not None
            if not self.label_path.endswith("/"):
                self.label_path += "/"
            if self.label_path.split("/")[-2] != "trainset":
                assert self.label_path.split("/")[-2] == "valset" or self.label_path.split("/")[-2] == "testset"
                temp_path = self.label_path.split("/")[:-2]
                temp_path = os.path.join(*temp_path)
                self.label_path = "/" + os.path.join(temp_path, "trainset/")
            self._get_all_fnames()
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == ".png")
            self.df = get_data(self.vars, self.label_path, self._image_fnames)
            ## model learned from the data
            self.model = estimate_mle(self.dataset_name, 
                                      self.vars, self.df,
                                      self.which_source)
        
    def _get_all_fnames(self):
        if os.path.isdir(self.label_path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.label_path) for root, _dirs, files in os.walk(self.label_path) for fname in files}
        else:
            raise IOError('Path must point to a directory or zip')
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def sampling_from_gt(
        self, num_samples: int = 100, 
        include_numbers: bool = False, normalize: bool = False
    ):
        assert self.use_ground_truth == True
        if normalize:
            samples = sample_gt(self.dataset_name, self.model,
                            num_samples=num_samples, include_numbers=include_numbers,
                            which_source=self.which_source)
            samples = preprocess_samples(*samples, dataset_name=self.dataset_name, which_source=self.which_source)
            return samples
        else:
            return sample_gt(self.dataset_name, self.model, 
                          num_samples=num_samples, include_numbers=include_numbers,
                          which_source=self.which_source)
    
    def sampling_from_learned_space(
        self, num_samples: int = 100,
        include_numbers: bool = False, normalize: bool = False
    ):
        assert self.use_ground_truth == False
        if normalize:
            samples = sample_from_model(self.dataset_name, self.model,
                                        num_samples=num_samples, include_numbers=include_numbers,
                                        which_source=self.which_source)
            samples = preprocess_samples(*samples, dataset_name=self.dataset_name, which_source=self.which_source)
            return samples
        else:
            return sample_from_model(self.dataset_name, self.model,
                                    num_samples=num_samples, include_numbers=include_numbers,
                                    which_source=self.which_source)
    def sampling_slant(
        self, thickness, normalize: bool = False,
        model_=None
    ):
        assert self.dataset_name in ["mnist-thickness-slant", "mnist-thickness-intensity-slant"]
        if self.dataset_name == "mnist-thickness-intensity-slant":
            assert self.which_source == "source2"
        thickness = thickness.reshape(-1, 1)
        if self.use_ground_truth:
            ## estimate the slant from the thickness
            normal_dist = self.model["normal"].sample((thickness.shape[0],))
            slant = 56 * torch.tanh(normal_dist + thickness - 2.5)
            slant = slant.reshape(-1, 1)
        else:
            thickness_normal = Normal(loc=torch.zeros_like(torch.tensor(self.model["slant_mu"])),
                            scale=torch.tensor(self.model["residual_std"]))
            temp_thickness = torch.tensor(thickness) + thickness_normal.sample((thickness.shape[0], 1))
            slant = sigmoid(temp_thickness, *self.model["slant_model"])
            slant = torch.tensor(slant, dtype=torch.float32).reshape(-1, 1)
        if normalize:
            thickness = (thickness - model_["thickness_mu"].reshape(-1, 1)) / model_["thickness_std"].reshape(-1, 1)
            slant = (slant - model_["slant_mu"].reshape(-1, 1)) / model_["slant_std"].reshape(-1, 1)
        return thickness, slant
    
    def sampling_intensity(
        self, thickness, normalize: bool = False,
        model_=None
    ):
        assert self.dataset_name in ["mnist-thickness-intensity", "mnist-thickness-intensity-slant"]
        if self.dataset_name == "mnist-thickness-intensity-slant":
            assert self.which_source == "source1"
        thickness = thickness.reshape(-1, 1)
        if self.use_ground_truth:
            ## estimate the intensity from the thickness
            normal_dist = self.model["normal"].sample((thickness.shape[0],))
            intensity = 191 * torch.sigmoid(normal_dist + 2 * thickness - 5) + 64
            intensity = intensity.reshape(-1, 1)
        else:
            thickness_normal = Normal(loc=torch.zeros_like(torch.tensor(self.model["intensity_mu"])),
                                    scale=torch.tensor(self.model["residual_std"]))
            temp_thickness = torch.tensor(thickness) + thickness_normal.sample((thickness.shape[0], 1))
            intensity = sigmoid(temp_thickness, *self.model["intensity_model"])
            intensity = torch.tensor(intensity, dtype=torch.float32).reshape(-1, 1)
        if normalize:
            thickness = (thickness - model_["thickness_mu"].reshape(-1, 1)) / model_["thickness_std"].reshape(-1, 1)
            intensity = (intensity - model_["intensity_mu"].reshape(-1, 1)) / model_["intensity_std"].reshape(-1, 1)
        return thickness, intensity

def model(dataset_name: str, use_ground_truth: bool = False, which_source: str = None):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant", "mnist-thickness-intensity-slant"]
    if use_ground_truth:
        if dataset_name == "mnist-thickness-intensity":
            gamma_dist = Gamma(torch.tensor([10.]), torch.tensor([5.]))
            norm_dist = Normal(torch.tensor([0.0]), torch.tensor([0.5]))
        elif dataset_name == "mnist-thickness-slant":
            gamma_dist = Gamma(torch.tensor([10.]), torch.tensor([5.]))
            norm_dist = Normal(torch.tensor([0.0]), torch.tensor([0.3]))
        elif dataset_name == "mnist-thickness-intensity-slant":
            gamma_dist = Gamma(torch.tensor([10.]), torch.tensor([5.]))
            if which_source == "source1":
                norm_dist = Normal(torch.tensor([0.0]), torch.tensor([0.5]))
            else:
                norm_dist = Normal(torch.tensor([0.0]), torch.tensor([0.3]))
        model = dict(
            gamma = gamma_dist,
            normal = norm_dist
        )
    return model

def sample_gt(
    dataset_name: str, 
    model,
    num_samples: int = 100,
    include_numbers: bool = False,
    which_source: str = None
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant", "mnist-thickness-intensity-slant"]
    classes = None
    if dataset_name == "mnist-thickness-intensity":
        thickness = 0.5 + model["gamma"].sample((num_samples, 1))
        normal_dist = model["normal"].sample((num_samples, 1))
        intensity = 191 * torch.sigmoid(normal_dist + 2 * thickness - 5) + 64
        intensity = intensity.reshape(-1, 1)
        slant = None
    elif dataset_name == "mnist-thickness-slant":
        thickness = 0.5 + model["gamma"].sample((num_samples, 1))
        normal_dist = model["normal"].sample((num_samples, 1))
        slant = 56 * torch.tanh(normal_dist + thickness - 2.5)
        slant = slant.reshape(-1, 1)
        intensity = None
    elif dataset_name == "mnist-thickness-intensity-slant":
        thickness = 0.5 + model["gamma"].sample((num_samples, 1))
        if which_source == "source1":
            intensity_normal_dist = model["normal"].sample((num_samples, 1))
            intensity = 191 * torch.sigmoid(intensity_normal_dist + 2 * thickness - 5) + 64
            intensity = intensity.reshape(-1, 1)
            slant = None
        else:
            slant_normal_dist = model["normal"].sample((num_samples, 1))
            slant = 56 * torch.tanh(slant_normal_dist + thickness - 2.5)
            slant = slant.reshape(-1, 1)
            intensity = None

    if include_numbers:
        classes = torch.randint(0, 10, size=(num_samples,))
        classes = torch.nn.functional.one_hot(classes, num_classes=10)
    return thickness.reshape(-1, 1), intensity, slant, classes

def sample_from_model(
    dataset_name: str,
    model,
    num_samples: int = 100,
    include_numbers: bool = False,
    which_source: str = None
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant",
                            "mnist-thickness-intensity-slant"]
    classes = None
    if include_numbers:
        classes = torch.randint(0, 10, size=(num_samples,))
        classes = torch.nn.functional.one_hot(classes, num_classes=10)
        
    if model["thickness_model"] == "beta":
        thickness = torch.tensor(
            model["min_thickness"]
            + (model["max_thickness"] - model["min_thickness"])
            * stats.beta.rvs(
                model["alpha_thickness"], model["beta_thickness"], 
                size=(num_samples, 1)
            ),
            dtype=torch.float32
        )
        normal = Normal(loc=torch.zeros_like(torch.tensor(model["thickness_mu"])),
                        scale=torch.tensor(model["thickness_std"]))
        temp_thickness = thickness + normal.sample((thickness.shape[0], 1))
    if dataset_name == "mnist-thickness-intensity":
        intensity = temp_thickness @ model["intensity_weights"].T + model["intensity_bias"]
        return thickness, intensity.reshape(-1, 1), None, classes
    elif dataset_name == "mnist-thickness-slant":
        slant = temp_thickness @ model["slant_weights"].T + model["slant_bias"]
        return thickness, None, slant.reshape(-1, 1), classes
    elif dataset_name == "mnist-thickness-intensity-slant":
        if which_source == "source1":
            thickness_normal = Normal(loc=torch.zeros_like(torch.tensor(model["intensity_mu"])),
                            scale=torch.tensor(model["residual_std"]))
            temp_thickness = thickness + thickness_normal.sample((thickness.shape[0], 1))
            intensity = sigmoid(temp_thickness, *model["intensity_model"])
            intensity = torch.tensor(intensity, dtype=torch.float32).reshape(-1, 1)
            slant = None
        else:
            thickness_normal = Normal(loc=torch.zeros_like(torch.tensor(model["slant_mu"])),
                            scale=torch.tensor(model["residual_std"]))
            temp_thickness = thickness + thickness_normal.sample((thickness.shape[0], 1))
            slant = sigmoid(temp_thickness, *model["slant_model"])
            slant = torch.tensor(slant, dtype=torch.float32).reshape(-1, 1)
            intensity = None
        return thickness, intensity, slant, classes

## learned model (latent space) using MLE estimation
def estimate_mle(
    dataset_name: str,
    vars: list,
    df: pd.DataFrame, ## with the data
    which_source: str = None
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant", "mnist-thickness-intensity-slant"]
    ## thickness
    thickness = df[vars[0]].values
    alpha_thickness, beta_thickness = estimate_beta1d(
        thickness, method="moments", eps=1e-4, 
        lo=thickness.min(), hi=thickness.max())
    
    model = dict(
        alpha_thickness = alpha_thickness,
        beta_thickness = beta_thickness,
        min_thickness = thickness.min(),
        max_thickness = thickness.max(),
        thickness_mu = thickness.mean(),
        thickness_std = thickness.std(),
        thickness_model = "beta",
    )
    if dataset_name == "mnist-thickness-intensity" or (dataset_name == "mnist-thickness-intensity-slant" and which_source == "source1"):
        ## intensity
        intensity = df[vars[1]].values
        p0 = [1, 0, np.mean(intensity), np.mean(intensity)] # this is an mandatory initial guess

        popt, pcov = curve_fit(sigmoid, thickness, intensity, p0=p0)
        intensity_hat = sigmoid(thickness, *popt)
        residual = inverse_sigmoid(intensity_hat, *popt) - inverse_sigmoid(intensity, *popt)
        residual_std = residual.std()
        model.update(
            intensity_model = popt,
            intensity_cov = pcov,
            intensity_mu = intensity.mean(),
            intensity_std = intensity.std(),
            residual_std = residual_std,
        )
    elif dataset_name =="mnist-thickness-slant" or (dataset_name == "mnist-thickness-intensity-slant" and which_source == "source2"):
        ## slant
        slant = df[vars[1]].values
        p0 = [1, 0, np.max(slant), np.min(slant)] # this is an mandatory initial guess

        popt, pcov = curve_fit(sigmoid, thickness, slant, p0=p0)
        slant_hat = sigmoid(thickness, *popt)
        residual = inverse_sigmoid(slant_hat, *popt) - inverse_sigmoid(slant, *popt)
        residual_std = residual.std()
        model.update(
            slant_model = popt,
            slant_cov = pcov,
            slant_mu = slant.mean(),
            slant_std = slant.std(),
            residual_std = residual_std,
        )
    digits = df[vars[-1]].values
    num_classes = np.unique(digits)
    digit_probs = {}
    for cls in num_classes:
        digit_probs[cls] = df.loc[df[vars[-1]] == cls].shape[0] / df.shape[0]
    model.update(
        digit_probs = digit_probs
    )
    return model

def estimate_beta1d(x, lo=40, hi=70, method="moments", eps=1e-4):
    x = (x - lo + eps) / (hi - lo + 2 * eps) ## min-max normalization with epsilon
    if method == "moments":
        mean = np.mean(x)
        var = np.var(x)
        alpha = mean * ((mean * (1 - mean) / var) - 1)
        beta = (1 - mean) * ((mean * (1 - mean) / var) - 1)
        return alpha, beta
    elif method == "mle":
        gx = np.exp(np.log(x).mean())
        g1mx = np.exp(np.log(1 - x).mean())
        alpha = 0.5 + gx / (2 * (1 - gx - g1mx))
        beta = 0.5 + g1mx / (2 * (1 - gx - g1mx))
        return alpha, beta
    else:
        raise ValueError("method must be moments or mle")

def linear_regression(y, x):
    lr = LinearRegression()
    lr.fit(x, y)
    return lr, lr.coef_, lr.intercept_

def sigmoid(x, a, b, c, d):
    y = c / (1.0 + np.exp(-a*(x-b))) + d
    return y

def inverse_sigmoid(y, a, b, c, d):
    x = -np.log(c / (y - d) - 1) / a + b
    return x

def get_data(
    vars: list,
    path: str = None,
    image_fnames: list = None
):
    if path.split("/")[-2] != "trainset":
        raise ValueError("path must be the trainset folder")
    # label_file = os.path.join(path, f"train_{which_source}.json")
    label_file = os.path.join(path, "dataset.json")
    with open(label_file, "rb") as f:
        # labels = json.load(f)
        labels = json.load(f)["labels"]
    labels = dict(labels)
    # covs_labels = labels["1"]
    # all_keys = list(covs_labels.keys())
    labels = [labels[fname.replace("\\", "/")] for fname in image_fnames] ## a dict
    new_labels = np.zeros(shape=(len(labels), len(vars)), dtype=np.float32)
    for num, l in enumerate(labels):
        temp = [l[var] for var in vars]
        new_labels[num, :] = temp
    new_labels = pd.DataFrame(new_labels, columns=vars)
    new_labels = new_labels[vars].dropna(axis=0)
    return new_labels


def preprocess_samples(
    thickness, intensity=None, slant=None, classes=None, 
    dataset_name: str=None,
    which_source: str = None
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant", 
                            "mnist-thickness-intensity-slant"]
    ## gamma normalized dist
    thickness = (thickness - thickness.mean()) / thickness.std()
    if dataset_name == "mnist-thickness-intensity" or (dataset_name == "mnist-thickness-intensity-slant" and which_source == "source1"):
        intensity = (intensity - intensity.mean()) / intensity.std()
        if dataset_name == "mnist-thickness-intensity":
            samples = torch.cat([thickness, intensity], 1)
    elif dataset_name == "mnist-thickness-slant" or (dataset_name == "mnist-thickness-intensity-slant" and which_source == "source2"):
        slant = (slant - slant.mean()) / slant.std()
        if dataset_name == "mnist-thickness-slant":
            samples = torch.cat([thickness, slant], 1)
    if dataset_name == "mnist-thickness-intensity-slant":
        samples = torch.cat([thickness, intensity], 1) if which_source == "source1" else torch.cat([thickness, slant], 1)
    if classes is not None:
        if classes.shape[1] != 10:
            classes = torch.nn.functional.one_hot(torch.tensor(classes), num_classes=10)
        samples = torch.cat([samples, classes], 1)
    return samples

def normalize_samples(
    thickness, intensity=None, slant=None, classes=None, 
    dataset: str=None, model=None
):
    ## gamma normalized dist
    thickness = (thickness - model["thickness_mu"]) / model["thickness_std"]
    if dataset == "mnist-thickness-intensity":
        intensity = (intensity - model["intensity_mu"]) / model["intensity_std"]
        samples = torch.cat([thickness, intensity], 1)
    elif dataset == "mnist-thickness-slant":
        slant = (slant - model["slant_mu"]) / model["slant_std"]
        samples = torch.cat([thickness, slant], 1)
    if classes is not None:
        if classes.shape[1] != 10:
            classes = torch.nn.functional.one_hot(torch.tensor(classes), num_classes=10)
        samples = torch.cat([samples, classes], 1)
    return samples