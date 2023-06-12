import os, sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch
from torch.distributions import Gamma, Normal


## select the dataset
def set_dataset(dataset_name: str):
    if dataset_name == "mnist-thickness-intensity":
        return ["thickness", "intensity"]
    elif dataset_name == "mnist-thickness-slant":
        return ["thickness", "slant"]
    else:
        raise ValueError("Dataset name not found")

### groud truth model (space)
class MorphoSampler:
    def __init__(
        self,
        dataset_name: str,
        use_groud_truth: bool = False,
        label_path: str = None,
    ):
        self.dataset_name = dataset_name
        self.use_ground_truth = use_groud_truth
        self.vars = set_dataset(self.dataset_name)
        self.label_path = label_path
        if use_groud_truth:
            self.model = model(self.dataset_name, self.use_ground_truth)
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
                                      self.vars, self.df)
        
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
                            num_samples=num_samples, include_numbers=include_numbers)
            samples = preprocess_samples(*samples, dataset_name=self.dataset_name)
            return samples
        else:
            return sample_gt(self.dataset_name, self.model, 
                          num_samples=num_samples, include_numbers=include_numbers)
    
    def sampling_from_learned_space(
        self, num_samples: int = 100,
        include_numbers: bool = False, normalize: bool = False
    ):
        assert self.use_ground_truth == False
        if normalize:
            samples = sample_from_model(self.dataset_name, self.model,
                                        num_samples=num_samples, include_numbers=include_numbers)
            samples = preprocess_samples(*samples, dataset_name=self.dataset_name)
            return samples
        else:
            return sample_from_model(self.dataset_name, self.model,
                                        num_samples=num_samples, include_numbers=include_numbers)
    def sampling_slant(
        self, thickness, normalize: bool = False,
        model_=None
    ):
        assert self.dataset_name == "mnist-thickness-slant"
        thickness = thickness.reshape(-1, 1)
        if self.use_ground_truth:
            ## estimate the slant from the thickness
            normal_dist = self.model["normal"].sample((thickness.shape[0],))
            slant = 56 * torch.tanh(normal_dist + thickness - 2.5)
            slant = slant.reshape(-1, 1)
        else:
            normal = Normal(loc=torch.zeros_like(torch.tensor(self.model["slant_mu"])),
                        scale=torch.tensor(self.model["slant_std"]))
            slant = thickness @ self.model["slant_weights"].T + self.model["slant_bias"] + normal.sample((thickness.shape[0], 1))
        if normalize:
            thickness = (thickness - model_["thickness_mu"].reshape(-1, 1)) / model_["thickness_std"].reshape(-1, 1)
            slant = (slant - model_["slant_mu"].reshape(-1, 1)) / model_["slant_std"].reshape(-1, 1)
        return thickness, slant
    
    def sampling_intensity(
        self, thickness, normalize: bool = False,
        model_=None
    ):
        assert self.dataset_name == "mnist-thickness-intensity"
        thickness = thickness.reshape(-1, 1)
        if self.use_ground_truth:
            ## estimate the intensity from the thickness
            normal_dist = self.model["normal"].sample((thickness.shape[0],))
            intensity = 191 * torch.sigmoid(normal_dist + 2 * thickness - 5) + 64
            intensity = intensity.reshape(-1, 1)
        else:
            normal = Normal(loc=torch.zeros_like(torch.tensor(self.model["intensity_mu"])),
                        scale=torch.tensor(self.model["intensity_std"]))
            intensity = thickness @ self.model["intensity_weights"].T + self.model["intensity_bias"] + normal.sample((thickness.shape[0], 1))
        if normalize:
            thickness = (thickness - model_["thickness_mu"].reshape(-1, 1)) / model_["thickness_std"].reshape(-1, 1)
            intensity = (intensity - model_["intensity_mu"].reshape(-1, 1)) / model_["intensity_std"].reshape(-1, 1)
        return thickness, intensity

def model(dataset_name: str, use_ground_truth: bool = False):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]
    if use_ground_truth:
        if dataset_name == "mnist-thickness-intensity":
            gamma_dist = Gamma(torch.tensor([10.]), torch.tensor([5.]))
            norm_dist = Normal(torch.tensor([0.0]), torch.tensor([0.5]))
        elif dataset_name == "mnist-thickness-slant":
            gamma_dist = Gamma(torch.tensor([10.]), torch.tensor([5.]))
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
    include_numbers: bool = False
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]
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

    if include_numbers:
        classes = torch.randint(0, 10, size=(num_samples,))
        classes = torch.nn.functional.one_hot(classes, num_classes=10)
    return thickness.reshape(-1, 1), intensity, slant, classes

def sample_from_model(
    dataset_name: str,
    model,
    num_samples: int = 100,
    include_numbers: bool = False
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]
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
    if dataset_name == "mnist-thickness-intensity":
        normal = Normal(loc=torch.zeros_like(torch.tensor(model["intensity_mu"])),
                        scale=torch.tensor(model["intensity_std"]))
        intensity = thickness @ model["intensity_weights"].T + model["intensity_bias"] + normal.sample((num_samples, 1))
        return thickness, intensity.reshape(-1, 1), None, classes
    elif dataset_name == "mnist-thickness-slant":
        normal = Normal(loc=torch.zeros_like(torch.tensor(model["slant_mu"])),
                        scale=torch.tensor(model["slant_std"]))
        slant = thickness @ model["slant_weights"].T + model["slant_bias"] + normal.sample((num_samples, 1))
        return thickness, None, slant.reshape(-1, 1), classes

## learned model (latent space) using MLE estimation
def estimate_mle(
    dataset_name: str,
    vars: list,
    df: pd.DataFrame ## with the data
    
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]
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
    if dataset_name == "mnist-thickness-intensity":
        ## intensity
        intensity = df[vars[1]].values
        thickness = torch.sigmoid(torch.tensor(thickness)).cpu().detach().numpy()
        intensity_weights, intensity_bias = linear_regression(intensity.reshape(-1, 1), thickness.reshape(-1, 1))
        model.update(
            intensity_weights = intensity_weights,
            intensity_bias = intensity_bias,
            intensity_mu = intensity.mean(),
            intensity_std = intensity.std(),
        )
    elif dataset_name == "mnist-thickness-slant":
        ## slant
        slant = df[vars[1]].values
        thickness = torch.sigmoid(torch.tensor(thickness)).cpu().detach().numpy()
        slant_weights, slant_bias = linear_regression(slant.reshape(-1, 1), thickness.reshape(-1, 1))
        model.update(
            slant_weights = slant_weights,
            slant_bias = slant_bias,
            slant_mu = slant.mean(),
            slant_std = slant.std(),
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
    return lr.coef_, lr.intercept_

def get_data(
    vars: list,
    path: str = None, 
    image_fnames: list = None
):
    if path.split("/")[-2] != "trainset":
        raise ValueError("path must be the trainset folder")
    label_file = os.path.join(path, "dataset.json")
    with open(label_file, "rb") as f:
        labels = json.load(f)["labels"]
    labels = dict(labels)
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
    dataset_name: str=None
):
    assert dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]
    ## gamma normalized dist
    thickness = (thickness - thickness.mean()) / thickness.std()
    if dataset_name == "mnist-thickness-intensity":
        intensity = (intensity - intensity.mean()) / intensity.std()
        samples = torch.cat([thickness, intensity], 1)
    elif dataset_name == "mnist-thickness-slant":
        slant = (slant - slant.mean()) / slant.std()
        samples = torch.cat([thickness, slant], 1)
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

# def sample_new(
#     dataset: str, 
#     model,
#     num_samples: int = 100,
#     include_numbers: bool = False
# ):
#     if dataset == "mnist-thickness-intensity":
#         thickness = 0.5 + model["gamma"].sample((num_samples, 1))
#         normal_dist = model["normal"].sample((num_samples, 1))
#         intensity = 191 * torch.sigmoid(normal_dist + 2 * thickness - 5) + 64
#         intensity = intensity.reshape(-1, 1)
#         slant = None
#     elif dataset == "mnist-thickness-slant":
#         thickness = model["gamma"].sample((num_samples, 1))
#         normal_dist = model["normal_slant"].sample((num_samples, 1))
#         slant = normal_dist + (20 * thickness - 50) * 5
#         slant = slant.reshape(-1, 1)
#         intensity = None
#     classes = torch.randint(0, 10, size=(num_samples,)) if include_numbers else None
#     classes = torch.nn.functional.one_hot(classes, num_classes=10) if include_numbers else None
#     return thickness.reshape(-1, 1), intensity, slant, classes



        
