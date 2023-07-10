### -------------------
### --- Third-party ---
### -------------------
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from torch.distributions import MultivariateNormal
import torch
import gmr
from gmr.mvn import invert_indices, regression_coefficients, MVN
from gmr.gmm import _safe_probability_density
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

## get settings
def get_settings(dataset: str, which_source: str = None):
    if dataset == "ukb":
        n_components = 10
        age_estimator = "gmm"
        log_volumes = True
    elif dataset == "adni":
        n_components = 8
        age_estimator = "beta"
        log_volumes = False
    elif dataset == "retinal":
        n_components = 13
        age_estimator = "gmm"
        if which_source == "source1":
            log_volumes = True
        elif which_source == "source2": ## have negative values
            log_volumes = True
    return n_components, age_estimator, log_volumes
## select a dataset
def set_dataset(name: str, which_source=None):
    # DATASET = "ukb"
    # DATASET = "oasis"
    ## The Reihfolge (Order) of the vars should be (age, sex, [cdr], VOLS)
    if name == "oasis3":
        ## ------------------------------------------------------
        ## VARS are for Oasis3
        VOLS = ["SubCortGrayVol", "IntraCranialVol", "CortexVol"]
        VARS = ["Age", "M/F", "cdr"] + VOLS
        ## ------------------------------------------------------
    elif name == "ukb":
        ## VARS are for UKB
        if which_source == "source1":
            VOLS = ["grey_matter"]
            # VOLS = ["brain"]
        elif which_source == "source2":
            VOLS = ["ventricle"]
        VARS = ["age"] + VOLS
        ## ------------------------------------------------------
    elif name == "adni":
        # VOLS = ["left_lateral_ventricle", "right_lateral_ventricle",
        #         "left_cerebral_cortex", "right_cerebral_cortex"
        # ]
        VOLS = ["left_lateral_ventricle", "right_lateral_ventricle",
        "left_cerebral_cortex", "right_cerebral_cortex",
        "left_hippocampus", "right_hippocampus"
        ]
        VARS = ["Age", "Sex", "CDGLOBAL"] + VOLS
        # VARS = ["Age", "Sex", "mmse"] + VOLS ## first ignore mmse for now
        ## ------------------------------------------------------
    elif name == "retinal":
        if which_source == "source1":
            VOLS = ["diastolic_bp"]
        elif which_source == "source2":
            VOLS = ["spherical_power_left"]
        VARS = ["age"] + VOLS
    return VARS, VOLS

class CausalSampling:
    def __init__(
        self, 
        dataset,
        label_path: str = None
    ):
        self.dataset = dataset
        if label_path is None:
            ## use default
            if self.dataset == "ukb":
                raise ValueError(f"{self.dataset} must have a label_path")
            elif self.dataset == "adni":
                self.label_path = "/scratch/wei-cheng.lai/adni/T1_3T_coronal_mni_nonlinear_4DT/trainset/"
            elif self.dataset == "retinal":
                raise ValueError(f"{self.dataset} must have a label_path")
            else:
                raise ValueError(f'dataset {self.dataset} not exists')
        else:
            self.label_path = label_path
            if not self.label_path.endswith("/"):
                self.label_path += "/"
            if self.label_path.split("/")[-2] != "trainset":
                assert self.label_path.split("/")[-2] == "valset" or self.label_path.split("/")[-2] == "testset"
                temp_path = self.label_path.split("/")[:-2]
                self.label_path = "/" + os.path.join(*temp_path) + "/trainset/"
        
        self.which_source = self.label_path.split("/")[-3].split("_")[-1]
        print(f"which_source: {self.which_source}")
        assert self.which_source.startswith("source")
        self.vars, self.vols = set_dataset(self.dataset, self.which_source)
        self.n_components, self.age_estimator, self.log_volumes = get_settings(self.dataset, self.which_source)
        ## we only know labels from the training set
        self._get_all_fnames()
        # PIL.Image.init()
        # self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in [PIL.Image.EXTENSION, "gz"])
        if dataset == "ukb":
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == ".gz")
        else: ## retinal
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == ".jpg")
        
    def _get_all_fnames(self):
        if os.path.isdir(self.label_path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.label_path) for root, _dirs, files in os.walk(self.label_path) for fname in files}
        else:
            raise IOError('Path must point to a directory or zip')
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _mu_std_df(self):
        df = self.df.copy()
        if self.log_volumes:
            if self.dataset == "retinal" and self.which_source == "source2": ## ukb
                df[self.vols[0] + "_shift"] = np.log(df[self.vols] + 1e2)
                df["shift_scale"] = 1e2
            else:
                df[self.vols] = np.log(df[self.vols])
        return {"mu": df.mean(), "std": df.std()}

    def get_graph(self):
        self.df = get_data(self.dataset, self.vars, self.label_path, self._image_fnames)
        self.mu_std_df = self._mu_std_df()
        return self.df
        
    def get_causal_model(self):
        self.model = estimate_mle(
            df = self.get_graph(),
            dataset=self.dataset,
            vars=self.vars,
            vols=self.vols,
            volumes_as_gmm=True,
            n_components=self.n_components,
            age_estimator=self.age_estimator,
            log_volumes=self.log_volumes
        )
        return self.model

    def sampling_model(self, num_samples: int):
        return sample_from_model(self.dataset, self.model, num_samples)
    
    def sample_normalize(self, num_samples: int):
        return preprocess_samples(
            *sample_from_model(self.dataset, self.model, num_samples),
            model=self.model,
            dataset=self.dataset
        )
    def sampling_given_age(self, age, normalize: bool = False):
        if normalize:
            ages, volumes = sample_from_model_given_age(age, self.dataset, self.model)
            ages = (ages - self.model["min_age"]) / (self.model["max_age"] - self.model["min_age"])
            # ages = (ages - self.mu_std_df["mu"][self.vars[0]]) / self.mu_std_df["mu"][self.vars[0]]
            if self.log_volumes:
                if self.model["log_shift"] is not None:
                    volumes = (torch.log(volumes + self.model["log_shift"]) - self.mu_std_df["mu"][self.vols[0] + "_shift"]) / self.mu_std_df["std"][self.vols[0] + "_shift"]
                else:
                    volumes = (torch.log(volumes) - self.mu_std_df["mu"][self.vols[0]]) / self.mu_std_df["std"][self.vols[0]]
            else:
                volumes = (volumes - self.mu_std_df["mu"][self.vols[0]]) / self.mu_std_df["std"][self.vols[0]]
            return torch.cat([ages, volumes], dim=1)
        else:
            return sample_from_model_given_age(age, self.dataset, self.model)
    
# UKB settings:
def get_ukb_model():
    df = get_data()
    return estimate_mle(
        df,
        age_estimator="beta",
        log_volumes=True,
        volumes_as_gmm=True,
        n_components=5,
    )

def get_data(dataset: str, vars: list, path: str = None, image_fnames: list = None):
    if path.split("/")[-2] != "trainset":
        raise ValueError("path must be the trainset folder")
    label_file = os.path.join(path, "dataset.json")
    with open(label_file, "rb") as f:
        labels = json.load(f)["labels"]
    labels = dict(labels)
    labels = [labels[fname.replace("\\", "/")] for fname in image_fnames] ## a dict

    new_labels = np.zeros(shape=(len(labels), len(vars)), dtype=np.float32) ## [, 2]
    if dataset == "adni":
        for num, l in enumerate(labels):
            i = list(l[vars[0]].items())[0][0]
            temp = [l[var][str(i)] for var in vars]
            temp[1] = 1 if temp[1] == "M" else 0
            ## cdr 
            temp[2] = 1 if temp[2] >= 1.0 else temp[2]
            new_labels[num, :] = temp
    else:
        for num, l in enumerate(labels):
            i = list(l[vars[0]].items())[0][0]
            temp = [l[var][str(i)] for var in vars]
            new_labels[num, :] = temp
    new_labels = pd.DataFrame(new_labels, columns=vars)
    new_labels = new_labels[vars].dropna(axis=0)
    return new_labels


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


def estimate_mle(
    df,
    dataset: str,       ## the name of the dataset
    vars,               ## labels: all variable, containing age, sex, volumes...
    vols,               ## labels: volumes from MRI images
    volumes_as_gmm=True,
    n_components=3,     ## for ukb = 5, oasis = 2
    gmm_random_state=0,
    age_estimator="normal",
    log_volumes=False,  ## default: False, if data == 'ukb' then true.
    log_shift=1e2,
):

    ## Age
    model = dict(
        age_model=age_estimator,
        vol_model="lognormal" if log_volumes else "normal",
    )
    labels_vol = vols ## labels: volumes from MRI images
    if df[vols].values.min() < 0 and log_volumes: ## need to separate positive and negative values
        log_neg = True
        vols = np.log(df[vols].values + log_shift)
    else:
        log_neg = False
        vols = np.log(df[vols].values) if log_volumes else df[vols].values

    ## Gaussian dist
    age = df[vars[0]].values
    if age_estimator == "normal":
        mu_age = age.mean()
        sigma_age = age.std(ddof=0)
        model["mu_age"] = mu_age
        model["sigma_age"] = sigma_age
    elif age_estimator == "beta":
        alpha_age, beta_age = estimate_beta1d(
            age, method="moments", eps=1e-4, lo=age.min(), hi=age.max()
        )
        model.update(
            alpha_age=alpha_age,
            beta_age=beta_age,
            min_age=age.min(),
            max_age=age.max(),
            age_model="beta",
        )
    elif age_estimator == "gmm":
        gm = GaussianMixture(n_components=n_components)
        gm.fit(age.reshape(-1, 1))
        model.update(
            gm_age = gm,
            min_age=age.min(),
            max_age=age.max(),
            age_model="gmm",
        )
    ## cdr only in Oasis and ADNI
    if dataset == "oasis3" or dataset == "adni":
        if dataset == "oasis3":
            cdr_weights, cdr_bias = softmax_regression(
                y=df["cdr"].values, x=df[["Age", "M/F"]]
            )
        else:
            ## assume CDR >= 1.0 is AD (1.0, 2.0, 3.0) == 1.0 
            cdr_weights, cdr_bias = softmax_regression(
            y=df["CDGLOBAL"].values, x=df[["Age", "Sex"]]
            )
        model.update(
            cdr_weights=torch.tensor(cdr_weights, dtype=torch.float),
            cdr_bias=torch.tensor(cdr_bias, dtype=torch.float),
            vol_means=vols.mean().values,
            vol_std=vols.std().values,
            cdr_encoder=OneHotEncoder().fit(df["cdr"].values.reshape(-1, 1)) if dataset == "oasis3" 
                else OneHotEncoder().fit(df["CDGLOBAL"].values.reshape(-1, 1)),
            cdr_classes=torch.tensor([0, 0.5, 1, 2]) if dataset == "oasis3"
                else torch.tensor([0, 0.5, 1]),
        )

    elif dataset == "ukb":
        model.update(
            vol_means=vols.mean() if log_volumes else df[labels_vol].mean().values,
            vol_std=vols.std() if log_volumes else df[labels_vol].std().values,
        )
    elif dataset == "retinal":
        if log_neg: ## need to separate positive and negative values
            model.update(
                c_means=vols.mean(),
                c_std=vols.std(),
                log_shift=log_shift,
            )
        else:
            model.update(
                c_means=vols.mean() if log_volumes else df[labels_vol].mean().values,
                c_std=vols.std() if log_volumes else df[labels_vol].std().values,
            )
    
    if volumes_as_gmm: ## gussian mixture model
        if dataset == "ukb" or dataset == "retinal":
            vol_gmm, vol_indices = gmm_regression(
                    y=vols.reshape(-1, 1),
                    x=df[vars[0]].values.reshape(-1, 1),
                    n_components=n_components,
                    random_state=gmm_random_state,
                )
        elif dataset == "adni":
            vol_gmm, vol_indices = gmm_regression(
                y=vols,
                x=df[["Age", "Sex", "CDGLOBAL"]],
                n_components=n_components,
                random_state=gmm_random_state,
            )
        model["vol_gmm"] = vol_gmm
        model["vol_indices"] = vol_indices
        model["log_shift"] = log_shift if log_neg else None

    else: ## multivariate linear regression
        if dataset == "ukb":
            vol_weights, vol_bias, vol_sigma = mv_linear_regression(
                y=vols, x=df[["age", "sex"]]
            )
        model["vol_weights"] = torch.tensor(vol_weights, dtype=torch.float)
        model["vol_bias"] = torch.tensor(vol_bias, dtype=torch.float)
        model["vol_sigma"] = torch.tensor(vol_sigma, dtype=torch.float)

    return model

def sample_from_model(dataset: str, model, num_samples=100):
    if model["age_model"] == "normal":
        A = torch.normal(model["mu_age"], model["sigma_age"], size=(num_samples, 1))
    elif model["age_model"] == "beta":
        A = torch.tensor(
            model["min_age"]
            + (model["max_age"] - model["min_age"])
            * stats.beta.rvs(
                model["alpha_age"], model["beta_age"], size=(num_samples, 1)
            ),
            dtype=torch.float32,
        )
    elif model["age_model"] == "gmm":
        A = torch.zeros(size=(num_samples, 1), dtype=torch.float32)
        for i in range(num_samples):
            a = -1
            while a < model["min_age"] or a > model["max_age"]:
                a = torch.tensor(model["gm_age"].sample(1)[0])
            A[i, :] = a
                
    else:
        raise ValueError(model["age_model"])
    X = A
    if dataset == "oasis3" or dataset == "adni":
        cdr_prob = torch.softmax(X @ model["cdr_weights"].T + model["cdr_bias"], dim=1)
        C = model["cdr_classes"][torch.multinomial(cdr_prob, num_samples=1)]

        X = torch.cat([A, C], dim=1)

    if "vol_gmm" in model:
        V = sample_from_gmm_custom(
            x=X, indices=model["vol_indices"], gmm=model["vol_gmm"]
        )
    else:
        mvn = MultivariateNormal(
            loc=torch.zeros_like(model["vol_bias"]),
            covariance_matrix=model["vol_sigma"],
        )
        V = X @ model["vol_weights"].T + model["vol_bias"] + mvn.sample((num_samples,))
    if model["vol_model"] == "lognormal":
        V.exp_()
        if model["log_shift"] is not None:
            V = V - model["log_shift"]
    if dataset == "ukb" or dataset == "retinal":
        return A, V
    elif dataset == "adni":
        return A, C, None, V

def sample_from_model_given_age(age, dataset: str, model):
    """
    age: (torch.tensor) age in years (float)
    model: causal model
    """
    num_samples = len(age)
    age = age.reshape(-1, 1)
    if dataset == "oasis3" or dataset == "adni":
        cdr_prob = torch.softmax(X @ model["cdr_weights"].T + model["cdr_bias"], dim=1)
        C = model["cdr_classes"][torch.multinomial(cdr_prob, num_samples=1)]

        X = torch.cat([age, C], dim=1)

    if "vol_gmm" in model:
        if type(age) != torch.Tensor:
            age = torch.tensor(age, dtype=torch.float32)
        V = sample_from_gmm_custom(
            x=age, indices=model["vol_indices"], gmm=model["vol_gmm"]
        )
    else:
        mvn = MultivariateNormal(
            loc=torch.zeros_like(model["vol_bias"]),
            covariance_matrix=model["vol_sigma"],
        )
        V = X @ model["vol_weights"].T + model["vol_bias"] + mvn.sample((num_samples,))
    if model["vol_model"] == "lognormal":
        V.exp_()
        if model["log_shift"] is not None:
            V = V - model["log_shift"]
    
    if dataset == "ukb" or dataset == "retinal":
        return age, V.reshape(-1, 1)
    elif dataset == "adni":
        return age, C, V
    
def sample_from_model_given_cdr(cdr, age, sex, dataset: str, model):
    """
    Intervene on CDR (as a one-hot vector)
    age and sex are parents of CDR (i.e. they are not intervened on)
    age: (torch.tensor) age in years (float)
    sex: (torch.tensor) sex in (0, 1)
    model: causal model
    """
    assert dataset == "oasis3" or dataset == "adni"
    assert cdr.shape[0] == age.shape[0]
    assert age.shape[0] == sex.shape[0]
    num_samples = cdr.shape[0]

    X = torch.cat([age, sex, cdr], dim=1)

    if "vol_gmm" in model:
        V = sample_from_gmm_custom(
            x=X, indices=model["vol_indices"], gmm=model["vol_gmm"]
        )
    else:
        # VenVol = torch.tensor(model["vent_vol_regr"].predict(X)).reshape(-1, 1)
        mvn = MultivariateNormal(
            loc=torch.zeros_like(model["vol_bias"]),
            covariance_matrix=model["vol_sigma"],
        )
        V = X @ model["vol_weights"].T + model["vol_bias"] + mvn.sample((num_samples,))
    if model["vol_model"] == "lognormal":
        V.exp_()
    if dataset == "oasis3":
        return sex, age, cdr, None, V
    elif dataset == "adni":
        return sex, age, cdr, None, V


def preprocess_samples(A, V=None, model=None, dataset:str = None):
    if dataset == "ukb":
        if model["vol_model"] == "lognormal":
            V = (torch.log(V) - model["vol_means"]) / model["vol_std"]
        else:
            V = (V - model["vol_means"]) / model["vol_std"]
    elif dataset == "retinal":
        if model["vol_model"] == "lognormal":
            if model["log_shift"] is not None:
                V = (torch.log(V + model["log_shift"]) - model["c_means"]) / model["c_std"]
            else:
                V = (torch.log(V) - model["c_means"]) / model["c_std"]
        else:
            V = (V - model["c_means"]) / model["c_std"]
    if dataset == "oasis3" or dataset == "adni":
        if dataset == "oasis3":
            A = (A - model["mu_age"]) / model["sigma_age"]
        else:
            A = (A - model["min_age"]) / (model["max_age"] - model["min_age"])
        ## one hot encode
        C = torch.tensor(model["cdr_encoder"].transform(C).toarray())
        
        return torch.cat([A, C, V], 1)
    elif dataset == "ukb" or dataset == "retinal":
        A = (A - model["min_age"]) / (model["max_age"] - model["min_age"])
        return torch.cat([A, V], 1)
    else:
        raise ValueError(f'{dataset} is wrong')

def unnormalize_samples(S, A, C=None, V=None, model=None, dataset: str = None):
    ### reverse function of preprocess_samples
    V = V * model["vol_std"] + model["vol_means"]
    if dataset == "oasis3" or dataset == "adni":
        if dataset == "oasis3":
            A = model["mu_age"] + A * model["sigma_age"]
            C = torch.tensor(model["cdr_encoder"].inverse_transform(C.cpu().detach().numpy()).reshape(-1, 1))
            return torch.cat([S.reshape(-1, 1), A.reshape(-1, 1), C, V], dim=1)
        else:
            A = model["min_age"] + (model["max_age"] - model["min_age"]) * A
            C = torch.tensor(model["cdr_encoder"].inverse_transform(C.cpu().detach().numpy()).reshape(-1, 1))
            return torch.cat([S.reshape(-1, 1), A.reshape(-1, 1), C, V], dim=1)
    elif dataset == "ukb":
        A = model["min_age"] + (model["max_age"] - model["min_age"]) * A
        return torch.cat([S.reshape(-1, 1), A.reshape(-1, 1), V], dim=-1)
    else:
        raise ValueError(f'{dataset} is wrong')

def softmax_regression(y, x):
    lr = LogisticRegression(multi_class="multinomial", penalty="none", max_iter=10000)
    lr.fit(x, y.astype("str"))
    return lr.coef_, lr.intercept_


def gmm_regression(y, x, n_components=3, random_state=0):
    gmm = gmr.GMM(n_components=n_components, random_state=random_state)
    xy = np.concatenate([x, y], axis=1)
    gmm.from_samples(xy)
    indices = np.arange(x.shape[1])
    return gmm, indices


def mv_linear_regression(y, x):
    Sigma = np.cov(y.T, ddof=1)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)

    W = np.linalg.solve(x.T @ x, x.T @ y)
    b = W[-1, :]
    W = W[:-1, :].T
    return W, b, Sigma


def random_forest_regression(y, x):
    regr = RandomForestRegressor(max_depth=5, random_state=0)
    regr.fit(x, y)
    return regr


def sample_from_gmm_custom(x, indices, gmm):
    """mostly copy-pasted from
    https://github.com/AlexanderFabisch/gmr/blob/master/gmr/gmm.py
    except for the last few lines
    """

    X = x.numpy()

    indices = np.asarray(indices, dtype=int)
    X = np.asarray(X)

    n_samples = len(X)
    output_indices = invert_indices(gmm.means.shape[1], indices)
    regression_coeffs = np.empty((gmm.n_components, len(output_indices), len(indices)))

    marginal_norm_factors = np.empty(gmm.n_components)
    marginal_exponents = np.empty((n_samples, gmm.n_components))

    for k in range(gmm.n_components):
        regression_coeffs[k] = regression_coefficients(
            gmm.covariances[k], output_indices, indices
        )
        mvn = MVN(
            mean=gmm.means[k],
            covariance=gmm.covariances[k],
            random_state=gmm.random_state,
        )
        marginal_norm_factors[k], marginal_exponents[:, k] = mvn.marginalize(
            indices
        ).to_norm_factor_and_exponents(X)

    # posterior_means = mean_y + cov_xx^-1 * cov_xy * (x - mean_x)
    posterior_means = gmm.means[:, output_indices][:, :, np.newaxis].T + np.einsum(
        "ijk,lik->lji", regression_coeffs, X[:, np.newaxis] - gmm.means[:, indices]
    )

    priors = _safe_probability_density(
        gmm.priors * marginal_norm_factors, marginal_exponents
    )

    priors, posterior_means
    covs = gmm.covariances

    samples = []
    for i in range(len(X)):
        component = np.random.choice(np.arange(priors.shape[-1]), size=1, p=priors[i])[
            0
        ]
        mean = posterior_means[i, :, component]
        cov = covs[component][output_indices, :][:, output_indices]
        sample = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
        samples.append(sample)

    return torch.tensor(np.array(samples), dtype=torch.float)



### --- for debugging ---
### using a gaussian distribution to represent a variable
### assume cdr without one hot vector
### labels are with shape = (num_samples, 6)
def gaussian_random_sample(mu=0, cov=1, num_samples=100):
    labels = torch.zeros(size=(num_samples, 6), dtype=torch.float32)
    for i in range(6):
        labels[:, i] = torch.normal(
            mean=mu, std=cov, generator=torch.Generator(), size=(num_samples, 1)
        ).reshape(-1)

    return labels
