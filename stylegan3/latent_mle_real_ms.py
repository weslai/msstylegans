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
import torch.nn.functional as F
import torch
import gmr
from gmr.mvn import invert_indices, regression_coefficients, MVN
from gmr.gmm import _safe_probability_density
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

## get settings
def get_settings(dataset: str):
    if dataset == "ukb":
        n_components = 10
        age_estimator = "beta"
        log_volumes = True
    elif dataset == "adni":
        n_components = 8
        age_estimator = "beta"
        log_volumes = False
    elif dataset == "nacc":
        n_components = None
        age_estimator = "beta"
        log_volumes = False
    elif dataset == "retinal":
        n_components = 13
        age_estimator = "gmm"
        log_volumes = True
    elif dataset in ["eyepacs", "rfmid"]:
        n_components = None
        age_estimator = None
        log_volumes = False
    return n_components, age_estimator, log_volumes
## select a dataset
def set_dataset(name: str):
    ## The Order of the vars should be (age, [cdr], VOLS)
    if name == "ukb":
        VOLS = ["ventricle", "grey_matter"]
        VARS = ["age"] + VOLS
        ## ------------------------------------------------------
    elif name == "adni":
        VOLS = ["left_hippocampus", "right_hippocampus"]
        # VARS = ["Age", "CDGLOBAL"]
        VARS = ["Age"] + VOLS
        ## ------------------------------------------------------
    elif name == "nacc":
        VOLS = None
        VARS = ["Age", "Apoe4"]
        # VARS = ["Age", "Apoe4", "CDGLOBAL"]
        ## ------------------------------------------------------
    elif name == "retinal":
        VOLS = ["spherical_power_left"]
        # VOLS = ["diastolic_bp", "spherical_power_left"]
        VARS = ["age", "cataract"] + VOLS
        ## ------------------------------------------------------
    elif name == "eyepacs":
        VOLS = None
        VARS = ["level"]
        ## ------------------------------------------------------
    elif name == "rfmid":
        VOLS = None
        VARS = ["Disease_Risk", "MH","TSLN"]
    return VARS, VOLS

class SourceSampling:
    def __init__(
        self, 
        dataset, ## ukb, adni, retinal, eyepacs
        label_path: str = None
    ):
        self.dataset = dataset
        assert self.dataset in ["ukb", "adni", "nacc", "retinal", "eyepacs", "rfmid"]
        if label_path is None:
            raise ValueError(f"{self.dataset} must have a label_path")
        else:
            self.label_path = label_path
            if not self.label_path.endswith("/"):
                self.label_path += "/"
            if self.label_path.split("/")[-2] != "trainset":
                assert self.label_path.split("/")[-2] == "valset" or self.label_path.split("/")[-2] == "testset"
                temp_path = self.label_path.split("/")[:-2]
                self.label_path = "/" + os.path.join(*temp_path) + "/trainset/"
        
        self.vars, self.vols = set_dataset(self.dataset)
        self.n_components, self.age_estimator, self.log_volumes = get_settings(self.dataset)
        ## we only know labels from the training set
        self._get_all_fnames()
        if dataset in ["ukb", "adni", "nacc"]:
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
            if self.dataset == "retinal": ## ukb
                # df[self.vols[0]] = np.log(df[self.vols[0]])
                df[self.vols[0] + "_shift"] = np.log(df[self.vols[0]] + 1e2)
                df["shift_scale"] = 1e2
            else:
                df[self.vols] = np.log(df[self.vols])
        return {"mu": df.mean(), "std": df.std()}

    def get_graph(self):
        self.df = get_data(self.dataset, self.vars, self.label_path, self._image_fnames)
        self.mu_std_df = self._mu_std_df()
        return self.df
        
    def get_causal_model(self):
        if self.dataset not in ["eyepacs", "rfmid"]:
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
        else:
            self.model = estimate_model(
                df= self.get_graph(),
                dataset=self.dataset,
                vars=self.vars,
                vols=self.vols
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
            if self.dataset == "nacc":
                return preprocess_samples(ages, volumes, model=self.model, dataset=self.dataset)
            ages = (ages - self.model["min_age"]) / (self.model["max_age"] - self.model["min_age"])
            # ages = (ages - self.mu_std_df["mu"][self.vars[0]]) / self.mu_std_df["mu"][self.vars[0]]
            if self.log_volumes:
                if self.model["log_shift"] is not None:
                    if self.dataset == "retinal":
                        volumes[:, 1] = (torch.log(volumes[:, 1] + self.model["log_shift"]) - self.mu_std_df["mu"][self.vols[0] + "_shift"]) / self.mu_std_df["std"][self.vols[0] + "_shift"]
                        # volumes[:, 0] = (torch.log(volumes[:, 0]) - self.mu_std_df["mu"][self.vols[0]]) / self.mu_std_df["std"][self.vols[0]]
                    else:
                        volumes = (torch.log(volumes + self.model["log_shift"]) - self.mu_std_df["mu"][self.vols[0] + "_shift"]) / self.mu_std_df["std"][self.vols[0] + "_shift"]
                else:
                    volumes = (torch.log(volumes) - self.mu_std_df["mu"][self.vols].values.reshape(1, -1)) / self.mu_std_df["std"][self.vols].values.reshape(1, -1)
            else:
                volumes = (volumes - self.mu_std_df["mu"][self.vols].values.reshape(1, -1)) / self.mu_std_df["std"][self.vols].values.reshape(1, -1)
            return torch.cat([ages, volumes], dim=1)
        else:
            return sample_from_model_given_age(age, self.dataset, self.model)

def get_data(dataset: str, vars: list, path: str = None, image_fnames: list = None):
    if path.split("/")[-2] != "trainset":
        raise ValueError("path must be the trainset folder")
    label_file = os.path.join(path, "dataset.json")
    with open(label_file, "rb") as f:
        labels = json.load(f)["labels"]
    labels = dict(labels)
    labels = [labels[fname.replace("\\", "/")] for fname in image_fnames] ## a dict

    new_labels = np.zeros(shape=(len(labels), len(vars)), dtype=np.float32) ## [, 2]
    for num, l in enumerate(labels):
        i = list(l[vars[0]].items())[0][0]
        temp = [l[var][str(i)] for var in vars]
        new_labels[num, :] = temp
    if dataset == "eyepacs": ## binary
        new_labels[new_labels > 0] = 1
    new_labels = pd.DataFrame(new_labels, columns=vars)
    if dataset == "nacc":
        new_labels = new_labels[new_labels["Apoe4"] != 9.0]
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
    vars,               ## labels: all variable, containing age, volumes...
    vols,               ## labels: volumes from MRI images
    volumes_as_gmm=True,
    n_components=3,
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
    labels_vol = vols ## labels
    if dataset != "retinal":
        if labels_vol is not None:
            if df[vols].values.min() < 0 and log_volumes: ## need to separate positive and negative values
                log_neg = True
                vols = np.log(df[vols].values + log_shift)
            else:
                log_neg = False
                vols = np.log(df[vols].values) if log_volumes else df[vols].values
    else:
        log_neg = []
        volumes = []
        for vol in vols:
            if df[vol].values.min() < 0 and log_volumes:
                log_n = True
                vol = np.log(df[vol].values + log_shift)
            else:
                log_n = False
                vol = np.log(df[vol].values) if log_volumes else df[vol].values
            log_neg.append(log_n)
            volumes.append(vol.reshape(-1, 1))
        if log_neg.count(True) > 0:
            log_neg = True
        vols = np.concatenate(volumes, axis=1)

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
    ## beside age
    ## cdr only in ADNI
    if dataset == "adni":
        ### left and right hippocampus
        model.update(
            vol_means=vols.mean(axis=0) if log_volumes else df[labels_vol].mean().values,
            vol_std=vols.std(axis=0) if log_volumes else df[labels_vol].std().values,
        )
    elif dataset == "nacc":
        ## apoe = 0, 1, 2
        apoe_weights, apoe_bias = softmax_regression(
        y=df["Apoe4"].values, x=df[["Age"]]
        )

        model.update(
            apoe_weights=torch.tensor(apoe_weights, dtype=torch.float),
            apoe_bias=torch.tensor(apoe_bias, dtype=torch.float),
            apoe_encoder=OneHotEncoder().fit(df["Apoe4"].values.reshape(-1, 1)),
            apoe_classes=torch.tensor([0, 1, 2]),
        )
    elif dataset == "ukb":
        model.update(
            vol_means=vols.mean(axis=0) if log_volumes else df[labels_vol].mean().values,
            vol_std=vols.std(axis=0) if log_volumes else df[labels_vol].std().values,
        )
    elif dataset == "retinal":
        if log_neg: ## need to separate positive and negative values
            model.update(
                c_means=vols.mean(axis=0),
                c_std=vols.std(axis=0),
                log_shift=log_shift,
            )
        else:
            model.update(
                c_means=vols.mean(axis=0) if log_volumes else df[labels_vol].mean().values,
                c_std=vols.std(axis=0) if log_volumes else df[labels_vol].std().values,
            )
        if vars[1] == "cataract":
            ## logistic regression
            cataract_weights, cataract_bias = softmax_regression(
                y=df[vars[1]].values, x=df[vars[0]].values.reshape(-1, 1)
            )
            model.update(
                cataract_weights=torch.tensor(cataract_weights, dtype=torch.float),
                cataract_bias=torch.tensor(cataract_bias, dtype=torch.float),
            )
    if labels_vol is not None:
        if volumes_as_gmm: ## gussian mixture model
            if dataset in ["ukb", "retinal", "adni"]:
                vol_gmm, vol_indices = gmm_regression(
                        y=vols,
                        x=df[vars[0]].values.reshape(-1, 1),
                        n_components=n_components,
                        random_state=gmm_random_state,
                    )
                model["vol_gmm"] = vol_gmm
                model["vol_indices"] = vol_indices
                model["log_shift"] = log_shift if log_neg else None

        else: ## multivariate linear regression
            if dataset == "ukb":
                vol_weights, vol_bias, vol_sigma = mv_linear_regression(
                    y=vols, x=df[["age"]]
                )
                model["vol_weights"] = torch.tensor(vol_weights, dtype=torch.float)
                model["vol_bias"] = torch.tensor(vol_bias, dtype=torch.float)
                model["vol_sigma"] = torch.tensor(vol_sigma, dtype=torch.float)

    return model

def estimate_model(
    df,
    dataset: str,       ## the name of the dataset
    vars,               ## labels: all variables
    vols: list = None   ## labels: volumes
):
    if dataset == "eyepacs":
        labels = df[vars].values
        labels_mu = np.mean(labels, axis=0)
        return labels_mu
    elif dataset == "rfmid":
        cls_prob = {}
        for var in vars[1:]:
            labels = df[var].values
            cls_prob[var] = np.sum(labels) / len(df)
        return cls_prob

def sample_from_model(dataset: str, model, num_samples=100):
    if dataset not in ["eyepacs", "rfmid"]:
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
        if dataset == "nacc":
            apoe_prob = torch.softmax(X @ model["apoe_weights"].T + model["apoe_bias"], dim=1)
            apoe = model["apoe_classes"][torch.multinomial(apoe_prob, num_samples=1)]
            # V = torch.cat((apoe, V), dim=1)
            V = apoe
        elif dataset == "retinal" and "cataract_weights" in model: ## for cataract
            cataract_prob = torch.sigmoid(X @ model["cataract_weights"].T + model["cataract_bias"])
            cataract_cls = torch.bernoulli(cataract_prob)
            if "vol_gmm" in model: ## spherical_power_left
                V = sample_from_gmm_custom(
                    x=X, indices=model["vol_indices"], gmm=model["vol_gmm"]
                )
        else:
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
                if dataset == "retinal":
                    temp_v = V[:, 0] - model["log_shift"]
                    V[:, 0] = temp_v
                else:
                    V = V - model["log_shift"]
        if dataset == "retinal":
            V = torch.cat([cataract_cls, V], dim=1)
    elif dataset == "eyepacs":
        A = None
        V = (torch.rand((num_samples, 1)) < model[0]) * 1.0
    elif dataset == "rfmid":
        A = None
        V = torch.zeros((num_samples, len(model)+1))
        V[:, 1:] = (torch.rand((num_samples, 2)) < torch.tensor([model["MH"], model["TSLN"]])) * 1.0
        disease_sum = torch.sum(V[:, 1:], dim=1)
        disease_idxs = torch.where(disease_sum > 0)[0]
        V[disease_idxs, 0] = 1.0
    return A, V

def sample_from_model_given_age(age, dataset: str, model):
    """
    age: (torch.tensor) age in years (float)
    model: causal model
    """
    num_samples = len(age)
    age = age.reshape(-1, 1)
    if type(age) != torch.Tensor:
            age = torch.tensor(age, dtype=torch.float32)
    
    if dataset == "nacc":
        apoe_prob = torch.softmax(age @ model["apoe_weights"].T + model["apoe_bias"], dim=1)
        apoe = model["apoe_classes"][torch.multinomial(apoe_prob, num_samples=1)]
        V = apoe
        # V = torch.cat((apoe, V), dim=1)
    elif dataset == "retinal" and "cataract_weights" in model: ## for cataract
        cataract_prob = torch.sigmoid(age @ model["cataract_weights"].T + model["cataract_bias"])
        cataract_cls = torch.bernoulli(cataract_prob)
        if "vol_gmm" in model:
            V = sample_from_gmm_custom(
                x=age, indices=model["vol_indices"], gmm=model["vol_gmm"]
            )
    else:
        if "vol_gmm" in model:
            V = sample_from_gmm_custom(
                x=age, indices=model["vol_indices"], gmm=model["vol_gmm"]
            )
        else:
            mvn = MultivariateNormal(
                loc=torch.zeros_like(model["vol_bias"]),
                covariance_matrix=model["vol_sigma"],
            )
            V = age @ model["vol_weights"].T + model["vol_bias"] + mvn.sample((num_samples,))
    if model["vol_model"] == "lognormal":
        V.exp_()
        if model["log_shift"] is not None:
            if dataset == "retinal":
                temp_v = V[:, 0] - model["log_shift"]
                V[:, 0] = temp_v
            else:
                V = V - model["log_shift"]
    if dataset == "retinal":
        V = torch.cat([cataract_cls, V], dim=1)
    return age, V

def preprocess_samples(A=None, V=None, model=None, dataset:str = None):
    if dataset in ["ukb", "adni"]:
        if model["vol_model"] == "lognormal":
            V = (torch.log(V) - model["vol_means"]) / model["vol_std"]
        else:
            V = (V - model["vol_means"]) / model["vol_std"]
    elif dataset == "retinal":
        if model["vol_model"] == "lognormal":
            if model["log_shift"] is not None:
                temp_V = (torch.log(V[:, 1] + model["log_shift"]) - model["c_means"][0]) / model["c_std"][0]
                V[:, 1] = temp_V
                # V[:, 0] = (torch.log(V[:, 0]) - model["c_means"][0]) / model["c_std"][0]
            else:
                V = (torch.log(V) - model["c_means"]) / model["c_std"]
        else:
            V = (V - model["c_means"]) / model["c_std"]
    
    elif dataset == "eyepacs":
        return V
    elif dataset == "nacc":
        apoe = torch.tensor(model["apoe_encoder"].transform(V).toarray())
        V = apoe
    else:
        raise ValueError(f'{dataset} is wrong')
    A = (A - model["min_age"]) / (model["max_age"] - model["min_age"])
    return torch.cat([A, V], 1)

# def unnormalize_samples(S, A, C=None, V=None, model=None, dataset: str = None):
def unnormalize_samples(S, A, V=None, model=None, dataset: str = None):
    ### reverse function of preprocess_samples
    V = V * model["vol_std"] + model["vol_means"]
    A = model["min_age"] + (model["max_age"] - model["min_age"]) * A
    return torch.cat([S.reshape(-1, 1), A.reshape(-1, 1), V], dim=1)

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