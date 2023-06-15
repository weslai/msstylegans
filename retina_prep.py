from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_PTH = "/dhc/projects/ukbiobank/derived/imaging/retinal_fundus/{side}/512_{side}/processed/"
UKB_FIELD = lambda side: {"left": "21015", "right": "21016"}[side]
PHENO_PTH = "/dhc/projects/ukbiobank/original/phenotypes/ukb49727.csv"

QCOVARIATES = {
    "age": 21022,
    "systolic_bp": 4080,
    "diastolic_bp": 4079,
    "cylindrical_power_left": 5086,
    "cylindrical_power_right": 5087,
    "spherical_power_left": 5085,
    "spherical_power_right": 5084,
}
CCOVARIATES = {
    "female": 31,
    "myopia": 6147,
    "hyperopia": 6147,
    "presbyopia": 6147,
    "astigmatism": 6147,
}
CCOVARIATE_CODING = {
    "myopia": 1,
    "hyperopia": 2,
    "astigmatism": 4,
    "presbyopia": 3,
    "female": 0,
}
NROWS = None


def load_data(seed=123):
    # insert real path
    df = pd.read_csv("tmp.csv", index_col=0)

    perm = np.random.RandomState(seed).permutation(len(df))
    df = df.iloc[perm]
    df_train, df_test = df.iloc[:-10000], df.iloc[-10000:]
    D_train = UKBRetina(df=df_train, covariates=["myopia", "female", "astigmatism"])
    D_test = UKBRetina(df=df_test, covariates=["myopia", "female", "astigmatism"])

    return D_train, D_test


def get_img_indivs(out_file="tmp.csv", side="left"):
    field = UKB_FIELD(side)
    pth = Path(IMG_PTH.format(side=side))
    pths = list(pth.glob(f"*_{field}_0_*.jpg"))
    iids = [int(pth.name.split("_")[0]) for pth in pths]
    df = pd.DataFrame({"iid": iids, "path": pths}).sort_values(by="path", axis=0)
    df.drop_duplicates(subset="iid", inplace=True, keep="last")
    df.index = df.iid
    df.drop(columns=["iid"], inplace=True)

    qdf, cdf = get_pheno(df.index.values)
    df = df.join(qdf).join(cdf)
    df.to_csv(out_file)

    return df


def get_pheno(iids):
    # quantitative:
    sniff = pd.read_csv(PHENO_PTH, nrows=2)

    qcols = ["eid"] + [
        col for col in sniff.columns if is_in_set(col, QCOVARIATES.values())
    ]
    qdf = pd.read_csv(PHENO_PTH, usecols=qcols, index_col=0, nrows=NROWS)
    I = sorted(np.intersect1d(qdf.index, iids))
    qdf = qdf.loc[I]
    for pheno in QCOVARIATES:
        cols = [col for col in qdf.columns if is_in_set(col, [QCOVARIATES[pheno]])]
        qdf[pheno] = qdf[cols].mean(1)
    qdf = qdf[QCOVARIATES.keys()]

    # categorical:
    ccols = ["eid"] + [
        col for col in sniff.columns if is_in_set(col, CCOVARIATES.values())
    ]
    cdf = pd.read_csv(PHENO_PTH, usecols=ccols, index_col=0, nrows=NROWS)
    I = sorted(np.intersect1d(cdf.index, iids))
    cdf = cdf.loc[I]
    for pheno in CCOVARIATES:
        cols = [col for col in cdf.columns if is_in_set(col, [CCOVARIATES[pheno]])]
        cdf[pheno] = (cdf[cols] == CCOVARIATE_CODING[pheno]).any(axis=1) * 1.0
    cdf = cdf[CCOVARIATES.keys()]

    return qdf, cdf


def is_in_set(col, cols):
    return any([col.startswith(f"{c}-0.") for c in cols])


class UKBRetina(Dataset):
    def __init__(self, df, covariates=[], tfms=None):
        available_covs = set(QCOVARIATES.keys()) | set(CCOVARIATES.keys())
        assert all(
            [cov in available_covs for cov in covariates]
        ), f"invalid covariate, must be in {available_covs}"
        self.covariates = covariates
        df = df.dropna(subset=covariates)
        self.df = df
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pth = self.df.iloc[idx].path
        img = Image.open(pth)
        covs = torch.from_numpy(
            self.df.iloc[idx][self.covariates].values.astype(np.float32)
        )
        if self.tfms is not None:
            img = self.tfms(img)
        return img, covs

if __name__ == "__main__":
    save_path = "/dhc/groups/fglippert/Ukbiobank/imaging/retinal_fundus/phenotype.csv"
    df = get_img_indivs(out_file=save_path, side="left")
