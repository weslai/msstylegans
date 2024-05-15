import os, glob
import pandas as pd

import matplotlib.pyplot as plt

DATAMODALITY = "mri"
FOLDER = "/dhc/home/wei-cheng.lai/experiments/multisources/real_ms/mri_imgs/mse_re128/disc_loss_lambda/00000-stylegan3-t-condTrue-multisource-ukb-augada-gpus2-batch256-gamma0.5-kimg40000"

if DATAMODALITY == "mri":
    DATASETS = ["ukb", "adni"]
    COVS = ["age", "grey_matter", "ventricle", "left_hippocampus", "right_hippocampus"]
elif DATAMODALITY == "retinal":
    DATASETS = ["ukb", "rfmid"]
    COVS = ["age", "cataract", "spherical_power", "disease_risk", "MH", "TSLN"]

file_combinations = os.path.join(FOLDER, "general_test_loss_")
file_combinations = sorted(glob.glob(file_combinations + "*"))
corr_scores = {}
for file in file_combinations:
    cov_name = file.split("/")[-1].split("_")[3:5]
    cov_name = "_".join(cov_name)
    df = pd.read_csv(file)
    if cov_name in corr_scores.keys():
        corr_scores[cov_name].append(df["corr"].values[0])
    else:
        corr_scores[cov_name] = [df["corr"].values[0]]

df_corr = pd.DataFrame.from_dict(corr_scores)

fig = plt.figure(figsize=(10, 8))
plt.plot(df_corr)
plt.legend(df_corr.columns)
plt.tight_layout()
plt.savefig("general_test_loss_corr.png")
plt.close()