import pandas as pd
import numpy as np
import os, sys
import glob
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans")

FILE = "/dhc/home/wei-cheng.lai/experiments/multisources/morpho/regression/estimated_norm_mse_lambda/new_opt_disc_shortcut/stratified_predictions_thickness_stra0_shortcuts.csv"
SAVEFILE = "/dhc/home/wei-cheng.lai/experiments/multisources/morpho/regression/estimated_norm_mse_lambda/new_opt_disc_shortcut/stratified_loss_thickness_shortcuts.csv"

dataset = "mnist-thickness-intensity-slant"
### cov
if dataset == "mnist-thickness-intensity-slant":
    covs = ["thickness", "intensity", "slant"]
elif dataset == "ukb":
    covs = ["age", "ventricle", "grey_matter"]
elif dataset == "retinal":
    covs = ["age", "diastolic", "spherical"]

for cov in covs:
    pattern = FILE.split("/")[-1].split("_")[:-3] + [cov]
    files_same_pattern = glob.glob(os.path.join("/".join(FILE.split("/")[:-1]), "_".join(pattern) + "_*.csv"))
    savefile_name = SAVEFILE.split("/")[:-1] + [f"stratified_loss_{cov}_shortcuts.csv"]
    savefile = pd.read_csv(os.path.join("/".join(savefile_name)))
    mae_diffs = []
    mse_diffs = []
    for i in range(len(files_same_pattern)):
        file = os.path.join("/".join(FILE.split("/")[:-1]), "_".join(pattern) + "_stra" + str(i) + "_shortcuts.csv")
        df = pd.read_csv(file)
        df_diff = df.values - df["labels"].values.reshape(-1, 1)
        mae = np.mean(np.abs(df_diff), axis=0).reshape(1, -1)
        mse = np.mean(np.square(df_diff), axis=0).reshape(1, -1)
        mae_diffs.append(mae)
        mse_diffs.append(mse)
    mae_diffs = np.concatenate(mae_diffs, axis=0)
    mse_diffs = np.concatenate(mse_diffs, axis=0)
    cols = df.columns[:-1]
    mae_diffs = pd.DataFrame(mae_diffs[:, :-1], columns=[f"{i}_mae" for i in cols])
    mse_diffs = pd.DataFrame(mse_diffs[:, :-1], columns=[f"{i}_mse" for i in cols])

    new_savefile = pd.concat([savefile, mae_diffs, mse_diffs], axis=1)
    new_savefile.to_csv(os.path.join("/".join(savefile_name)), index=False)