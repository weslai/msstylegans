import os
import numpy as np
import torch
import pandas as pd
import glob

stratified_file = "/dhc/home/wei-cheng.lai/experiments/singlesource/retinal/regression/source2_spherical_log_lambda/new_opt_strata/corr/stratified_loss_cataract.csv"
cataract_folder = "/dhc/home/wei-cheng.lai/experiments/singlesource/retinal/regression/source2_spherical_log_lambda/new_opt_strata/corr"

path_name = os.path.join(cataract_folder, "stratified_predictions_cataract_stra*")
path_list = sorted(glob.glob(path_name))

stratified_df = pd.read_csv(stratified_file)
print(stratified_df.head())
gen_corr = []
for path in path_list:
    df = pd.read_csv(path)
    gen_predict = df["gen_predict"].values
    real_predict = df["real_predict"].values
    gen_predict = torch.sigmoid(torch.tensor(gen_predict)).numpy()
    real_predict = torch.sigmoid(torch.tensor(real_predict)).numpy()
    real_predict = np.where(real_predict > 0.5, 1, 0)
    labels = df["labels"].values
    corr = np.corrcoef(gen_predict, labels)
    gen_corr.append(corr[0, 1])

stratified_df["corr"] = gen_corr
stratified_df.to_csv(stratified_file, index=False)