import os
import numpy as np
import pandas as pd

fname_ms = "/dhc/home/wei-cheng.lai/experiments/multisources/ukb/fids/ukb_half_log_lambda/fid_train/ms_stratified_fid.csv"
fname_s1 = "/dhc/home/wei-cheng.lai/experiments/singlesource/ukb/fids/source1_ventricle_log_lambda/fid_train/ms_stratified_fid.csv"
fname_s2 = "/dhc/home/wei-cheng.lai/experiments/singlesource/ukb/fids/source2_greymatter_log_lambda/fid_train/ms_stratified_fid.csv"

save_dir = "/dhc/home/wei-cheng.lai/experiments/multisources/ukb/fids/ukb_half_log_lambda/fids_final/"
os.makedirs(save_dir, exist_ok=True)

df_ms = pd.read_csv(fname_ms)
df_s1 = pd.read_csv(fname_s1)
df_s2 = pd.read_csv(fname_s2)

num_ms = df_ms["num_samples"].values
num_s1 = df_s1["num_samples"].values
num_s2 = df_s2["num_samples"].values

fid_ms = df_ms["fid_score"].values
fid_s1 = df_s1["fid_score"].values
fid_s2 = df_s2["fid_score"].values

w_fid_ms = fid_ms * (num_ms / np.sum(num_ms))
w_fid_s1 = fid_s1 * (num_s1 / np.sum(num_s1))
w_fid_s2 = fid_s2 * (num_s2 / np.sum(num_s2))

df_ms["w_fid"] = w_fid_ms
df_s1["w_fid"] = w_fid_s1
df_s2["w_fid"] = w_fid_s2

w_avg_fid_ms = np.sum(w_fid_ms).reshape(1,)
w_avg_fid_s1 = np.sum(w_fid_s1).reshape(1,)
w_avg_fid_s2 = np.sum(w_fid_s2).reshape(1,)

w_avg_fid_df = pd.DataFrame(np.array([w_avg_fid_ms, w_avg_fid_s1, w_avg_fid_s2]).reshape(1,-1), columns=["ms", "s1", "s2"])

fname_ms = os.path.join(save_dir, "ms_stratified_fid_weighted.csv")
fname_s1 = os.path.join(save_dir, "s1_stratified_fid_weighted.csv")
fname_s2 = os.path.join(save_dir, "s2_stratified_fid_weighted.csv")
df_ms.to_csv(fname_ms, index=False)
df_s1.to_csv(fname_s1, index=False)
df_s2.to_csv(fname_s2, index=False)

fname_w_avg = os.path.join(save_dir, "w_avg_fid.csv")
w_avg_fid_df.to_csv(fname_w_avg, index=False)
