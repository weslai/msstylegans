#### compare the strata MSE and MAE of the multisource and single source models (baseline)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##  read the data
DATASET = "mnist-thickness-intensity-slant"
COVARIANCES = ["thickness", "intensity", "slant"]
# COVARIANCES = ["age", "ventricle", "grey_matter"]

COMMON_COV_SCORES = "/dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_norm_mse_lambda/regr/thickness/new_opt/w_avg.csv"
COV1_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COVARIANCES[1], *COMMON_COV_SCORES.split("/")[-2:])
COV2_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COVARIANCES[2], *COMMON_COV_SCORES.split("/")[-2:])
print(COMMON_COV_SCORES)
print(COV1_SCORES)
print(COV2_SCORES)

METRICS = ["mse", "mae", "corr"]

def read_scores(scores_path, cov):
    df = pd.read_csv(scores_path)
    ms_dict = {}
    s1_dict = {}
    s2_dict = {}
    for metric in METRICS:
        ms_dict[metric] = df[f"ms_{metric}"].values
        s1_dict[metric] = df[f"s1_{metric}"].values
        s2_dict[metric] = df[f"s2_{metric}"].values

    ms_df = pd.DataFrame(ms_dict, index=[cov])
    s1_df = pd.DataFrame(s1_dict, index=[cov])
    s2_df = pd.DataFrame(s2_dict, index=[cov])
    return ms_df, s1_df, s2_df

ms_df = None
s1_df = None
s2_df = None
for score, cov in zip([COMMON_COV_SCORES, COV1_SCORES, COV2_SCORES], COVARIANCES):
    assert os.path.exists(score), f"{score} does not exist!"
    ms, s1, s2 = read_scores(score, cov)
    if ms_df is None:
        ms_df = ms
        s1_df = s1
        s2_df = s2
    else:
        ms_df = pd.concat([ms_df, ms], axis=0)
        s1_df = pd.concat([s1_df, s1], axis=0)
        s2_df = pd.concat([s2_df, s2], axis=0)

print("multisource")
print(ms_df)
print("source1")
print(s1_df)
print("source2")
print(s2_df)
# plot the table
for metric in METRICS:
    df = pd.concat([s1_df[metric], ms_df[metric], s2_df[metric]], axis=1)
    df.index.name = "Covariances"
    df.columns = pd.MultiIndex.from_product([["Source 1", "Multisource", "Source 2"], [metric]])
    print(df)
    fig = plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.5)
    # Customize label font size
    sns.set_context("paper")
    sns.set_style("whitegrid")
    ax = sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, linewidths=1, linecolor="white",
                     annot_kws={"fontsize":16})
    ax.set(xlabel="Metric")
    
    os.makedirs(os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], "new_opt"), exist_ok=True)
    strata_file = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], "new_opt", f"strata_{metric}.png")

    fig.savefig(strata_file, bbox_inches="tight", dpi=300)
