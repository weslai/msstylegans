#### compare the strata MSE and MAE of the multisource and single source models (baseline)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SETTING = "twomultisource_mri"

##  read the data
### Semi multisource setting
if SETTING == "mnist-thickness-intensity-slant":
    DATASET = "mnist-thickness-intensity-slant"
    COVARIANCES = ["thickness", "intensity", "slant"]
elif SETTING == "ukb":
    DATASET = "ukb"
    COVARIANCES = ["age", "ventricle", "grey_matter"]
elif SETTING == "retinal":
    DATASET = "retinal"
    COVARIANCES = ["age", "cataract", "spherical_power"]
## Multisource setting
elif SETTING == ["twomultisource_mri", "threemultisource_mri"]:
    DATASET = "ukb"
    COVARIANCES = ["age", "ventricle", "grey_matter"]
    DATASET1 = "adni"
    COVARIANCES1 = ["age", "left_hippocampus", "right_hippocampus"]
    if SETTING == "threemultisource_mri":
        DATASET2 = "nacc"
        #### TODO: add the covariances
        COVARIANCES2 = ["age", "left_hippocampus", "right_hippocampus"]
elif SETTING in ["twomultisource_retinal", "threemultisource_retinal"]:
    DATASET = "ukb"
    COVARIANCES = ["age", "cataract", "spherical_power"]
    DATASET1 = "rfmid"
    COVARIANCES1 = ["disease_risk", "MH", "TSLN"]
    if SETTING == "threemultisource_mri":
        DATASET2 = "eyepacs"
        COVARIANCES2 = ["DR_level"]

if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
    COMMON_COV_SCORES = "/dhc/home/wei-cheng.lai/experiments/multisources/ukb/plots/ukb_log_lambda/regr/age/new_opt_strata/w_avg.csv"
    COV1_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COVARIANCES[1], *COMMON_COV_SCORES.split("/")[-2:])
    COV2_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COVARIANCES[2], *COMMON_COV_SCORES.split("/")[-2:])
    print(COMMON_COV_SCORES)
    print(COV1_SCORES)
    print(COV2_SCORES)

elif SETTING in ["twomultisource_mri", "twomultisource_retinal"]:
    COMMON_COV_SCORES = "/dhc/home/wei-cheng.lai/experiments/multisources/real_ms/mri_imgs/mse_re128/regression/opt_general/ukb/general_test_loss_age.csv"
    COV1_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-1], f"general_test_loss_{COVARIANCES[1]}")
    COV2_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-1], f"general_test_loss_{COVARIANCES[2]}")
    print(COMMON_COV_SCORES)
    print(COV1_SCORES)
    print(COV2_SCORES)
    COV3_SCORES = "/dhc/home/wei-cheng.lai/experiments/multisources/real_ms/mri_imgs/mse_re128/regression/opt_general/adni/general_test_loss_age.csv"
    COV4_SCORES = os.path.join("/", *COV3_SCORES.split("/")[:-1], f"general_test_loss_{COVARIANCES1[1]}")
    COV5_SCORES = os.path.join("/", *COV3_SCORES.split("/")[:-1], f"general_test_loss_{COVARIANCES1[2]}")
    print(COV3_SCORES)
    print(COV4_SCORES)
    print(COV5_SCORES)

if SETTING not in ["twomultisource_retinal", "threemultisource_retinal"]:
    METRICS = ["mse", "mae", "corr"]

def read_scores(scores_path, cov):
    df = pd.read_csv(scores_path)
    ms_dict = {}
    s1_dict = {}
    s2_dict = {}
    for metric in METRICS:
        if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
            ms_dict[metric] = df[f"ms_{metric}"].values
            s1_dict[metric] = df[f"s1_{metric}"].values
            s2_dict[metric] = df[f"s2_{metric}"].values
        else:
            ms_dict[metric] = df[metric].values
            s1_dict[metric] = df[metric].values
            s2_dict[metric] = df[metric].values

    ms_df = pd.DataFrame(ms_dict, index=[cov])
    s1_df = pd.DataFrame(s1_dict, index=[cov])
    s2_df = pd.DataFrame(s2_dict, index=[cov])
    return ms_df, s1_df, s2_df

ms_df = None
s1_df = None
s2_df = None

if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
    scores = [COMMON_COV_SCORES, COV1_SCORES, COV2_SCORES]
    all_covairances = COVARIANCES
elif SETTING in ["twomultisource_mri", "twomultisource_retinal"]:
    scores = [COMMON_COV_SCORES, COV1_SCORES, COV2_SCORES, COV3_SCORES, COV4_SCORES, COV5_SCORES]
    all_covairances = COVARIANCES + COVARIANCES1
for score, cov in zip(scores, all_covairances):
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
    if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
        df.columns = pd.MultiIndex.from_product([["Source 1", "Multisource", "Source 2"], [metric]])
    elif SETTING in ["twomultisource_mri", "twomultisource_retinal"]:
        df.columns = pd.MultiIndex.from_product([[DATASET, "Multisource", DATASET1], [metric]])
    print(df)
    fig = plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.5)
    # Customize label font size
    sns.set_context("paper")
    sns.set_style("whitegrid")
    ax = sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, linewidths=1, linecolor="white",
                     annot_kws={"fontsize":16})
    ax.set(xlabel="Metric")
    
    if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:    
        os.makedirs(os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], "new_opt_strata"), exist_ok=True)
        strata_file = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], "new_opt_strata", f"strata_{metric}.png")
    else:
        directory = "mri" if DATASET1 == "adni" else "retinal" if DATASET1 == "rfmid" else "unknown"
        directory = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-6], "plots", directory, "new_opt_general")
        os.makedirs(directory, exist_ok=True)
        strata_file = os.path.join(directory, f"general_covariate_{metric}.png")
    fig.savefig(strata_file, bbox_inches="tight", dpi=300)
