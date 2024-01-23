#### compare the strata MSE and MAE of the multisource and single source models (baseline)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# SETTING = "twomultisource_mri"
# SETTING = "mnist-thickness-intensity-slant"
# SETTING = "ukb"
# SETTING = "retinal"
COMMON_COV_SCORES = [
    "/dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_half_norm_mse_lambda/regr/thickness/new_opt/w_avg.csv",
    "/dhc/home/wei-cheng.lai/experiments/multisources/ukb/plots/ukb_half_log_lambda/regr/age/new_opt_strata/no_aug/w_avg.csv",
    "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/plots/ms_half_log_lambda/regr/age/new_opt_strata/corr/w_avg.csv"
]
COMMON_COV_SCORES_HIGH = [
    "/dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_norm_mse_lambda/regr/thickness/new_opt/w_avg.csv",
    "/dhc/home/wei-cheng.lai/experiments/multisources/ukb/plots/ukb_log_lambda/regr/age/new_opt_strata/no_aug/w_avg.csv",
    "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/plots/ms_log_lambda/regr/age/new_opt_strata/corr/w_avg.csv"
]
COVARIANCES_ALL = []
COV_SCORES_ALL = []
COV_SCORES_HIGH_ALL = []
DATASETS = ["morpho", "mri", "retinal"]
for SETTING, common_cov_score, common_cov_score_high in zip(["mnist-thickness-intensity-slant", "ukb", "retinal"], COMMON_COV_SCORES, COMMON_COV_SCORES_HIGH):
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
        COVARIANCES = ["age", "cataract", "spherical"]
        # COVARIANCES = ["cataract"]
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
    COVARIANCES_ALL.append(COVARIANCES)

    if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
        # COMMON_COV_SCORES = "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/plots/ms_half_log_lambda/regr/age/new_opt_strata/w_avg.csv"
        if DATASET == "mnist-thickness-intensity-slant":
            COV1_SCORES = os.path.join("/", *common_cov_score.split("/")[:-3], COVARIANCES[1], *common_cov_score.split("/")[-2:])
            COV2_SCORES = os.path.join("/", *common_cov_score.split("/")[:-3], COVARIANCES[2], *common_cov_score.split("/")[-2:])
            COV1_SCORES_HIGH = os.path.join("/", *common_cov_score_high.split("/")[:-3], COVARIANCES[1], *common_cov_score_high.split("/")[-2:])
            COV2_SCORES_HIGH = os.path.join("/", *common_cov_score_high.split("/")[:-3], COVARIANCES[2], *common_cov_score_high.split("/")[-2:])
        else:
            COV1_SCORES = os.path.join("/", *common_cov_score.split("/")[:-4], COVARIANCES[1], *common_cov_score.split("/")[-3:])
            # if DATASET != "retinal":
            COV2_SCORES = os.path.join("/", *common_cov_score.split("/")[:-4], COVARIANCES[2], *common_cov_score.split("/")[-3:])
            # COMMON_COV_SCORES_HIGH = "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/plots/ms_log_lambda/regr/age/new_opt_strata/w_avg.csv"
            COV1_SCORES_HIGH = os.path.join("/", *common_cov_score_high.split("/")[:-4], COVARIANCES[1], *common_cov_score_high.split("/")[-3:])
            # if DATASET != "retinal":
            COV2_SCORES_HIGH = os.path.join("/", *common_cov_score_high.split("/")[:-4], COVARIANCES[2], *common_cov_score_high.split("/")[-3:])

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
    COV_SCORES_ALL.append([common_cov_score, COV1_SCORES, COV2_SCORES])
    COV_SCORES_HIGH_ALL.append([common_cov_score_high, COV1_SCORES_HIGH, COV2_SCORES_HIGH])
    # COV_SCORES_ALL.append([common_cov_score, COV1_SCORES, COV2_SCORES] if DATASET != "retinal" else [common_cov_score, COV1_SCORES])
    # COV_SCORES_HIGH_ALL.append([common_cov_score_high, COV1_SCORES_HIGH, COV2_SCORES_HIGH] if DATASET != "retinal" else [common_cov_score_high, COV1_SCORES_HIGH])

if SETTING not in ["twomultisource_retinal", "threemultisource_retinal"]:
    METRICS = ["mse", "mae", "corr"]
    if SETTING == "retinal":
        METRICS = ["mse", "mae", "corr"]
        # METRICS = ["corr", "accuracy", "f1"]
        # METRICS = ["corr"]
        # METRICS = ["accuracy", "f1", "balanced_acc"]
else: #### TODO: add the metrics for retinal
    pass

def read_scores(scores_path, cov, dataset):
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
    if cov == "grey_matter":
        cov = "grey matter"
    elif cov == "spherical_power":
        cov = "spherical power"
    ms_df = pd.DataFrame(ms_dict, index=[dataset + " " + cov])
    s1_df = pd.DataFrame(s1_dict, index=[dataset + " " + cov])
    s2_df = pd.DataFrame(s2_dict, index=[dataset + " " + cov])
    return ms_df, s1_df, s2_df

ms_df = None
ms_high_df = None
s1_df = None
s2_df = None

if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
    # scores = [COMMON_COV_SCORES, COV1_SCORES]
    # scores_high = [COMMON_COV_SCORES_HIGH, COV1_SCORES_HIGH]
    scores = COV_SCORES_ALL
    scores_high = COV_SCORES_HIGH_ALL
    all_covairances = COVARIANCES_ALL
elif SETTING in ["twomultisource_mri", "twomultisource_retinal"]:
    scores = [COMMON_COV_SCORES, COV1_SCORES, COV2_SCORES, COV3_SCORES, COV4_SCORES, COV5_SCORES]
    scores_high = None
    all_covairances = COVARIANCES + COVARIANCES1

for score_ds, cov_ds, dataset in zip(scores, all_covairances, DATASETS):
    for score, cov in zip(score_ds, cov_ds):
        assert os.path.exists(score), f"{score} does not exist!"
        ms, s1, s2 = read_scores(score, cov, dataset)
        if ms_df is None:
            ms_df = ms
            s1_df = s1
            s2_df = s2
        else:
            ms_df = pd.concat([ms_df, ms], axis=0)
            s1_df = pd.concat([s1_df, s1], axis=0)
            s2_df = pd.concat([s2_df, s2], axis=0)
if scores_high is not None:
    for score_ds, cov_ds, dataset in zip(scores_high, all_covairances, DATASETS):
        for score, cov in zip(score_ds, cov_ds):
            assert os.path.exists(score), f"{score} does not exist!"
            ms_high, s1, s2 = read_scores(score, cov, dataset)
            if ms_high_df is None:
                ms_high_df = ms_high
            else:
                ms_high_df = pd.concat([ms_high_df, ms_high], axis=0)
print("multisource")
print(ms_df)
print("multisource high")
print(ms_high_df)
print("source1")
print(s1_df)
print("source2")
print(s2_df)
# plot the table
for metric in METRICS:
    if ms_high_df is not None:
        df = pd.concat([s1_df[metric], ms_df[metric], ms_high_df[metric], s2_df[metric]], axis=1)
    else:
        df = pd.concat([s1_df[metric], ms_df[metric], s2_df[metric]], axis=1)
    (vmin, vmax) = (0, 1) if metric == "corr" else (0, 1.5)
    df.index.name = "Covariances"
    if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:
        if metric == "corr":
            metric = "Pearson's Correlation"
        if ms_high_df is not None:
            # df.columns = pd.MultiIndex.from_product([["Source 1", "Multisource half", "Multisource high", "Source 2"], [metric]])
            df.columns = ["Source 1", "half \n MS", "high \n MS", "Source 2"]
        else:
            df.columns = pd.MultiIndex.from_product([["Source 1", "Multisource", "Source 2"], [metric]])
    elif SETTING in ["twomultisource_mri", "twomultisource_retinal"]:
        df.columns = pd.MultiIndex.from_product([[DATASET, "Multisource", DATASET1], [metric]])
    print(df)

    fig, axes = plt.subplots(3, 1, figsize=(23, 20), sharex=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    cbar_ax.tick_params(labelsize=40)
    sns.set(font_scale=1.5)
    # Customize label font size
    sns.set_context("paper")
    sns.set_style("whitegrid")
    # Plot the first heatmap
    g1 = sns.heatmap(df.iloc[:3, :], annot=True, fmt=".3f", vmin=vmin, vmax=vmax, cmap="Greens", 
                     cbar=False, cbar_ax=None, linewidths=1, linecolor="white", ax=axes[0],
                     annot_kws={"fontsize":50})
    g1.set(xlabel=" ", ylabel=" ")
    g1.set_yticklabels(g1.get_yticklabels(), rotation = 0, horizontalalignment='right', fontsize=45)
    # Plot the second heatmap
    g2 = sns.heatmap(df.iloc[3:6, :], annot=True, fmt=".3f", vmin=vmin, vmax=vmax, cmap="Greens", 
                     cbar=False, cbar_ax=None, linewidths=1, linecolor="white", ax=axes[1],
                     annot_kws={"fontsize":50})
    g2.set(xlabel= " ", ylabel=" ")
    g2.set_yticklabels(g2.get_yticklabels(), rotation = 0, horizontalalignment='right', fontsize=45)
    
    # Plot the third heatmap
    g3 = sns.heatmap(df.iloc[6:, :], annot=True, fmt=".3f", vmin=vmin, vmax=vmax, cmap="Greens", 
                     cbar=True, cbar_ax=cbar_ax, linewidths=1, linecolor="white", ax=axes[2],
                     annot_kws={"fontsize":50})
    
    g3.set(xlabel= " ", ylabel=" ")
    g3.set_xticklabels(g3.get_xticklabels(), rotation=0, fontsize=45)
    g3.set_yticklabels(g3.get_yticklabels(), rotation = 0, horizontalalignment='right', fontsize=45)
    # ax = sns.heatmap(df, annot=True, fmt=".3f", vmin=vmin, vmax=vmax, cmap="Greens", 
    #                  cbar=True, linewidths=1, linecolor="white",
    #                  annot_kws={"fontsize":18},
    #                  )
    # ax.set(xlabel= "", ylabel="")
    # # ax.set_xlabel(ax.get_xlabel(), fontsize=25)
    # # ax.set_ylabel(ax.get_ylabel(), fontsize=25)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=15)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=15, horizontalalignment='right', fontsize=17)
    fig.tight_layout(rect=[0, 0, .9, 1])
    if SETTING not in ["twomultisource_mri", "twomultisource_retinal", "threemultisource_mri", "threemultisource_retinal"]:    
        os.makedirs(os.path.join("/", *COMMON_COV_SCORES[-1].split("/")[:-4], "new_opt_strata"), exist_ok=True)
        strata_file = os.path.join("/", *COMMON_COV_SCORES[-1].split("/")[:-4], "new_opt_strata", f"strata_{metric}.png")
        strata_file_pdf = strata_file.replace(".png", ".pdf")
    else:
        directory = "mri" if DATASET1 == "adni" else "retinal" if DATASET1 == "rfmid" else "unknown"
        directory = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-6], "plots", directory, "new_opt_general")
        os.makedirs(directory, exist_ok=True)
        strata_file = os.path.join(directory, f"general_covariate_{metric}.png")
        strata_file_pdf = strata_file.replace(".png", ".pdf")
    fig.savefig(strata_file, bbox_inches="tight", dpi=300)
    fig.savefig(strata_file_pdf, bbox_inches="tight", dpi=300)
