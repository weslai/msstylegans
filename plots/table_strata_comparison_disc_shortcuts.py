#### compare the strata MSE and MAE of the multisource and single source models (baseline)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##  read the data
DATASET = "morpho-mnist"
COVARIANCES = ["thickness", "intensity", "slant"]
# COVARIANCES = ["age", "ventricle", "grey_matter"]

COMMON_COV_SCORES = "/dhc/home/wei-cheng.lai/experiments/multisources/morpho/plots/estimated_norm_mse_lambda/regr/thickness/new_opt_disc_shortcut/w_avg.csv"
COV1_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COVARIANCES[1], *COMMON_COV_SCORES.split("/")[-2:])
COV2_SCORES = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COVARIANCES[2], *COMMON_COV_SCORES.split("/")[-2:])
print(COMMON_COV_SCORES)
print(COV1_SCORES)
print(COV2_SCORES)

METRICS = ["mse", "mae"]

def read_scores(scores_path, cov):
    df = pd.read_csv(scores_path)
    real_ms_dict = {}
    gen_ms_dict = {}
    for metric in METRICS:
        real_ms_dict[f"real_{metric}"] = df[f"real_ms_{metric}"].values
        real_ms_dict[f"real_disc_{metric}"] = df[f"real_disc_{metric}"].values
        real_ms_dict[f"real_predict_{metric}"] = df[f"real_predict_{metric}"].values

        gen_ms_dict[f"gen_{metric}"] = df[f"gen_ms_{metric}"].values
        gen_ms_dict[f"gen_disc_{metric}"] = df[f"gen_disc_{metric}"].values
        gen_ms_dict[f"gen_predict_{metric}"] = df[f"gen_predict_{metric}"].values

    real_ms_df = pd.DataFrame(real_ms_dict, index=[cov])
    gen_ms_df = pd.DataFrame(gen_ms_dict, index=[cov])
    return real_ms_df, gen_ms_df

real_ms_df = None
gen_ms_df = None
for score, cov in zip([COMMON_COV_SCORES, COV1_SCORES, COV2_SCORES], COVARIANCES):
    assert os.path.exists(score), f"{score} does not exist!"
    real_ms, gen_ms = read_scores(score, cov)
    if real_ms_df is None:
        real_ms_df = real_ms
        gen_ms_df = gen_ms
    
    else:
        real_ms_df = pd.concat([real_ms_df, real_ms], axis=0)
        gen_ms_df = pd.concat([gen_ms_df, gen_ms], axis=0)

print("Real multisource")
print(real_ms_df)
print("Gen multisource")
print(gen_ms_df)

# plot the table
for ms_df, cond in zip([real_ms_df, gen_ms_df], ["real", "gen"]):
    df = ms_df.copy()
    df.index.name = "Covariances"
    # df.columns = pd.MultiIndex.from_product([["Real Multisource", "Gen Multisource"], [metric]])
    print(df)
    fig = plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.5)
    # Customize label font size
    sns.set_context("paper")
    sns.set_style("whitegrid")
    ax = sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True,
                    linewidths=1, linecolor="white",
                    annot_kws={"fontsize":16})
    ax.set(xlabel="Metric")
    
    os.makedirs(os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COMMON_COV_SCORES.split("/")[-2]), exist_ok=True)
    strata_file = os.path.join("/", *COMMON_COV_SCORES.split("/")[:-3], COMMON_COV_SCORES.split("/")[-2], f"strata_{cond}.png")

    fig.savefig(strata_file, bbox_inches="tight", dpi=300)
