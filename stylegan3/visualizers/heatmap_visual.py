### Third-party ###
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import click

DATASET = "retinal"
MS_FID_PATH = "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/fids/ms_stratified_fid.csv"
SOURCE1_FID_PATH = "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/fids/source1_fid.csv"
SOURCE2_FID_PATH = "/dhc/home/wei-cheng.lai/experiments/multisources/retinal/fids/source2_fid.csv"

def get_covs(dataset):
    if dataset == "retinal":
        COVS = {"c1": "age", "c2": "diastolic bp", "c3": "spherical power left"}
    elif dataset == "ukb":
        pass
    elif dataset == "mnist-thickness-intensity-slant":
        COVS = {"c1": "thickness", "c2": "intensity", "c3": "slant"}
    return COVS

@click.command()
@click.option("--dataset", type=str, default="retinal")
@click.option("--ms_path", type=str, default=None)
@click.option("--source1_path", type=str, default=None)
@click.option("--source2_path", type=str, default=None)
@click.option("--group_by", type=str, default="c1")
@click.option("--title", type=str, default="Multi-Source")
@click.option('--goal', type=str, default="ms")
@click.option("--save_path", type=str, default="heatmap")

def main(
    dataset,
    ms_path, 
    source1_path,
    source2_path,
    group_by = "c1",
    goal = "ms",
    title = "Multi-Source",
    save_path = "heatmap"
):
    ms_df_groups, source1_df_groups, source2_df_groups, vmin, vmax = load_fids(ms_path, source1_path, source2_path, group_by)
    if goal == "ms":
        plot_heatmap(dataset, ms_df_groups, vmin, vmax, title, save_path)
    elif goal == "source1":
        plot_heatmap(dataset, source1_df_groups, vmin, vmax, title, save_path)
    elif goal == "source2":
        plot_heatmap(dataset, source2_df_groups, vmin, vmax, title, save_path)

def load_fids(
    ms_path, 
    source1_path,
    source2_path,
    group_by = "c1"
):
    ms_df = pd.read_csv(ms_path)
    source1_df = pd.read_csv(source1_path)
    source2_df = pd.read_csv(source2_path)

    ## vmin, vmax
    vmin = min(ms_df["fid_score"].min(), source1_df["fid_score"].min(), source2_df["fid_score"].min())
    vmax = max(ms_df["fid_score"].max(), source1_df["fid_score"].max(), source2_df["fid_score"].max())

    ## group by
    ms_df_groups = []
    source1_df_groups = []
    source2_df_groups = []
    for c in ms_df[f"{group_by}_min"].unique():
        cur_df = ms_df.loc[ms_df[f"{group_by}_min"] == c].drop(columns=[f"{group_by}_min", f"{group_by}_max"])
        ms_df_groups.append({"cov": group_by, "value": c, "scores": cur_df})
        cur_df = source1_df.loc[source1_df[f"{group_by}_min"] == c].drop(columns=[f"{group_by}_min", f"{group_by}_max"])
        source1_df_groups.append({"cov": group_by, "value": c, "scores": cur_df})
        cur_df = source2_df.loc[source2_df[f"{group_by}_min"] == c].drop(columns=[f"{group_by}_min", f"{group_by}_max"])
        source2_df_groups.append({"cov": group_by, "value": c, "scores": cur_df})
    
    ## make score matrixs
    for groups in [ms_df_groups, source1_df_groups, source2_df_groups]:
        for df in groups:
            cur_df = df["scores"]
            cur_df.index = np.arange(len(cur_df))
            df_cols = list(cur_df.keys())
            col = df_cols[-2].split("_")[0]
            idx = df_cols[-4].split("_")[0]
            score_matrix = pd.DataFrame.pivot_table(cur_df, values="fid_score", index=[f"{idx}_min"], columns=[f"{col}_min"])
            cols = [f"{col}_{i}" for i in range(score_matrix.shape[0])]
            idxs = [f"{idx}_{i}" for i in range(score_matrix.shape[1])]
            score_matrix.columns = cols
            score_matrix.index = idxs
            df["score_matrix"] = score_matrix
    return ms_df_groups, source1_df_groups, source2_df_groups, vmin, vmax

def plot_heatmap(dataset, groups, vmin, vmax, title, save_path):
    save_dir = os.path.join("/", *save_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)
    COVS = get_covs(dataset)
    for df in groups:
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(9, 6))
        fig_heatmap = sns.heatmap(df["score_matrix"], annot=True, fmt=".1f", vmin=vmin, vmax=vmax, linewidths=.5, ax=ax)
        cur_title = title + f" {df['cov']} : {df['value']}"
        fig_heatmap.set_title(cur_title)
        xlabel = df["score_matrix"].columns[0].split("_")[0]
        cur_xlabel = xlabel + f" ({COVS[xlabel]})"
        ylabel = df["score_matrix"].index[0].split("_")[0]
        cur_ylabel = ylabel + f" ({COVS[ylabel]})"
        fig_heatmap.set(xlabel=cur_xlabel, ylabel=cur_ylabel)
        fig.savefig(save_path + "_{:s}_{:1f}.png".format(df['cov'], df['value']))
        plt.close(fig)

if __name__ == "__main__":
    main()