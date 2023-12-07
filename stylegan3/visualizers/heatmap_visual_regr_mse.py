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

## --------------------------------------------------------- ##
def get_covs(dataset):
    if dataset == "retinal":
        COVS = {"c1": "age", "c2": "diastolic blood pressure", "c3": "spherical power"}
    elif dataset == "ukb":
        COVS = {"c1": "age", "c2": "ventricle", "c3": "grey matter"}
    elif dataset == "mnist-thickness-intensity-slant":
        COVS = {"c1": "thickness", "c2": "intensity", "c3": "slant"}
    return COVS

## --------------------------------------------------------- ##
def load_metrics(
    ms_path, 
    source1_path,
    source2_path,
    group_by = "c1"
):
    ms_df = pd.read_csv(ms_path)
    source1_df = pd.read_csv(source1_path)
    source2_df = pd.read_csv(source2_path)

    ## vmin, vmax
    mse_vmin = min(ms_df["gen_mse"].min(), ms_df["real_mse"].min(),
               source1_df["gen_mse"].min(), source2_df["gen_mse"].min())
    mse_vmax = max(ms_df["gen_mse"].max(), ms_df["real_mse"].max(),
               source1_df["gen_mse"].max(), source2_df["gen_mse"].max())
    
    mae_vmin = min(ms_df["gen_mae"].min(), ms_df["real_mae"].min(),
               source1_df["gen_mae"].min(), source2_df["gen_mae"].min())
    mae_vmax = max(ms_df["gen_mae"].max(), ms_df["real_mae"].max(),
               source1_df["gen_mae"].max(), source2_df["gen_mae"].max())

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
    for i, groups in enumerate([ms_df_groups, source1_df_groups, source2_df_groups]):
        for df in groups:
            cur_df = df["scores"]
            cur_df.index = np.arange(len(cur_df))
            df_cols = list(cur_df.keys())
            col = df_cols[1].split("_")[0]
            idx = df_cols[3].split("_")[0]

            score_matrix_mse = pd.DataFrame.pivot_table(cur_df, values="gen_mse", 
                                                        index=[f"{idx}_min"], columns=[f"{col}_min"])
            score_matrix_mae = pd.DataFrame.pivot_table(cur_df, values="gen_mae", 
                                                        index=[f"{idx}_min"], columns=[f"{col}_min"])
            cols = [f"{col}_{i}" for i in range(score_matrix_mse.shape[1])]
            idxs = [f"{idx}_{i}" for i in range(score_matrix_mse.shape[0])]
            score_matrix_mse.columns = cols
            score_matrix_mse.index = idxs
            df["mse_matrix"] = score_matrix_mse

            cols = [f"{col}_{i}" for i in range(score_matrix_mae.shape[1])]
            idxs = [f"{idx}_{i}" for i in range(score_matrix_mae.shape[0])]
            score_matrix_mae.columns = cols
            score_matrix_mae.index = idxs
            df["mae_matrix"] = score_matrix_mae

            if i == 0:
                score_matrix_mse = pd.DataFrame.pivot_table(cur_df, values="real_mse", 
                                                            index=[f"{idx}_min"], columns=[f"{col}_min"])
                score_matrix_mae = pd.DataFrame.pivot_table(cur_df, values="real_mae", 
                                                            index=[f"{idx}_min"], columns=[f"{col}_min"])
                cols = [f"{col}_{i}" for i in range(score_matrix_mse.shape[1])]
                idxs = [f"{idx}_{i}" for i in range(score_matrix_mse.shape[0])]
                score_matrix_mse.columns = cols
                score_matrix_mse.index = idxs
                df["mse_matrix_real"] = score_matrix_mse

                cols = [f"{col}_{i}" for i in range(score_matrix_mae.shape[1])]
                idxs = [f"{idx}_{i}" for i in range(score_matrix_mae.shape[0])]
                score_matrix_mae.columns = cols
                score_matrix_mae.index = idxs
                df["mae_matrix_real"] = score_matrix_mae

    return ms_df_groups, source1_df_groups, source2_df_groups, (mse_vmin, mae_vmin), (mse_vmax, mae_vmax)

## --------------------------------------------------------- ##
def plot_heatmap(dataset, groups, vmin, vmax, title, save_path):
    save_dir = os.path.join("/", *save_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)
    COVS = get_covs(dataset)
    for df in groups:
        sns.set_theme()
        sns.set_style("ticks")
        sns.set_context("paper")
        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=(9, 6))
        fig_heatmap = sns.heatmap(df["score_matrix"], annot=True, fmt=".1f", vmin=vmin, vmax=vmax, linewidths=.5, ax=ax)
        cur_title = title + f" {df['cov']} : {df['value']:.1f}"
        fig_heatmap.set_title(cur_title)
        xlabel = df["score_matrix"].columns[0].split("_")[0]
        cur_xlabel = xlabel + f" ({COVS[xlabel]})"
        ylabel = df["score_matrix"].index[0].split("_")[0]
        cur_ylabel = ylabel + f" ({COVS[ylabel]})"
        fig_heatmap.set(xlabel=cur_xlabel, ylabel=cur_ylabel)
        fig.savefig(save_path + "_{:s}_{:1f}.png".format(df['cov'], df['value']))
        plt.close(fig)

## --------------------------------------------------------- ##
def plot_heatmap_all(dataset, regr_cov, group1, group2, group3, vmin, vmax, save_path):
    metrics = ["mse", "mae"]
    save_dir = os.path.join("/", *save_path.split("/")[:-1], regr_cov)
    os.makedirs(save_dir, exist_ok=True)
    COVS = get_covs(dataset)
    title = ["source1 ", "multi source ", "source2 ", "real source "]

    for num, metric in enumerate(metrics):
        for i, (df2, df1, df3) in enumerate(zip(group2, group1, group3)):
            sns.set_theme()
            sns.set_style("ticks")
            sns.set_context("paper")
            sns.set_palette("colorblind")
            fig, ax = plt.subplots(nrows=1, ncols=5,
                                gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.08], 
                                             "height_ratios": [1]}, 
                                figsize=(6, 5))
            ax[0].get_shared_y_axes().join(ax[1], ax[2], ax[3])

            for j, df in enumerate([df2, df1, df3]):
                ax[j].set_aspect('equal')
                
                fig_heatmap = sns.heatmap(df[f"{metric}_matrix"], annot=True, fmt=".2f", vmin=vmin[num], vmax=vmax[num],
                                            linewidths=.5, ax=ax[j],
                                            cbar=False)
                if j == 1:
                    ax[3].set_aspect('equal')
                    fig_heatmap_real = sns.heatmap(df[f"{metric}_matrix_real"], annot=True, fmt=".2f", 
                                                    vmin=vmin[num], vmax=vmax[num],
                                                    linewidths=.5, ax=ax[3],
                                                    cbar_ax=ax[4],
                                                    cbar_kws={"shrink": .45, "pad": 0.01, 
                                                              "use_gridspec": False})
                    fig_heatmap_real.set_ylabel("")
                    fig_heatmap_real.set_xlabel("")
                    fig_heatmap_real.set_yticks([])
                    fig_heatmap_real.set_title(title[-1])
                fig_heatmap.set_ylabel("")
                fig_heatmap.set_xlabel("")
                if j != 0:
                    fig_heatmap.set_yticks([])

                cur_title = title[j] #+ f" {df['cov']} : {df['value']:.2f} "
                fig_heatmap.set_title(cur_title)
                if j == 0:
                    ylabel = df[f"{metric}_matrix"].index[0].split("_")[0]
                    cur_ylabel = ylabel + f" ({COVS[ylabel]})"
                    fig_heatmap.set(ylabel=cur_ylabel)
                    tx = fig_heatmap.get_yticklabels()
                    fig_heatmap.set_yticklabels(tx, rotation=90)
            xlabel = df1[f"{metric}_matrix"].columns[0].split("_")[0]
            cur_xlabel = xlabel + f" ({COVS[xlabel]})"
            fig.supxlabel(f"{cur_xlabel}", y=0.1, ha="center", fontsize=12)
            fig.savefig(save_dir + "/{:s}_subplots_{:s}_{:1f}.png".format(metric, df['cov'], df['value']))
            fig.savefig(save_dir + "/{:s}_subplots_{:s}_{:1f}.pdf".format(metric, df['cov'], df['value']))
            plt.close(fig)

## --------------------------------------------------------- ##
@click.command()
@click.option("--dataset", type=str, default="retinal")
@click.option("--ms_path", type=str, default=None)
@click.option("--source1_path", type=str, default=None)
@click.option("--source2_path", type=str, default=None)
@click.option("--group_by", type=str, default="c1")
@click.option("--regr_cov", type=str, default="age")
@click.option("--title", type=str, default="Multi-Source")
@click.option('--goal', type=str, default="ms")
@click.option("--save_path", type=str, default="heatmap")

def main(
    dataset,
    ms_path, 
    source1_path,
    source2_path,
    group_by = "c1",
    regr_cov = "age",
    goal = "ms",
    title = "Multi-Source",
    save_path = "heatmap"
):
    ms_df_groups, source1_df_groups, source2_df_groups, vmin, vmax = load_metrics(ms_path, 
                                                                                  source1_path, 
                                                                                  source2_path, 
                                                                                  group_by)
    if dataset == "retinal":
        assert regr_cov in ["age", "diastolic", "spherical"]
    elif dataset == "ukb":
        assert regr_cov in ["age", "ventricle", "grey_matter"]
    elif dataset == "mnist-thickness-intensity-slant":
        assert regr_cov in ["thickness", "intensity", "slant"]

    if goal == "ms":
        plot_heatmap(dataset, ms_df_groups, vmin, vmax, title, save_path)
    elif goal == "source1":
        plot_heatmap(dataset, source1_df_groups, vmin, vmax, title, save_path)
    elif goal == "source2":
        plot_heatmap(dataset, source2_df_groups, vmin, vmax, title, save_path)
    elif goal == "all":
        plot_heatmap_all(dataset, regr_cov, 
                         ms_df_groups, source1_df_groups, source2_df_groups, vmin, vmax, save_path)

if __name__ == "__main__":
    main()