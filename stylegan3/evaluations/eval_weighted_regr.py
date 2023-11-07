import os
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--fname_ms", type=str, default=None)
@click.option("--fname_s1", type=str, default=None)
@click.option("--fname_s2", type=str, default=None)
@click.option("--save_dir", type=str, default="strata_mae_compare.csv")


def main(
    fname_ms,
    fname_s1,
    fname_s2,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    df_ms = pd.read_csv(fname_ms)
    df_s1 = pd.read_csv(fname_s1)
    df_s2 = pd.read_csv(fname_s2)

    num_ms = df_ms["num_samples"].values
    num_s1 = df_s1["num_samples"].values
    num_s2 = df_s2["num_samples"].values

    mae_ms = df_ms["mae"].values
    mae_s1 = df_s1["mae"].values
    mae_s2 = df_s2["mae"].values

    mse_ms = df_ms["mse"].values
    mse_s1 = df_s1["mse"].values
    mse_s2 = df_s2["mse"].values

    w_mae_ms = mae_ms * (num_ms / np.sum(num_ms))
    w_mae_s1 = mae_s1 * (num_s1 / np.sum(num_s1))
    w_mae_s2 = mae_s2 * (num_s2 / np.sum(num_s2))

    w_mse_ms = mse_ms * (num_ms / np.sum(num_ms))
    w_mse_s1 = mse_s1 * (num_s1 / np.sum(num_s1))
    w_mse_s2 = mse_s2 * (num_s2 / np.sum(num_s2))

    df_ms["w_mae"] = w_mae_ms
    df_s1["w_mae"] = w_mae_s1
    df_s2["w_mae"] = w_mae_s2

    df_ms["w_mse"] = w_mse_ms
    df_s1["w_mse"] = w_mse_s1
    df_s2["w_mse"] = w_mse_s2

    w_avg_mae_ms = np.sum(w_mae_ms).reshape(1,)
    w_avg_mae_s1 = np.sum(w_mae_s1).reshape(1,)
    w_avg_mae_s2 = np.sum(w_mae_s2).reshape(1,)

    w_avg_mse_ms = np.sum(w_mse_ms).reshape(1,)
    w_avg_mse_s1 = np.sum(w_mse_s1).reshape(1,)
    w_avg_mse_s2 = np.sum(w_mse_s2).reshape(1,)

    avg_mae_ms = np.mean(mae_ms)
    avg_mae_s1 = np.mean(mae_s1)
    avg_mae_s2 = np.mean(mae_s2)
    std_mae_ms = np.std(mae_ms)
    std_mae_s1 = np.std(mae_s1)
    std_mae_s2 = np.std(mae_s2)

    avg_mse_ms = np.mean(mse_ms)
    avg_mse_s1 = np.mean(mse_s1)
    avg_mse_s2 = np.mean(mse_s2)
    std_mse_ms = np.std(mse_ms)
    std_mse_s1 = np.std(mse_s1)
    std_mse_s2 = np.std(mse_s2)

    w_avg_fid_df = pd.DataFrame(np.array([w_avg_mae_ms, w_avg_mae_s1, w_avg_mae_s2,
                                          w_avg_mse_ms, w_avg_mse_s1, w_avg_mse_s2]).reshape(1,-1), 
                                          columns=["ms_mae", "s1_mae", "s2_mae", "ms_mse", "s1_mse", "s2_mse"])
    avg_regr_df = pd.DataFrame(np.array([avg_mae_ms, avg_mae_s1, avg_mae_s2, 
                                         std_mae_ms, std_mae_s1, std_mae_s2,
                                         avg_mse_ms, avg_mse_s1, avg_mse_s2,
                                         std_mse_ms, std_mse_s1, std_mse_s2]).reshape(1,-1),
                                         columns=["ms_mae_mu", "s1_mae_mu", "s2_mae_mu",
                                                "ms_mae_std", "s1_mae_std", "s2_mae_std",
                                                "ms_mse_mu", "s1_mse_mu", "s2_mse_mu",
                                                "ms_mse_std", "s1_mse_std", "s2_mse_std"])

    fname_ms = os.path.join(save_dir, "ms_stratified_weighted.csv")
    fname_s1 = os.path.join(save_dir, "s1_stratified_weighted.csv")
    fname_s2 = os.path.join(save_dir, "s2_stratified_weighted.csv")
    df_ms.to_csv(fname_ms, index=False)
    df_s1.to_csv(fname_s1, index=False)
    df_s2.to_csv(fname_s2, index=False)

    fname_w_avg = os.path.join(save_dir, "w_avg.csv")
    fname_avg = os.path.join(save_dir, "regr_avg.csv")
    w_avg_fid_df.to_csv(fname_w_avg, index=False)
    avg_regr_df.to_csv(fname_avg, index=False)

if __name__ == "__main__":
    main()