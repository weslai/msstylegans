import os
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--fname_ms", type=str, default=None)
@click.option("--fname_s1", type=str, default=None)
@click.option("--fname_s2", type=str, default=None)
@click.option("--covariate", type=str, default=None)
@click.option("--dataset", type=str, default=None)
@click.option("--task", type=str, default="regression")
@click.option("--save_dir", type=str, default="strata_mae_compare.csv")


def main(
    fname_ms,
    fname_s1,
    fname_s2,
    covariate,
    dataset,
    task,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    df_ms = pd.read_csv(fname_ms)
    df_s1 = pd.read_csv(fname_s1)
    df_s2 = pd.read_csv(fname_s2)

    if task == "regression":
        mae_ms = df_ms["mae"].values
        mae_s1 = df_s1["mae"].values
        mae_s2 = df_s2["mae"].values

        mse_ms = df_ms["mse"].values
        mse_s1 = df_s1["mse"].values
        mse_s2 = df_s2["mse"].values

        corr_ms = df_ms["corr"].values
        corr_s1 = df_s1["corr"].values
        corr_s2 = df_s2["corr"].values

        summary_df = pd.DataFrame(np.array([mae_ms, mae_s1, mae_s2,
                                            mse_ms, mse_s1, mse_s2,
                                            corr_ms, corr_s1, corr_s2]).reshape(1,-1),
                                            columns=["ms_mae", "s1_mae", "s2_mae", 
                                                    "ms_mse", "s1_mse", "s2_mse", 
                                                    "ms_corr", "s1_corr", "s2_corr"])
    elif task == "classification":
        accuracy_ms = df_ms["accuracy"].values
        accuracy_s1 = df_s1["accuracy"].values
        accuracy_s2 = df_s2["accuracy"].values

        precision_ms = df_ms["precision"].values
        precision_s1 = df_s1["precision"].values
        precision_s2 = df_s2["precision"].values

        recall_ms = df_ms["recall"].values
        recall_s1 = df_s1["recall"].values
        recall_s2 = df_s2["recall"].values

        f1_ms = df_ms["f1"].values
        f1_s1 = df_s1["f1"].values
        f1_s2 = df_s2["f1"].values

        corr_ms = df_ms["corr"].values
        corr_s1 = df_s1["corr"].values
        corr_s2 = df_s2["corr"].values


        summary_df = pd.DataFrame(np.array([accuracy_ms, accuracy_s1, accuracy_s2,
                                            precision_ms, precision_s1, precision_s2,
                                            recall_ms, recall_s1, recall_s2,
                                            f1_ms, f1_s1, f1_s2,
                                            corr_ms, corr_s1, corr_s2]).reshape(1,-1), 
                                            columns=["ms_accuracy", "s1_accuracy", "s2_accuracy",
                                                "ms_precision", "s1_precision", "s2_precision",
                                                "ms_recall", "s1_recall", "s2_recall",
                                                "ms_f1", "s1_f1", "s2_f1",
                                                "ms_corr", "s1_corr", "s2_corr"])
    else:
        raise NotImplementedError

    fname_w_avg = os.path.join(save_dir, f"summary_eval_loss_{covariate}_{dataset}.csv")
    summary_df.to_csv(fname_w_avg, index=False)
if __name__ == "__main__":
    main()