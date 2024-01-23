import os
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--fname_ms", type=str, default=None)
@click.option("--fname_s1", type=str, default=None)
@click.option("--fname_s2", type=str, default=None)
@click.option("--task", type=str, default="regression")
@click.option("--save_dir", type=str, default="strata_mae_compare.csv")


def main(
    fname_ms,
    fname_s1,
    fname_s2,
    task,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    df_ms = pd.read_csv(fname_ms)
    df_s1 = pd.read_csv(fname_s1)
    df_s2 = pd.read_csv(fname_s2)

    num_ms = df_ms["num_samples"].values
    num_s1 = df_s1["num_samples"].values
    num_s2 = df_s2["num_samples"].values
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

        w_mae_ms = mae_ms * (num_ms / np.sum(num_ms))
        w_mae_s1 = mae_s1 * (num_s1 / np.sum(num_s1))
        w_mae_s2 = mae_s2 * (num_s2 / np.sum(num_s2))

        w_mse_ms = mse_ms * (num_ms / np.sum(num_ms))
        w_mse_s1 = mse_s1 * (num_s1 / np.sum(num_s1))
        w_mse_s2 = mse_s2 * (num_s2 / np.sum(num_s2))

        w_corr_ms = corr_ms * (num_ms / np.sum(num_ms))
        w_corr_s1 = corr_s1 * (num_s1 / np.sum(num_s1))
        w_corr_s2 = corr_s2 * (num_s2 / np.sum(num_s2))

        df_ms["w_mae"] = w_mae_ms
        df_s1["w_mae"] = w_mae_s1
        df_s2["w_mae"] = w_mae_s2

        df_ms["w_mse"] = w_mse_ms
        df_s1["w_mse"] = w_mse_s1
        df_s2["w_mse"] = w_mse_s2

        df_ms["w_corr"] = w_corr_ms
        df_s1["w_corr"] = w_corr_s1
        df_s2["w_corr"] = w_corr_s2

        w_avg_mae_ms = np.sum(w_mae_ms).reshape(1,)
        w_avg_mae_s1 = np.sum(w_mae_s1).reshape(1,)
        w_avg_mae_s2 = np.sum(w_mae_s2).reshape(1,)

        w_avg_mse_ms = np.sum(w_mse_ms).reshape(1,)
        w_avg_mse_s1 = np.sum(w_mse_s1).reshape(1,)
        w_avg_mse_s2 = np.sum(w_mse_s2).reshape(1,)

        w_avg_corr_ms = np.sum(w_corr_ms).reshape(1,)
        w_avg_corr_s1 = np.sum(w_corr_s1).reshape(1,)
        w_avg_corr_s2 = np.sum(w_corr_s2).reshape(1,)

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

        avg_corr_ms = np.mean(corr_ms)
        avg_corr_s1 = np.mean(corr_s1)
        avg_corr_s2 = np.mean(corr_s2)
        std_corr_ms = np.std(corr_ms)
        std_corr_s1 = np.std(corr_s1)
        std_corr_s2 = np.std(corr_s2)

        w_avg_df = pd.DataFrame(np.array([w_avg_mae_ms, w_avg_mae_s1, w_avg_mae_s2,
                                            w_avg_mse_ms, w_avg_mse_s1, w_avg_mse_s2,
                                            w_avg_corr_ms, w_avg_corr_s1, w_avg_corr_s2]).reshape(1,-1), 
                                            columns=["ms_mae", "s1_mae", "s2_mae", 
                                                    "ms_mse", "s1_mse", "s2_mse", 
                                                    "ms_corr", "s1_corr", "s2_corr"])
        avg_regr_df = pd.DataFrame(np.array([avg_mae_ms, avg_mae_s1, avg_mae_s2, 
                                            std_mae_ms, std_mae_s1, std_mae_s2,
                                            avg_mse_ms, avg_mse_s1, avg_mse_s2,
                                            std_mse_ms, std_mse_s1, std_mse_s2,
                                            avg_corr_ms, avg_corr_s1, avg_corr_s2,
                                            std_corr_ms, std_corr_s1, std_corr_s2]).reshape(1,-1),
                                            columns=["ms_mae_mu", "s1_mae_mu", "s2_mae_mu",
                                                    "ms_mae_std", "s1_mae_std", "s2_mae_std",
                                                    "ms_mse_mu", "s1_mse_mu", "s2_mse_mu",
                                                    "ms_mse_std", "s1_mse_std", "s2_mse_std",
                                                    "ms_corr_mu", "s1_corr_mu", "s2_corr_mu",
                                                    "ms_corr_std", "s1_corr_std", "s2_corr_std"])
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

        corr_ms = df_ms["balanced_acc"].values
        corr_s1 = df_s1["balanced_acc"].values
        corr_s2 = df_s2["balanced_acc"].values

        w_accuracy_ms = accuracy_ms * (num_ms / np.sum(num_ms))
        w_accuracy_s1 = accuracy_s1 * (num_s1 / np.sum(num_s1))
        w_accuracy_s2 = accuracy_s2 * (num_s2 / np.sum(num_s2))

        w_precision_ms = precision_ms * (num_ms / np.sum(num_ms))
        w_precision_s1 = precision_s1 * (num_s1 / np.sum(num_s1))
        w_precision_s2 = precision_s2 * (num_s2 / np.sum(num_s2))

        w_recall_ms = recall_ms * (num_ms / np.sum(num_ms))
        w_recall_s1 = recall_s1 * (num_s1 / np.sum(num_s1))
        w_recall_s2 = recall_s2 * (num_s2 / np.sum(num_s2))

        w_f1_ms = f1_ms * (num_ms / np.sum(num_ms))
        w_f1_s1 = f1_s1 * (num_s1 / np.sum(num_s1))
        w_f1_s2 = f1_s2 * (num_s2 / np.sum(num_s2))

        w_corr_ms = corr_ms * (num_ms / np.sum(num_ms))
        w_corr_s1 = corr_s1 * (num_s1 / np.sum(num_s1))
        w_corr_s2 = corr_s2 * (num_s2 / np.sum(num_s2))

        df_ms["w_accuracy"] = w_accuracy_ms
        df_s1["w_accuracy"] = w_accuracy_s1
        df_s2["w_accuracy"] = w_accuracy_s2

        df_ms["w_precision"] = w_precision_ms
        df_s1["w_precision"] = w_precision_s1
        df_s2["w_precision"] = w_precision_s2

        df_ms["w_recall"] = w_recall_ms
        df_s1["w_recall"] = w_recall_s1
        df_s2["w_recall"] = w_recall_s2

        df_ms["w_f1"] = w_f1_ms
        df_s1["w_f1"] = w_f1_s1
        df_s2["w_f1"] = w_f1_s2

        df_ms["w_balanced_acc"] = w_corr_ms
        df_s1["w_balanced_acc"] = w_corr_s1
        df_s2["w_balanced_acc"] = w_corr_s2

        w_avg_accuracy_ms = np.sum(w_accuracy_ms).reshape(1,)
        w_avg_accuracy_s1 = np.sum(w_accuracy_s1).reshape(1,)
        w_avg_accuracy_s2 = np.sum(w_accuracy_s2).reshape(1,)

        w_avg_precision_ms = np.sum(w_precision_ms).reshape(1,)
        w_avg_precision_s1 = np.sum(w_precision_s1).reshape(1,)
        w_avg_precision_s2 = np.sum(w_precision_s2).reshape(1,)

        w_avg_recall_ms = np.sum(w_recall_ms).reshape(1,)
        w_avg_recall_s1 = np.sum(w_recall_s1).reshape(1,)
        w_avg_recall_s2 = np.sum(w_recall_s2).reshape(1,)

        w_avg_f1_ms = np.sum(w_f1_ms).reshape(1,)
        w_avg_f1_s1 = np.sum(w_f1_s1).reshape(1,)
        w_avg_f1_s2 = np.sum(w_f1_s2).reshape(1,)

        w_avg_corr_ms = np.sum(w_corr_ms).reshape(1,)
        w_avg_corr_s1 = np.sum(w_corr_s1).reshape(1,)
        w_avg_corr_s2 = np.sum(w_corr_s2).reshape(1,)

        avg_accuracy_ms = np.mean(accuracy_ms)
        avg_accuracy_s1 = np.mean(accuracy_s1)
        avg_accuracy_s2 = np.mean(accuracy_s2)
        std_accuracy_ms = np.std(accuracy_ms)
        std_accuracy_s1 = np.std(accuracy_s1)
        std_accuracy_s2 = np.std(accuracy_s2)

        avg_precision_ms = np.mean(precision_ms)
        avg_precision_s1 = np.mean(precision_s1)
        avg_precision_s2 = np.mean(precision_s2)
        std_precision_ms = np.std(precision_ms)
        std_precision_s1 = np.std(precision_s1)
        std_precision_s2 = np.std(precision_s2)

        avg_recall_ms = np.mean(recall_ms)
        avg_recall_s1 = np.mean(recall_s1)
        avg_recall_s2 = np.mean(recall_s2)
        std_recall_ms = np.std(recall_ms)
        std_recall_s1 = np.std(recall_s1)
        std_recall_s2 = np.std(recall_s2)

        avg_f1_ms = np.mean(f1_ms)
        avg_f1_s1 = np.mean(f1_s1)
        avg_f1_s2 = np.mean(f1_s2)
        std_f1_ms = np.std(f1_ms)
        std_f1_s1 = np.std(f1_s1)
        std_f1_s2 = np.std(f1_s2)

        avg_corr_ms = np.mean(corr_ms)
        avg_corr_s1 = np.mean(corr_s1)
        avg_corr_s2 = np.mean(corr_s2)
        std_corr_ms = np.std(corr_ms)
        std_corr_s1 = np.std(corr_s1)
        std_corr_s2 = np.std(corr_s2)

        w_avg_df = pd.DataFrame(np.array([w_avg_accuracy_ms, w_avg_accuracy_s1, w_avg_accuracy_s2,
                                        w_avg_precision_ms, w_avg_precision_s1, w_avg_precision_s2,
                                        w_avg_recall_ms, w_avg_recall_s1, w_avg_recall_s2,
                                        w_avg_f1_ms, w_avg_f1_s1, w_avg_f1_s2,
                                        w_avg_corr_ms, w_avg_corr_s1, w_avg_corr_s2]).reshape(1,-1), 
                                        columns=["ms_accuracy", "s1_accuracy", "s2_accuracy",
                                                "ms_precision", "s1_precision", "s2_precision",
                                                "ms_recall", "s1_recall", "s2_recall",
                                                "ms_f1", "s1_f1", "s2_f1",
                                                "ms_balanced_acc", "s1_balanced_acc", "s2_balanced_acc"])
        avg_regr_df = pd.DataFrame(np.array([avg_accuracy_ms, avg_accuracy_s1, avg_accuracy_s2,
                                            std_accuracy_ms, std_accuracy_s1, std_accuracy_s2,
                                            avg_precision_ms, avg_precision_s1, avg_precision_s2,
                                            std_precision_ms, std_precision_s1, std_precision_s2,
                                            avg_recall_ms, avg_recall_s1, avg_recall_s2,
                                            std_recall_ms, std_recall_s1, std_recall_s2,
                                            avg_f1_ms, avg_f1_s1, avg_f1_s2,
                                            std_f1_ms, std_f1_s1, std_f1_s2,
                                            avg_corr_ms, avg_corr_s1, avg_corr_s2,
                                            std_corr_ms, std_corr_s1, std_corr_s2]).reshape(1,-1),
                                            columns=["ms_accuracy_mu", "s1_accuracy_mu", "s2_accuracy_mu",
                                                    "ms_accuracy_std", "s1_accuracy_std", "s2_accuracy_std",
                                                    "ms_precision_mu", "s1_precision_mu", "s2_precision_mu",
                                                    "ms_precision_std", "s1_precision_std", "s2_precision_std",
                                                    "ms_recall_mu", "s1_recall_mu", "s2_recall_mu",
                                                    "ms_recall_std", "s1_recall_std", "s2_recall_std",
                                                    "ms_f1_mu", "s1_f1_mu", "s2_f1_mu",
                                                    "ms_f1_std", "s1_f1_std", "s2_f1_std",
                                                    "ms_balanced_acc_mu", "s1_balanced_acc_mu", "s2_balanced_acc_mu",
                                                    "ms_balanced_acc_std", "s1_balanced_acc_std", "s2_balanced_acc_std"])
    else:
        raise NotImplementedError

    fname_ms = os.path.join(save_dir, "ms_stratified_weighted.csv")
    fname_s1 = os.path.join(save_dir, "s1_stratified_weighted.csv")
    fname_s2 = os.path.join(save_dir, "s2_stratified_weighted.csv")
    df_ms.to_csv(fname_ms, index=False)
    df_s1.to_csv(fname_s1, index=False)
    df_s2.to_csv(fname_s2, index=False)

    fname_w_avg = os.path.join(save_dir, "w_avg.csv")
    fname_avg = os.path.join(save_dir, "regr_avg.csv")
    w_avg_df.to_csv(fname_w_avg, index=False)
    avg_regr_df.to_csv(fname_avg, index=False)

if __name__ == "__main__":
    main()