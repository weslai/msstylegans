### Third-party ###
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--dataset", type=str, default="retinal")
@click.option("--ms_path", type=str, default=None)
@click.option("--source1_path", type=str, default=None)
@click.option("--source2_path", type=str, default=None)
@click.option("--save_path", type=str, default="strata_mae_compare.csv")

def main(
    dataset,
    ms_path, 
    source1_path,
    source2_path,
    save_path = "strata_mae_compare.csv"
):
    compare_df = load_fids(dataset, ms_path, source1_path, source2_path)
    compare_df.to_csv(save_path, index=False)

def load_fids(
    dataset,
    ms_path, 
    source1_path,
    source2_path
):
    ms_df = pd.read_csv(ms_path)
    source1_df = pd.read_csv(source1_path)
    source2_df = pd.read_csv(source2_path)

    ## scores
    scores_mae_ms = 0
    scores_mae_s1 = 0
    scores_mae_s2 = 0
    scores_mse_ms = 0
    scores_mse_s1 = 0
    scores_mse_s2 = 0
    ## mean scores
    mean_mae_ms = ms_df["gen_mae"].mean()
    mean_mae_s1 = source1_df["gen_mae"].mean()
    mean_mae_s2 = source2_df["gen_mae"].mean()

    mean_mse_ms = ms_df["gen_mse"].mean()
    mean_mse_s1 = source1_df["gen_mse"].mean()
    mean_mse_s2 = source2_df["gen_mse"].mean()

    assert len(ms_df["gen_mae"]) == len(source1_df["gen_mae"]) == len(source2_df["gen_mae"])
    assert len(ms_df["gen_mse"]) == len(source1_df["gen_mse"]) == len(source2_df["gen_mse"])
    total_len = len(ms_df["gen_mae"])
    for s0, s1, s2 in zip(ms_df["gen_mae"], source1_df["gen_mae"], source2_df["gen_mae"]):
        scores = np.array([s0, s1, s2])
        min_idx = np.argmin(scores)
        if min_idx == 0:
            scores_mae_ms += 1
        elif min_idx == 1:
            scores_mae_s1 += 1
        elif min_idx == 2:
            scores_mae_s2 += 1
    print("scores_ms: {:d}, scores_s1: {:d}, scores_s2: {:d}".format(scores_mae_ms, scores_mae_s1, scores_mae_s2))

    total_len = len(ms_df["gen_mse"])
    for s0, s1, s2 in zip(ms_df["gen_mse"], source1_df["gen_mse"], source2_df["gen_mse"]):
        scores = np.array([s0, s1, s2])
        min_idx = np.argmin(scores)
        if min_idx == 0:
            scores_mse_ms += 1
        elif min_idx == 1:
            scores_mse_s1 += 1
        elif min_idx == 2:
            scores_mse_s2 += 1
    print("scores_ms: {:d}, scores_s1: {:d}, scores_s2: {:d}".format(scores_mse_ms, scores_mse_s1, scores_mse_s2))

    ## dataframe
    arr = np.array([scores_mae_ms, scores_mae_s1, scores_mae_s2, mean_mae_ms, mean_mae_s1, mean_mae_s2, 
                    scores_mse_ms, scores_mse_s1, scores_mse_s2, mean_mse_ms, mean_mse_s1, mean_mse_s2,
                    total_len, dataset]).reshape(1, -1)
    compare_df = pd.DataFrame(arr, columns=["ms_mae", "souce1_mae", "source2_mae", "mean_mae_ms", "mean_mae_s1", "mean_mae_s2", 
                                            "ms_mse", "source1_mse", "source2_mse", "mean_mse_ms", "mean_mse_s1", "mean_mse_s2",
                                            "length", "dataset"])

    return compare_df

if __name__ == "__main__":
    main()
