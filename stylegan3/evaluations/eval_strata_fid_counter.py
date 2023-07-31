### Third-party ###
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--dataset", type=str, default="retinal")
@click.option("--ms_path", type=str, default=None)
@click.option("--source1_path", type=str, default=None)
@click.option("--source2_path", type=str, default=None)
@click.option("--save_path", type=str, default="mean_fid")

def main(
    dataset,
    ms_path, 
    source1_path,
    source2_path,
    save_path = "strata_fid_compare.csv"
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
    scores_ms = 0
    scores_s1 = 0
    scores_s2 = 0
    ## mean scores
    mean_ms = ms_df["fid_score"].mean()
    mean_s1 = source1_df["fid_score"].mean()
    mean_s2 = source2_df["fid_score"].mean()

    assert len(ms_df["fid_score"]) == len(source1_df["fid_score"]) == len(source2_df["fid_score"])
    total_len = len(ms_df["fid_score"])
    for s0, s1, s2 in zip(ms_df["fid_score"], source1_df["fid_score"], source2_df["fid_score"]):
        scores = np.array([s0, s1, s2])
        min_idx = np.argmin(scores)
        if min_idx == 0:
            scores_ms += 1
        elif min_idx == 1:
            scores_s1 += 1
        elif min_idx == 2:
            scores_s2 += 1
    print("scores_ms: {:d}, scores_s1: {:d}, scores_s2: {:d}".format(scores_ms, scores_s1, scores_s2))

    ## dataframe
    arr = np.array([scores_ms, scores_s1, scores_s2, mean_ms, mean_s1, mean_s2, total_len, dataset]).reshape(1, -1)
    compare_df = pd.DataFrame(arr, columns=["ms", "souce1", "source2", "mean_ms", "mean_s1", "mean_s2", "length", "dataset"])

    return compare_df

if __name__ == "__main__":
    main()
