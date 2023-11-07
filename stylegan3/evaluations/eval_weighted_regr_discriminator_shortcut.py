import os
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--fname_ms", type=str, default=None) ## multi-source
@click.option("--save_dir", type=str, default="strata_mae_compare.csv")


def main(
    fname_ms,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    df_ms = pd.read_csv(fname_ms)

    num_ms = df_ms["num_samples"].values
    ## mae
    real_mae_ms = df_ms["real_mae"].values
    gen_mae_ms = df_ms["gen_mae"].values
    real_disc_mae_ms = df_ms["real_disc_mae"].values
    real_predict_mae_ms = df_ms["real_predict_mae"].values
    gen_disc_mae_ms = df_ms["gen_disc_mae"].values
    gen_predict_mae_ms = df_ms["gen_predict_mae"].values

    ## mse
    real_mse_ms = df_ms["real_mse"].values
    gen_mse_ms = df_ms["gen_mse"].values
    real_disc_mse_ms = df_ms["real_disc_mse"].values
    real_predict_mse_ms = df_ms["real_predict_mse"].values
    gen_disc_mse_ms = df_ms["gen_disc_mse"].values
    gen_predict_mse_ms = df_ms["gen_predict_mse"].values
    
    ## weighted mae
    real_w_mae_ms = real_mae_ms * (num_ms / np.sum(num_ms))
    gen_w_mae_ms = gen_mae_ms * (num_ms / np.sum(num_ms))
    real_disc_w_mae_ms = real_disc_mae_ms * (num_ms / np.sum(num_ms))
    real_predict_w_mae_ms = real_predict_mae_ms * (num_ms / np.sum(num_ms))
    gen_disc_w_mae_ms = gen_disc_mae_ms * (num_ms / np.sum(num_ms))
    gen_predict_w_mae_ms = gen_predict_mae_ms * (num_ms / np.sum(num_ms))
    
    ## weighted mse
    real_w_mse_ms = real_mse_ms * (num_ms / np.sum(num_ms))
    gen_w_mse_ms = gen_mse_ms * (num_ms / np.sum(num_ms))
    real_disc_w_mse_ms = real_disc_mse_ms * (num_ms / np.sum(num_ms))
    real_predict_w_mse_ms = real_predict_mse_ms * (num_ms / np.sum(num_ms))
    gen_disc_w_mse_ms = gen_disc_mse_ms * (num_ms / np.sum(num_ms))
    gen_predict_w_mse_ms = gen_predict_mse_ms * (num_ms / np.sum(num_ms))
    
    ## add weighted mae and mse to dataframes
    df_ms["real_w_mae"] = real_w_mae_ms
    df_ms["gen_w_mae"] = gen_w_mae_ms
    df_ms["real_disc_w_mae"] = real_disc_w_mae_ms
    df_ms["real_predict_w_mae"] = real_predict_w_mae_ms
    df_ms["gen_disc_w_mae"] = gen_disc_w_mae_ms
    df_ms["gen_predict_w_mae"] = gen_predict_w_mae_ms

    df_ms["real_w_mse"] = real_w_mse_ms
    df_ms["gen_w_mse"] = gen_w_mse_ms
    df_ms["real_disc_w_mse"] = real_disc_w_mse_ms
    df_ms["real_predict_w_mse"] = real_predict_w_mse_ms
    df_ms["gen_disc_w_mse"] = gen_disc_w_mse_ms
    df_ms["gen_predict_w_mse"] = gen_predict_w_mse_ms
    
    real_w_avg_mae_ms = np.sum(real_w_mae_ms).reshape(1,)
    gen_w_avg_mae_ms = np.sum(gen_w_mae_ms).reshape(1,)
    real_disc_w_avg_mae_ms = np.sum(real_disc_w_mae_ms).reshape(1,)
    real_predict_w_avg_mae_ms = np.sum(real_predict_w_mae_ms).reshape(1,)
    gen_disc_w_avg_mae_ms = np.sum(gen_disc_w_mae_ms).reshape(1,)
    gen_predict_w_avg_mae_ms = np.sum(gen_predict_w_mae_ms).reshape(1,)
    
    real_w_avg_mse_ms = np.sum(real_w_mse_ms).reshape(1,)
    gen_w_avg_mse_ms = np.sum(gen_w_mse_ms).reshape(1,)
    real_disc_w_avg_mse_ms = np.sum(real_disc_w_mse_ms).reshape(1,)
    real_predict_w_avg_mse_ms = np.sum(real_predict_w_mse_ms).reshape(1,)
    gen_disc_w_avg_mse_ms = np.sum(gen_disc_w_mse_ms).reshape(1,)
    gen_predict_w_avg_mse_ms = np.sum(gen_predict_w_mse_ms).reshape(1,)

    real_avg_mae_ms = np.mean(real_mae_ms)
    gen_avg_mae_ms = np.mean(gen_mae_ms)
    real_disc_avg_mae_ms = np.mean(real_disc_mae_ms)
    real_predict_avg_mae_ms = np.mean(real_predict_mae_ms)
    gen_disc_avg_mae_ms = np.mean(gen_disc_mae_ms)
    gen_predict_avg_mae_ms = np.mean(gen_predict_mae_ms)
    
    real_std_mae_ms = np.std(real_mae_ms)
    gen_std_mae_ms = np.std(gen_mae_ms)
    real_disc_std_mae_ms = np.std(real_disc_mae_ms)
    real_predict_std_mae_ms = np.std(real_predict_mae_ms)
    gen_disc_std_mae_ms = np.std(gen_disc_mae_ms)
    gen_predict_std_mae_ms = np.std(gen_predict_mae_ms)
    

    real_avg_mse_ms = np.mean(real_mse_ms)
    gen_avg_mse_ms = np.mean(gen_mse_ms)
    real_disc_avg_mse_ms = np.mean(real_disc_mse_ms)
    real_predict_avg_mse_ms = np.mean(real_predict_mse_ms)
    gen_disc_avg_mse_ms = np.mean(gen_disc_mse_ms)
    gen_predict_avg_mse_ms = np.mean(gen_predict_mse_ms)
    
    real_std_mse_ms = np.std(real_mse_ms)
    gen_std_mse_ms = np.std(gen_mse_ms)
    real_disc_std_mse_ms = np.std(real_disc_mse_ms)
    real_predict_std_mse_ms = np.std(real_predict_mse_ms)
    gen_disc_std_mse_ms = np.std(gen_disc_mse_ms)
    gen_predict_std_mse_ms = np.std(gen_predict_mse_ms)
    
    w_avg_fid_df = pd.DataFrame(np.array([real_w_avg_mae_ms, gen_w_avg_mae_ms,
                                        real_disc_w_avg_mae_ms, real_predict_w_avg_mae_ms,
                                        gen_disc_w_avg_mae_ms, gen_predict_w_avg_mae_ms,
                                        real_w_avg_mse_ms, gen_w_avg_mse_ms,
                                        real_disc_w_avg_mse_ms, real_predict_w_avg_mse_ms,
                                        gen_disc_w_avg_mse_ms, gen_predict_w_avg_mse_ms]).reshape(1,-1),
                                          columns=["real_ms_mae", "gen_ms_mae", "real_disc_mae", "real_predict_mae",
                                                   "gen_disc_mae", "gen_predict_mae", 
                                                   "real_ms_mse", "gen_ms_mse", "real_disc_mse", "real_predict_mse",
                                                   "gen_disc_mse", "gen_predict_mse"])
    avg_regr_df = pd.DataFrame(np.array([real_avg_mae_ms, gen_avg_mae_ms,
                                         real_disc_avg_mae_ms, real_predict_avg_mae_ms,
                                         gen_disc_avg_mae_ms, gen_predict_avg_mae_ms,
                                         real_std_mae_ms, gen_std_mae_ms,
                                         real_disc_std_mae_ms, real_predict_std_mae_ms,
                                         gen_disc_std_mae_ms, gen_predict_std_mae_ms,
                                         real_avg_mse_ms, gen_avg_mse_ms,
                                         real_disc_avg_mse_ms, real_predict_avg_mse_ms,
                                         gen_disc_avg_mse_ms, gen_predict_avg_mse_ms,
                                         real_std_mse_ms, gen_std_mse_ms,
                                         real_disc_std_mse_ms, real_predict_std_mse_ms,
                                         gen_disc_std_mse_ms, gen_predict_std_mse_ms]).reshape(1,-1),
                                         columns=["real_ms_mae_mu", "gen_ms_mae_mu", 
                                                "real_disc_mae_mu", "real_predict_mae_mu",
                                                "gen_disc_mae_mu", "gen_predict_mae_mu",
                                                "real_ms_mae_std", "gen_ms_mae_std",
                                                "real_disc_mae_std", "real_predict_mae_std",
                                                "gen_disc_mae_std", "gen_predict_mae_std",
                                                "real_ms_mse_mu", "gen_ms_mse_mu", 
                                                "real_disc_mse_mu", "real_predict_mse_mu",
                                                "gen_disc_mse_mu", "gen_predict_mse_mu",
                                                "real_ms_mse_std", "gen_ms_mse_std",
                                                "real_disc_mse_std", "real_predict_mse_std",
                                                "gen_disc_mse_std", "gen_predict_mse_std"])

    fname_ms = os.path.join(save_dir, "ms_stratified_weighted.csv")
    df_ms.to_csv(fname_ms, index=False)

    fname_w_avg = os.path.join(save_dir, "w_avg.csv")
    fname_avg = os.path.join(save_dir, "regr_avg.csv")
    w_avg_fid_df.to_csv(fname_w_avg, index=False)
    avg_regr_df.to_csv(fname_avg, index=False)

if __name__ == "__main__":
    main()