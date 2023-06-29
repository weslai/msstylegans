import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import torch


### Heatmap calculation
def heatmap_covs(
):
    pass


### Condition on two covariates and generate images
def generate_labels_two_covs(
    dataset_name: str,
    sampler1,
    sampler2,
    dataset1,
    dataset2,
    df1_desc: pd.DataFrame,
):
    if dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]:
        c1_mu = df1_desc.loc["mean", "thickness"]
        c1_25 = df1_desc.loc["25%", "thickness"]
        c1_75 = df1_desc.loc["75%", "thickness"]
        c1_std = df1_desc.loc["std", "thickness"]
        c1_min = df1_desc.loc["min", "thickness"]
        c1_max = df1_desc.loc["max", "thickness"]
        thickness_mu = dataset1.model["thickness_mu"]
        thickness_std = dataset1.model["thickness_std"]

        c1_norm = np.linspace(c1_25, c1_75, num=7).reshape(-1, 1)
        c1 = c1_norm * thickness_std + thickness_mu
        _, c2 = sampler1.sampling_intensity(c1, normalize=True, model_=dataset1.model)
        _, c3 = sampler2.sampling_slant(c1, normalize=True, model_=dataset2.model)
        c = torch.cat([torch.tensor(c1_norm), c2, c3], dim=1)
        return c

    elif dataset_name == "ukb":
        pass
    else:
        raise NotImplementedError(f"dataset_name {dataset_name} not implemented")
    
def plot_two_covs_images(
    images,
    c2, 
    c3,
    dataset_name: str,
    save_path: str = None,
    single_source: bool = True
):
    if single_source:
        if dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]:
            c2_name = "thickness"
            c3_name = "intensity" if dataset_name == "mnist-thickness-intensity" else "slant"
        elif dataset_name.split("_")[0] == "ukb":
            c2_name = "age"
            c3_name = "brain" if dataset_name.split("_")[-1] == "source1" else "ventricles"
        elif dataset_name.split("_")[0] == "retinal":
            c2_name = "age"
            c3_name = "systolic bp" if dataset_name.split("_")[-1] == "source1" else "cylindrical power"
            dataset_name = "retinal"
    else:
        if dataset_name in ["mnist-thickness-intensity", "mnist-thickness-slant"]:
            c2_name, c3_name = "intensity", "slant"
        elif dataset_name == "ukb":
            c2_name, c3_name = "brain", "ventricles"
        elif dataset_name == "retinal":
            c2_name, c3_name = "systolic bp", "cylindrical power"
    images = images.cpu().detach().numpy()
    ncols = np.sqrt(images.shape[0]).astype(int)
    nrows = ncols
    fig = plt.figure(figsize=(ncols*3, nrows*2))
    gs = gridspec.GridSpec(nrows, ncols,
        wspace=0.0, hspace=0.0
    )
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(gs[i, j])
            # ax.imshow(img, cmap="gray")
            if dataset_name == "retinal":
                img = images[i * ncols + j] ### (M, M, 3)
                ax.imshow(img, vmin=0, vmax=255)
            else:
                img = images[i * ncols + j][:, :, 0]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if j == 0:
                ax.set_ylabel("{:.2f}".format(c2[i][0]), fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel("{:.2f}".format(c3[j][0]), fontsize=8)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    fig.suptitle("Generated Images", fontsize=14)
    fig.supxlabel(f"c3: {c3_name}", fontsize=14)
    fig.supylabel(f"c2: {c2_name}", fontsize=14)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


