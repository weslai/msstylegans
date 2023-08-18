import matplotlib.pyplot as plt
from matplotlib import gridspec
import PIL.Image
import seaborn as sns
import numpy as np
import pandas as pd
import torch

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
    c2_name = None,
    c3_name = None,
    save_path: str = None
):
    images = images.cpu().detach().numpy()
    nrows = len(c2)
    ncols = len(c3)
    # ncols = np.sqrt(images.shape[0]).astype(int)
    # nrows = ncols
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    fig = plt.figure(figsize=(ncols*1.2, nrows*1.2))
    gs = gridspec.GridSpec(nrows, ncols,
        wspace=0.0, hspace=0.0
    )
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(gs[i, j])
            if dataset_name == "retinal":
                img = images[i * ncols + j] ### (M, M, 3)
                ax.imshow(img, vmin=0, vmax=255)
            else:
                img = images[i * ncols + j][:, :, 0]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if j == 0:
                ax.set_ylabel("{:.1f}".format(c2[i][0]), fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("{:.1f}".format(c3[j][0]), fontsize=10)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Generated Images", fontsize=14)
    fig.supxlabel(f"{c3_name}", fontsize=14)
    fig.supylabel(f"{c2_name}", fontsize=14)
    if save_path is not None:
        plt.savefig(save_path)
        save_path = save_path.replace(".png", ".pdf")
        plt.savefig(save_path)
    plt.close()
def plot_two_covs_images_dualsources(
    images,
    c2,
    c3,
    dataset_name: str,
    c2_name = None,
    c3_name = None,
    save_path: str = None
):
    images = images.cpu().detach().numpy()
    nrows = len(c2)
    ncols = len(c3)
    # ncols = np.sqrt(images.shape[0]).astype(int)
    # nrows = ncols
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    fig = plt.figure(figsize=(ncols*1.5, nrows*1.2))
    gs = gridspec.GridSpec(nrows, ncols,
        wspace=0.0, hspace=0.0
    )
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(gs[i, j])
            if dataset_name == "retinal":
                img = images[i * ncols + j] ### (M, M, 3)
                img = np.array(PIL.Image.fromarray(img).resize((192, 128))).astype(np.uint8)
                # img = np.concatenate(img[:, :, 0].resize((128,192)).reshape(128, 192, 1),
                #                      img[:, :, 1].resize((128,192)).reshape(128, 192, 1),
                #                      img[:, :, 2].resize((128,192)).reshape(128, 192, 1),
                #                     axis=-1)
                ax.imshow(img, vmin=0, vmax=255)
            else:
                img = images[i * ncols + j][:, :, 0]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if j == 0:
                ax.set_ylabel("{:.1f}".format(c2[i][0]), fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("{:.1f}".format(c3[j][0]), fontsize=10)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Generated Images", fontsize=14)
    fig.supxlabel(f"{c3_name}", fontsize=14)
    fig.supylabel(f"{c2_name}", fontsize=14)
    if save_path is not None:
        plt.savefig(save_path)
        save_path = save_path.replace(".png", ".pdf")
        plt.savefig(save_path)
    plt.close()

def plot_covs_images_threesources(
    images: dict,
    c2,
    c3,
    group_cov, 
    dataset_name: str,
    c2_name = None,
    c3_name = None,
    group_name = None,
    save_path: str = None
):
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    # fig = plt.figure(figsize=(ncols*1.2, nrows*1.2))
    if len(group_cov) == 4:
        fig = plt.figure(figsize=(9, 12))
    else:
        fig = plt.figure(figsize=(18, 8))
    nrows = len(c2)
    ncols = len(c3)
    for k, (key, imgs) in enumerate(images.items()):
        imgs = imgs.cpu().detach().numpy()
        gs = gridspec.GridSpec(nrows, ncols,
            wspace=0.0, hspace=0.0
        )
        if len(group_cov) == 4:
            if key == 0:
                gs.update(left=0.05, right=0.50, bottom=0.50, top=0.90)
            elif key == 1:
                gs.update(left=0.52, right=0.98, bottom=0.50, top=0.90)
            elif key == 2:
                gs.update(left=0.05, right=0.50, bottom=0.05, top=0.45)
            elif key == 3:
                gs.update(left=0.52, right=0.98, bottom=0.05, top=0.45)
        elif len(group_cov) == 3:
            if key == 0:
                gs.update(left=0.07, right=0.37, bottom=0.15, top=0.9)
            elif key == 1:
                gs.update(left=0.38, right=0.68, bottom=0.15, top=0.9)
            elif key == 2:
                gs.update(left=0.69, right=0.99, bottom=0.15, top=0.9)
        for i in range(nrows):
            for j in range(ncols):
                ax = plt.subplot(gs[i, j])
                if dataset_name == "retinal":
                    img = imgs[i * ncols + j] ### (M, M, 3)
                    ax.imshow(img, vmin=0, vmax=255)
                else:
                    img = imgs[i * ncols + j][:, :, 0]
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                if j == 0 and key == 0:
                    ax.set_ylabel("{}".format(round(c2[i][0])), fontsize=10)
                if i == nrows - 1:
                    ax.set_xlabel("{:.1f}".format(c3[j][0]), fontsize=10)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        ax_sub = fig.add_subplot(gs[:])
        ax_sub.axis("off")
        ax_sub.set_title(f"c4: {group_name} = {group_cov[k][0]:.1f}", fontsize=14)
        fig.suptitle("Generated Images", fontsize=18)
        fig.supxlabel(f"{c3_name}", fontsize=16)
        fig.supylabel(f"{c2_name}", fontsize=16)
    if save_path is not None:
        plt.savefig(save_path)
        save_path = save_path.replace(".png", ".pdf")
        plt.savefig(save_path)
    plt.close()

def plot_negpos_images(
    real_images: dict,
    gen_images: dict,
    labels: dict,
    dataset_name: str,
    c2_name = None,
    c3_name = None,
    save_path: str = None,
    single_source: bool = True
):
    if single_source:
        if dataset_name == "mnist-thickness-intensity-slant":
            c2_name = "thickness"
            c3_name = "intensity" if dataset_name.split("_")[-1] == "source1" else "slant"
        elif dataset_name.split("_")[0] == "ukb":
            c2_name = "age"
            c3_name = "brain" if dataset_name.split("_")[-1] == "source1" else "ventricles"
        elif dataset_name.split("_")[0] == "retinal":
            c2_name = "age"
            c3_name = "diastolic bp" if dataset_name.split("_")[-1] == "source1" else "spherical power"
            dataset_name = "retinal"

    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    fig = plt.figure(figsize=(16, 10))
    for num_sub in range(4):
        reimgs = real_images[str(num_sub)].cpu().detach().numpy()
        genimgs = gen_images[str(num_sub)].cpu().detach().numpy()
        real_labels = labels[str(num_sub)]
        nrows = 1
        ncols = int(gen_images[str(num_sub)].shape[0] // nrows)
        gs = gridspec.GridSpec(nrows, ncols,
            wspace=0.0, hspace=0.0
        )
        gs1 = gridspec.GridSpec(nrows, ncols,
            wspace=0.0, hspace=0.0
        )
        if num_sub == 0:
            gs.update(left=0.06, right=0.51, bottom=0.50, top=0.75)
            gs1.update(left=0.06, right=0.51, bottom=0.75, top=0.95)
        elif num_sub == 1:
            gs.update(left=0.52, right=0.97, bottom=0.50, top=0.75)
            gs1.update(left=0.52, right=0.97, bottom=0.75, top=0.95)
        elif num_sub == 2:
            gs.update(left=0.06, right=0.51, bottom=0.03, top=0.25)
            gs1.update(left=0.06, right=0.51, bottom=0.25, top=0.50)
        elif num_sub == 3:
            gs.update(left=0.52, right=0.97, bottom=0.03, top=0.25)
            gs1.update(left=0.52, right=0.97, bottom=0.25, top=0.50)
        
        for i in range(nrows):
            for j in range(ncols): 
                ax = plt.subplot(gs[i, j]) ### real
                ax1 = plt.subplot(gs1[i, j]) ### generated
                if dataset_name == "retinal":
                    img = reimgs[i * ncols + j] ### (M, M, 3)
                    ax.imshow(img, vmin=0, vmax=255)
                    img = genimgs[i * ncols + j] ### (M, M, 3)
                    ax1.imshow(img, vmin=0, vmax=255)
                else:
                    img = reimgs[i * ncols + j][:, :, 0]
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                    img = genimgs[i * ncols + j][:, :, 0]
                    ax1.imshow(img, cmap="gray", vmin=0, vmax=255)
                if j == 0 and num_sub in [0, 2]:
                    ax.set_ylabel("Real", fontsize=12)
                    ax1.set_ylabel("Generated", fontsize=12)
                # if i == nrows - 1:
                #     ax.set_xlabel("{:.2f}".format(real_labels[i * ncols + j][1]), fontsize=10)
                # if i == nrows - 1:
                #     ax.set_xlabel("{:.2f}".format(c3[j][0]), fontsize=8)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax1.set_xticks([])
                ax1.set_yticks([])
        ax_sub = fig.add_subplot(gs[:])
        ax_sub.axis("off")
        ax_sub1 = fig.add_subplot(gs1[:])
        ax_sub1.axis("off")
        if num_sub == 0:
            ax_sub.set_title(f"The smallest {c2_name} and {c3_name}", fontsize=14)
        elif num_sub == 1:
            ax_sub.set_title(f"The smallest {c2_name} and the largest {c3_name}", fontsize=14)
        elif num_sub == 2:
            ax_sub.set_title(f"The largest {c2_name} and the smallest {c3_name}", fontsize=14)
        elif num_sub == 3:
            ax_sub.set_title(f"The largest {c2_name} and {c3_name}", fontsize=14)
    fig.suptitle(f"The comparison between real and generated images by controlling {c2_name}, {c3_name}", fontsize=20)
    fig.supxlabel(f"{c3_name}", fontsize=20)
    fig.supylabel(f"{c2_name}", fontsize=20)
    if save_path is not None:
        plt.savefig(save_path)
        save_path = save_path.replace(".png", ".pdf")
        plt.savefig(save_path)
    plt.close()