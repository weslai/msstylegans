### Multi-Source Counterfactual
### Use Image2StyleGAN++ to reconstruct the latent code of the target image
### Reference: https://github.com/zaidbhat1234/Image2StyleGAN/blob/main/Image2Style_Implementation.ipynb

import os, re
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import models
from torchvision.utils import save_image
import numpy as np
from math import log10
import matplotlib.pyplot as plt
import click
from typing import List, Tuple, Union

### Our ###
from utils import load_generator, generate_images
from training.dataset_real_ms import UKBiobankMRIDataset2D, UKBiobankRetinalDataset, AdniMRIDataset2D, KaggleEyepacsDataset

# --------------------------------------------------------------------------------------
class VGG16_perceptual(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16_perceptual, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu1_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        return h_relu1_1, h_relu1_2, h_relu3_2, h_relu4_2

# --------------------------------------------------------------------------------------
def loss_function(syn_img, img, img_p, MSE_loss, upsample, perceptual):
    #UpSample synthesized image to match the input size of VGG-16 input. 
    #Extract mid level features for real and synthesized image and find the MSE loss between them for perceptual loss. 
    #Find MSE loss between the real and synthesized images of actual size
    if syn_img.shape[1] == 1:
        syn_img = syn_img.repeat(1,3,1,1)
    if img_p.shape[1] == 1:
        img_p = img_p.repeat(1,3,1,1)
    syn_img_p = upsample(syn_img)
    syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
    r0, r1, r2, r3 = perceptual(img_p)
    mse = MSE_loss(syn_img,img)

    per_loss = 0
    per_loss += MSE_loss(syn0,r0)
    per_loss += MSE_loss(syn1,r1)
    per_loss += MSE_loss(syn2,r2)
    per_loss += MSE_loss(syn3,r3)

    return mse, per_loss
# --------------------------------------------------------------------------------------
def PSNR(mse, flag = 0):
    #flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
    if flag == 0:
        psnr = 10 * log10(1 / mse.item())
    return psnr
     
# --------------------------------------------------------------------------------------
def embedding_function(image, Gen, device):
    upsample = torch.nn.Upsample(scale_factor = 256/256, mode = 'bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)
    #Perceptual loss initialise object
    perceptual = VGG16_perceptual().to(device)

    #MSE loss object
    MSE_loss = nn.MSELoss(reduction="mean")
    #since the synthesis network expects 16 w vectors of size 1x512 thus we take latent vector of the same size
    latents = torch.zeros((1, 16, 512), requires_grad = True, device = device)
    #Optimizer to change latent code in each backward step
    optimizer = optim.Adam({latents},lr=0.01,betas=(0.9,0.999),eps=1e-8)

    #Loop to optimise latent vector to match the generated image to input image
    loss_ = []
    loss_psnr = []
    for e in range(1500):
        optimizer.zero_grad()
        syn_img = Gen.synthesis(latents)
        syn_img = (syn_img+1.0)/2.0
        mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
        psnr = PSNR(mse, flag = 0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()
        loss_np=loss.detach().cpu().numpy()
        loss_p=per_loss.detach().cpu().numpy()
        loss_m=mse.detach().cpu().numpy()
        loss_psnr.append(psnr)
        loss_.append(loss_np)
        if (e+1)%500==0 :
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e+1,loss_np,loss_m,loss_p,psnr))
            save_image(syn_img.clamp(0,1),"test_{}.png".format(e+1))
            #np.save("loss_list.npy",loss_)
            #np.save("latent_W.npy".format(),latents.detach().cpu().numpy())

    print("loss = mse + perceptua", loss_)
    print("PSNR", loss_psnr)
    # plt.plot(loss_, label = 'Loss = MSELoss + Perceptual')
    # plt.plot(loss_psnr, label = 'PSNR')
    # plt.legend()
    return latents
# --------------------------------------------------------------------------------------
def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges
# --------------------------------------------------------------------------------------
def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------
def get_covs(dataset):
    if dataset == "retinal":
        COVS = {"c1": "age", "c2": "diastolic bp", "c3": "spherical power left"}
    elif dataset == "ukb":
        pass
    elif dataset == "mnist-thickness-intensity-slant":
        COVS = {"c1": "thickness", "c2": "intensity", "c3": "slant"}
    return COVS
#----------------------------------------------------------------------------

# @click.command()
# @click.option('--network_pkl', 'network_pkl', help='Network pickle filename', default=None)
# @click.option('--network', 'metric_jsonl', help='Metric jsonl file for one training', default=None)
# @click.option('--group-by', 'group_by', type=str, default="c1", show_default=True)
# @click.option('--dataset', 'dataset', type=click.Choice(['mnist-thickness-intensity-slant', 'ukb', 
#                                                          'retinal', None]), default=None, show_default=True)
# @click.option('--data-path1', 'data_path1', type=str, help='Path to the data source 1', required=True)
# @click.option('--data-path2', 'data_path2', type=str, help='Path to the data source 2', required=True)
# @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.8, show_default=True)
# @click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), 
#               default='const', show_default=True)
# @click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, 
#               default='0,0', show_default=True, metavar='VEC2')
# @click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, 
#               show_default=True, metavar='ANGLE')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# def run_visualizer_two_covs(
#     network_pkl: str,
#     metric_jsonl: str,
#     group_by: str,
#     dataset: str,
#     data_path1: str,
#     data_path2: str,
#     seeds: List[int],
#     truncation_psi: float,
#     noise_mode: str,
#     translate: Tuple[float,float],
#     rotate: float,
#     outdir: str
# ):
def run_visualizer_two_covs(
    network_pkl: str,
    metric_jsonl: str,
    dataset: str,
    data_path1: str,
    data_path2: str,
):
    device = torch.device('cuda')
    # Load the network.
    Gen = load_generator(
        network_pkl=network_pkl,
        metric_jsonl=metric_jsonl,
        use_cuda=True
    )
    Gen = Gen.eval()

    # os.makedirs(outdir, exist_ok=True)
    ## load the testset
    ## import sampler
    if dataset == "mri":
        ds1 = UKBiobankMRIDataset2D(data_name="ukb", ## UKB c = (age, greymatter, ventricle)
                                    path=data_path1, 
                                    mode="test", 
                                    use_labels=True,
                                    xflip=False)
        ds1 = iter(DataLoader(ds1, batch_size=1, shuffle=False, num_workers=0))
        # sampler_ukb = SourceSampling(dataset="ukb",
        #                              label_path=os.path.join(data_path1, "train"))
        ds2 = AdniMRIDataset2D(data_name="adni",  ## ADNI c = (age, cdr)
                                path=data_path2, 
                                mode="test", 
                                use_labels=True,
                                xflip=False)
        ds2 = DataLoader(ds2, batch_size=1, shuffle=False, num_workers=0)
        # sampler_adni = SourceSampling(dataset="adni",
        #                               label_path=os.path.join(data_path2, "train"))
    elif dataset == "retinal":
        ds1 = UKBiobankRetinalDataset(data_name=dataset,
                                      path=data_path1,
                                      mode="test",
                                      use_labels=True,
                                      xflip=False)
        ds1 = DataLoader(ds1, batch_size=1, shuffle=False, num_workers=0)
        # sampler_retinal = SourceSampling(dataset="eyepacs",
        #                             label_path=os.path.join(data_path2, "train"))
        ds2 = KaggleEyepacsDataset(data_name=dataset,
                                    path=data_path2,
                                    mode="test",
                                    use_labels=True,
                                    xflip=False)
        ds2 = DataLoader(ds2, batch_size=1, shuffle=False, num_workers=0)
        # sampler_eyepacs = SourceSampling(dataset="eyepacs",
        #                               label_path=os.path.join(data_path2, "train"))
        
    img = next(ds1)[0]
    img = img.to(device).to(torch.float32) / 127.5 - 1 ## (-1, 1)
    latent_ws = embedding_function(img, Gen, device)

    ## back prop to (z, c)
    latent_ws 