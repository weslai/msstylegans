# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop_real_ms
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

### ------------ ###
### --- Own ---- ###
### ------------ ###
from evaluations.eval_utils import get_k_lowest_checkpoints
#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop_real_ms.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    print(f'Dataset path:        {c.validation_set_kwargs.path}')
    print(f'Dataset size:        {c.validation_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.validation_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.validation_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.validation_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, name, use_labels, mode: str,
                        max_size: tuple = None, scenario: str = None):
    try:
        ## UKB, ADNI, Retinal, EyePACS
        if name == "ukb":
            dataset_kwargs = dnnlib.EasyDict(
                class_name="training.dataset_real_ms.UKBiobankMRIDataset2D",
                data_name=name,
                path=data,
                mode=mode,
                use_labels=use_labels,
                max_size=max_size,
                xflip=False
            )
        elif name == "retinal":
            dataset_kwargs = dnnlib.EasyDict(
                class_name="training.dataset_real_ms.UKBiobankRetinalDataset",
                data_name=name,
                path=data,
                mode=mode,
                use_labels=use_labels,
                max_size=max_size,
                xflip=False
            )
        elif name == "eyepacs":
            dataset_kwargs = dnnlib.EasyDict(
                class_name="training.dataset_real_ms.KaggleEyepacsDataset",
                data_name=name,
                path=data,
                mode=mode,
                use_labels=use_labels,
                max_size=max_size,
                xflip=False
            )
        elif name == "adni":
            dataset_kwargs = dnnlib.EasyDict(
                class_name = "training.dataset_real_ms.AdniMRIDataset2D",
                data_name=name,
                path=data,
                mode=mode,
                use_labels=use_labels,
                max_size=max_size,
                xflip=False
            )
        elif name == "rfmid":
            dataset_kwargs = dnnlib.EasyDict(
                class_name="training.dataset_real_ms.RFMiDDataset",
                data_name=name,
                path=data,
                mode=mode,
                use_labels=use_labels,
                max_size=max_size,
                xflip=False
            )
        else:
            raise click.ClickException(f'--data_name: {name} dataloader not implemented')
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        if mode == "train":
            if scenario == "half":
                dataset_kwargs.max_size = int(len(dataset_obj) / 2)
            else:
                dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        else:
            dataset_kwargs.max_size = len(dataset_obj)
        return dataset_kwargs, dataset_obj.data_name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data1',         help='Training data 1', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--data_name1',    help='Dataset name 1', metavar='STR',                             type=str, required=True)
@click.option('--data2',         help='Training data 2', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--data_name2',    help='Dataset name 2', metavar='STR',                             type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)
@click.option('--validation_metrics', help='validation metrics', metavar='STR',                 type=parse_comma_separated_list, default=['fid50k_full'], required=False)
@click.option('--wandb_name',   help='wandb experiment name', metavar='STR',                    type=str,  default=None, required=False)
@click.option('--wandb_pj_v',   help='wandb project version', metavar='STR',                    type=str,  default=None, required=False)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--exact_resume',help='Resume from given exact network pickle',                   type=bool, default=False, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--resume_kimg',  help='Resume from given kilo-images', metavar='INT',           type=int, default=0, show_default=True)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--data_scenario', help='data scenario', metavar='STR',                           type=click.Choice(["high", "half"]), default='high', show_default=True)
@click.option('--lambda_',      help='lambda for the loss', metavar='FLOAT',                    type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--lambda_step',  help='number of steps to update lambda', metavar='FLOAT',       type=click.IntRange(min=1), default=1000, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
# @click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=16384, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default=['fid50k_ms_real'], show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_real_ms.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    if opts.data_scenario == "high":
        max_size = None
        max_size1 = None
    elif opts.data_scenario == "half":
        max_size = int(10000/2)
        max_size1 = int(10000/2)
    ## first dataset
    c.training_set_kwargs,dataset_name1 = init_dataset_kwargs(data=opts.data1, name=opts.data_name1, 
                                                              use_labels=opts.cond, mode="train", 
                                                              max_size=max_size, scenario=opts.data_scenario)
    c.validation_set_kwargs, val_dataset_name1 = init_dataset_kwargs(data=opts.data1, name=opts.data_name1, 
                                                                     use_labels=opts.cond, mode="val", 
                                                                     max_size=None)
    ## second dataset
    c.training_set_kwargs1, dataset_name2 = init_dataset_kwargs(data=opts.data2, name=opts.data_name2, 
                                                                use_labels=opts.cond, mode="train", 
                                                                max_size=max_size1, scenario=opts.data_scenario)
    c.validation_set_kwargs1, val_dataset_name2 = init_dataset_kwargs(data=opts.data2, name=opts.data_name2, 
                                                                      use_labels=opts.cond, mode="val", 
                                                                      max_size=None)
    assert val_dataset_name1 == dataset_name1
    assert val_dataset_name2 == dataset_name2

    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException(f'--cond=True requires labels specified in dataset.json for {opts.data_name}')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.validation_set_kwargs.use_labels = opts.cond
    c.validation_set_kwargs.xflip = opts.mirror

    c.training_set_kwargs1.use_labels = opts.cond
    c.training_set_kwargs1.xflip = opts.mirror
    c.validation_set_kwargs1.use_labels = opts.cond
    c.validation_set_kwargs1.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_lambda = opts.lambda_
    c.lambda_step = opts.lambda_step
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.validation_metrics = opts.validation_metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    
    ## wandb experiment name
    c.wandb_pj_v = opts.wandb_pj_v
    c.wandb_name = opts.wandb_name

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        if opts.exact_resume:
            c.resume_pkl = opts.resume
            ckpt = int(opts.resume.split('/')[-1].split('.')[0].split('-')[-1])
        else:    
            k_path, fid_score, kidx = get_k_lowest_checkpoints(metric_jsonl=opts.resume, k=1)
            print(f"Resume from {k_path[0]} with fid score {fid_score[0]}")
            result_path = "/" + os.path.join(*opts.resume.split('/')[:-1])
            c.resume_pkl = os.path.join(result_path, k_path[0])
            ckpt = int(k_path[0].split('.')[0].split('-')[-1])
        c.resume_kimg = ckpt
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-cond{opts.cond}-multisource-{dataset_name1:s}-aug{opts.aug:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}-kimg{c.total_kimg:d}'
    
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    if torch.cuda.is_available():
        print("it starts running")
        # Launch.
        launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)
    else:
        RuntimeError("not using Cuda!")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
    
#----------------------------------------------------------------------------
