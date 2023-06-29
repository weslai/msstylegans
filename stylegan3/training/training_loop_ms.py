# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os, sys
sys.path.append("/dhc/home/wei-cheng.lai/projects/causal_gans/stylegan3/")
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
import wandb

### --- Own ---
from latent_mle_ms import CausalSampling
from latent_dist_morphomnist import MorphoSampler
from training.dataset import ConcatDataset
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = list(np.random.randint(0, len(label_order), size=gh * gw))
        # grid_indices = []
        # for y in range(gh):
        #     label = label_order[y % len(label_order)]
        #     indices = label_groups[label]
        #     grid_indices += [indices[x % len(indices)] for x in range(gw)]
        #     label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    training_set_kwargs1    = {},       # Options for training set.
    validation_set_kwargs   = {},       # Options for validation set.
    validation_set_kwargs1  = {},       # Options for validation set.
    use_ground_truth        = False,    # Whether to use ground truth labels or estimation.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    validation_metrics      = [],       # Metrics to evaluate after training.
    random_seed             = 0,        # Global random seed.
    wandb_name              = "default",# Name of the run in wandb
    wandb_pj_v              = "default",# Version of the project in wandb
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 200,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 200,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    ## start logging
    wandb.init(
        project=wandb_pj_v,
        name=wandb_name,
        dir="/dhc/home/wei-cheng.lai/experiments/wandb/",
        config=locals()
    )
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set1 = dnnlib.util.construct_class_by_name(**training_set_kwargs1) # subclass of training.dataset.Dataset
    ### activate datasets
    training_set.get_label(0)
    training_set1.get_label(0)
    concat_training_set = ConcatDataset(training_set, training_set1)
    training_set_sampler = misc.InfiniteSampler(dataset=concat_training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=concat_training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    val_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs) # subclass of training.dataset.Dataset
    val_set1 = dnnlib.util.construct_class_by_name(**validation_set_kwargs1) # subclass of training.dataset.Dataset
    concat_val_set = ConcatDataset(val_set, val_set1) # subclass of training.dataset.Dataset
    val_set_sampler = misc.InfiniteSampler(dataset=concat_val_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    val_set_iterator = iter(torch.utils.data.DataLoader(dataset=concat_val_set, sampler=val_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    

    # Load Latent Space sampler
    if training_set.data_name == "mnist-thickness-intensity" or training_set.data_name == "mnist-thickness-slant" or training_set.data_name == "mnist-thickness-intensity-slant":
        latent_sampler1 = MorphoSampler(dataset_name=training_set.data_name,
                                        label_path=training_set._path,
                                        use_groud_truth=use_ground_truth,
                                        which_source="source1")
        latent_sampler2 = MorphoSampler(dataset_name=training_set1.data_name,
                                        label_path=training_set1._path,
                                        use_groud_truth=use_ground_truth,
                                        which_source="source2")
    else:
        latent_sampler1 = CausalSampling(dataset=training_set.data_name,
                                         label_path=training_set._path)
        df1 = latent_sampler1.get_graph()
        latent_model1 = latent_sampler1.get_causal_model()
        latent_sampler2 = CausalSampling(dataset=training_set1.data_name,
                                        label_path=training_set1._path)
        df2 = latent_sampler2.get_graph()
        latent_model2 = latent_sampler2.get_causal_model()

    if rank == 0:
        print()
        print("Dataset name 1: ", training_set.data_name)
        print('Num images train 1: ', len(training_set))
        print("Num images val 1: ", len(val_set))
        print('Image shape 1: ', training_set.image_shape)
        print('Label shape 1: ', training_set.label_shape)
        print()
        print("Dataset name 2: ", training_set1.data_name)
        print('Num images train 2: ', len(training_set1))
        print("Num images val 2: ", len(val_set1))
        print('Image shape 2: ', training_set1.image_shape)
        print('Label shape 2: ', training_set1.label_shape)
        print()
        wandb.config.update(
            {
                "dataset1": training_set.data_name,
                "num_images1": len(training_set),
                "image_shape1": training_set.image_shape,
                "label_shape1": training_set.label_shape,
                "num_images_val1": len(val_set),

                "dataset2": training_set1.data_name,
                "num_images2": len(training_set1),
                "image_shape2": training_set1.image_shape,
                "label_shape2": training_set1.label_shape,
                "num_images_val2": len(val_set1),
            }
        )

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    g_label_dim = training_set.label_dim + 1
    common_kwargs = dict(c_dim=g_label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        
        ## labels estimation for source 1
        if training_set.data_name == "mnist-thickness-intensity" or training_set.data_name == "mnist-thickness-slant" or training_set.data_name == "mnist-thickness-intensity-slant":
            thickness = labels[:, 0] * training_set1.model["thickness_std"] + training_set1.model["thickness_mu"]
            _, slant = latent_sampler2.sampling_slant(thickness, normalize=True, model_=training_set1.model)
            labels = np.concatenate([labels, slant], axis=1)
        elif training_set.data_name == "ukb" or training_set.data_name == "retinal":
            age = labels[:, 0] * (training_set1.model["age_max"] - training_set1.model["age_min"])  + training_set1.model["age_min"]
            labels_w_ventr = latent_sampler2.sampling_given_age(age, normalize=True).cpu().detach().numpy()
            labels = np.concatenate([labels, labels_w_ventr[:, -1].reshape(-1, 1)], axis=1)
        else:
            raise NotImplementedError("Unknown dataset name: ", training_set.data_name)
        
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        # try:
        #     import torch.utils.tensorboard as tensorboard
        #     stats_tfevents = tensorboard.SummaryWriter(run_dir)
        # except ImportError as err:
        #     print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            ### Real data
            source1, source2 = next(training_set_iterator)
            phase_real_img1, phase_real_c1 = source1[0], source1[1] ## (c1), c2
            phase_real_img2, phase_real_c2 = source2[0], source2[1] ## (c1), c3
            ## source 1
            phase_real_img1 = (phase_real_img1.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            ## source 2
            phase_real_img2 = (phase_real_img2.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

            ## estimate the latent space (hidden variables)
            if training_set.data_name == "mnist-thickness-intensity" or training_set.data_name == "mnist-thickness-slant" or training_set.data_name == "mnist-thickness-intensity-slant":
                ## estimate slants
                thickness = phase_real_c1[:, 0] * torch.tensor(training_set1.model["thickness_std"]) + torch.tensor(training_set1.model["thickness_mu"])
                _, slant = latent_sampler2.sampling_slant(thickness.reshape(-1, 1), normalize=True, model_=training_set1.model)
                phase_real_c1 = torch.cat([phase_real_c1, slant], dim=1).to(device).split(batch_gpu)
                ## estimate intensities
                thickness = phase_real_c2[:, 0] * torch.tensor(training_set.model["thickness_std"]) + torch.tensor(training_set.model["thickness_mu"])
                _, intensity = latent_sampler1.sampling_intensity(thickness.reshape(-1, 1), normalize=True, model_=training_set.model)
                phase_real_c2 = torch.cat([phase_real_c2[:, 0].reshape(-1, 1), intensity, phase_real_c2[:, 1].reshape(-1, 1)], 
                                    dim=1).to(device).split(batch_gpu)
            elif training_set.data_name == "ukb" or training_set.data_name == "retinal" : 
                ## estimate c2, c3
                ## estimate ventricle volumes for source 1
                age = phase_real_c1[:, 0] * torch.tensor(training_set1.model["age_max"] - training_set1.model["age_min"]) + torch.tensor(training_set1.model["age_min"])
                gen_c1_w_ventr = latent_sampler2.sampling_given_age(age, normalize=True)
                phase_real_c1 = torch.cat([phase_real_c1, gen_c1_w_ventr[:, -1].reshape(-1, 1)], dim=1).to(device).split(batch_gpu)
                ## estimate brain volumes for source 2
                age = phase_real_c2[:, 0] * torch.tensor(training_set.model["age_max"] - training_set.model["age_min"]) + torch.tensor(training_set.model["age_min"])
                gen_c2_w_brain = latent_sampler1.sampling_given_age(age, normalize=True)
                phase_real_c2 = torch.cat([phase_real_c2[:, 0].reshape(-1, 1), gen_c2_w_brain[:, -1].reshape(-1, 1), 
                                        phase_real_c2[:, -1].reshape(-1, 1)], dim=1).to(device).split(batch_gpu)
            else:
                raise NotImplementedError("Unknown dataset name: ", training_set.data_name)

            ### Gen data
            all_gen_z = torch.randn([len(phases) * batch_size * 2, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu * 2) for phase_gen_z in all_gen_z.split(batch_size * 2)]
            ## source 1
            all_gen_c1 = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            ## source 2
            all_gen_c2 = [training_set1.get_label(np.random.randint(len(training_set1))) for _ in range(len(phases) * batch_size)]

            all_gen_c1 = torch.from_numpy(np.stack(all_gen_c1)) ## (c1), c2
            all_gen_c2 = torch.from_numpy(np.stack(all_gen_c2)) ## (c1), c3

            ## estimate the latent space (hidden variables)
            if training_set.data_name == "mnist-thickness-intensity" or training_set.data_name == "mnist-thickness-slant" or training_set.data_name == "mnist-thickness-intensity-slant": 
                ## estimate slants
                thickness = all_gen_c1[:, 0] * torch.tensor(training_set1.model["thickness_std"]) + torch.tensor(training_set1.model["thickness_mu"])
                _, slant = latent_sampler2.sampling_slant(thickness.reshape(-1, 1), normalize=True, model_=training_set1.model)
                all_gen_c1 = torch.cat([all_gen_c1, slant], dim=1).pin_memory().to(device)
                ## estimate intensities
                thickness = all_gen_c2[:, 0] * torch.tensor(training_set.model["thickness_std"]) + torch.tensor(training_set.model["thickness_mu"])
                _, intensity = latent_sampler1.sampling_intensity(thickness.reshape(-1, 1), normalize=True, model_=training_set.model)
                all_gen_c2 = torch.cat([all_gen_c2[:, 0].reshape(-1, 1), intensity, all_gen_c2[:, 1].reshape(-1, 1)], dim=1).pin_memory().to(device)
            elif training_set.data_name == "ukb" or training_set.data_name == "retinal": 
                ## estimate (c2, c3)
                ## estimate ventricle volumes for source 1
                age = all_gen_c1[:, 0] * torch.tensor(training_set1.model["age_max"] - training_set1.model["age_min"]) + torch.tensor(training_set1.model["age_min"])
                gen_c1_w_ventr = latent_sampler2.sampling_given_age(age, normalize=True)
                all_gen_c1 = torch.cat([all_gen_c1, gen_c1_w_ventr[:, -1].reshape(-1, 1)], dim=1).pin_memory().to(device)
                ## estimate brain volumes for source 2
                age = all_gen_c2[:, 0] * torch.tensor(training_set.model["age_max"] - training_set.model["age_min"]) + torch.tensor(training_set.model["age_min"])
                gen_c2_w_brain = latent_sampler1.sampling_given_age(age, normalize=True)
                all_gen_c2 = torch.cat([all_gen_c2[:, 0].reshape(-1, 1), gen_c2_w_brain[:, -1].reshape(-1, 1), 
                                        all_gen_c2[:, -1].reshape(-1, 1)], dim=1).pin_memory().to(device)
            else:
                raise NotImplementedError("Unknown dataset name: ", training_set.data_name)
            all_gen_c = torch.cat([all_gen_c1, all_gen_c2], dim=0)
            rand_idx = torch.randperm(all_gen_c.shape[0])
            all_gen_c = all_gen_c[rand_idx].view(*all_gen_c.shape)
            all_gen_c = [phase_gen_c.split(batch_gpu * 2) for phase_gen_c in all_gen_c.split(batch_size * 2)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c): ## Number of phases
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            ## source 1
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img1, phase_real_c1, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c,
                                        gen_z=gen_z[:batch_gpu], gen_c=gen_c[:batch_gpu], gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            ## source 2
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img2, phase_real_c2, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, 
                                        gen_z=gen_z[batch_gpu:], gen_c=gen_c[batch_gpu:], gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                # if phase.name in ["Dmain", "Dreg", "Dboth"]:
                #     ### TODO update certain weights
                #     pass
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            wandb_imgs = (images.transpose(0, 2, 3, 1) * 127.5 + 128).clip(0, 255).astype(np.uint8)
            wandb.log({'gen_imgs':[wandb.Image(img) for img in wandb_imgs[:8]]}, step=cur_nimg)
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=validation_set_kwargs, dataset_kwargs_1=validation_set_kwargs1,
                    sampler = (latent_sampler1, latent_sampler2),
                    num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    wandb.log(result_dict, step=cur_nimg)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            wandb.log(fields)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        # if stats_tfevents is not None:
        global_step = int(cur_nimg / 1e3)
        walltime = timestamp - start_time
        for name, value in stats_dict.items():
            # stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            wandb.log({name: value.mean}, step=cur_nimg)
        for name, value in stats_metrics.items():
            # stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            wandb.log({f"Metrics/{name}": value}, step=cur_nimg)
        # stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
