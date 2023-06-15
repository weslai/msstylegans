# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib

### Own ###
from latent_dist_morphomnist import  preprocess_samples, normalize_samples
from training.random_variable_estimation import slant_estimation, intensity_estimation, cdr_estimation, volumes_estimation
#----------------------------------------------------------------------------

class MetricOptionsGeneral:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, causal_samplers=None,
        data_model = None, hybrid: bool = False, eval_dataset: str = None, num_samples=None, 
        num_gpus=1, rank=0, device=None, progress=None, cache=False):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        if type(dataset_kwargs) == list:
            self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs[0])
            self.dataset_kwargs1 = dnnlib.EasyDict(dataset_kwargs[1])
        else:
            assert eval_dataset == "inner"
            self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
            self.dataset_kwargs1 = None
        self.causal_samplers = causal_samplers
        self.data_model = data_model
        self.hybrid = hybrid
        self.eval_dataset = eval_dataset
        self.num_samples    = num_samples
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.causal_samplers is not None: ## causal
        if type(opts.causal_samplers) == list:
            causal_sampler = opts.causal_samplers[0]
        else:
            causal_sampler = opts.causal_samplers
        while True:
            if opts.dataset_kwargs.data_name == "mnist-thickness-intensity" or opts.dataset_kwargs.data_name == "mnist-thickness-slant": ## MorphoMNIST
                gen_c = sample_new(dataset=opts.dataset_kwargs.data_name,
                            model=causal_sampler, num_samples=batch_size,
                            include_numbers=opts.dataset_kwargs.include_numbers)
                c = normalize_samples(*gen_c, dataset=opts.dataset_kwargs.data_name,
                            model=opts.data_model[0])
                # c = preprocess_samples(*gen_c, dataset=opts.dataset_kwargs.data_name)
            else:
                c = causal_sampler.sample_normalize(batch_size)
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c
    else:
        if opts.G.c_dim == 0: ## unconditional
            c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
            while True:
                yield c
        else: ## conditional
            dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
            while True:
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                yield c

## second dataset
def iterate_random_labels1(opts, batch_size):
    if opts.causal_samplers is not None: ## causal
        if type(opts.causal_samplers) == list:
            causal_sampler = opts.causal_samplers[1]
        else:
            causal_sampler = opts.causal_samplers
        while True:
            if opts.dataset_kwargs1.data_name == "mnist-thickness-intensity" or opts.dataset_kwargs1.data_name == "mnist-thickness-slant": ## MorphoMNIST
                gen_c = sample_new(dataset=opts.dataset_kwargs1.data_name,
                            model=causal_sampler, num_samples=batch_size,
                            include_numbers=opts.dataset_kwargs1.include_numbers)
                c = normalize_samples(*gen_c, dataset=opts.dataset_kwargs1.data_name,
                                        model=opts.data_model[1])
                # c = preprocess_samples(*gen_c, dataset=opts.dataset_kwargs1.data_name)
            else:
                c = causal_sampler.sample_normalize(batch_size)
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c
    else:
        if opts.G.c_dim == 0: ## unconditional
            c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
            while True:
                yield c
        else: ## conditional
            dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs1)
            while True:
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs) # subclass of training.dataset.Dataset
    if opts.dataset_kwargs1 is not None:
        dataset1 = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs1)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        # data_loader_kwargs = dict(pin_memory=True, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, dataset_kwargs1=opts.dataset_kwargs1, 
                    detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        # cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        if opts.dataset_kwargs1 is not None:
            cache_tag = f'{dataset.data_name}-{dataset1.data_name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        else:
            cache_tag = f'{dataset.data_name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    if opts.eval_dataset == "both":
        total_items = len(dataset) + len(dataset1)
    elif opts.eval_dataset == "inner":
        total_items = len(dataset)
    elif opts.eval_dataset == "outer":
        total_items = len(dataset1)
    else:
        raise ValueError(f"Unknown eval_dataset: {opts.eval_dataset}")
    num_maxitems = min(total_items, max_items) if max_items is not None else total_items
    if opts.eval_dataset == "both":
        num_max_dataset = int(num_maxitems * (len(dataset) / (len(dataset) + len(dataset1))))
        num_max_dataset1 = num_maxitems - num_max_dataset
    else:
        num_max_dataset = num_maxitems

    stats = FeatureStats(max_items=num_maxitems, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_maxitems, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    if opts.eval_dataset == "inner" or opts.eval_dataset == "both":
        item_subset = [(i * opts.num_gpus + opts.rank) % num_max_dataset for i in range((num_max_dataset - 1) // opts.num_gpus + 1)]
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector(images.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.max_items)

    if opts.eval_dataset == "both":
        item_subset = [(i * opts.num_gpus + opts.rank) % num_max_dataset1 for i in range((num_max_dataset1 - 1) // opts.num_gpus + 1)]
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset1, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector(images.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.max_items)
    
    if opts.eval_dataset == "outer":
        item_subset = [(i * opts.num_gpus + opts.rank) % num_max_dataset for i in range((num_max_dataset - 1) // opts.num_gpus + 1)]
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset1, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector(images.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.max_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)
    if opts.hybrid: ## hybrid
        c_iter1 = iterate_random_labels1(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            if opts.hybrid: ## hybrid
                if opts.dataset_kwargs.data_name == "mnist-thickness-intensity" or opts.dataset_kwargs.data_name == "mnist-thickness-slant": ## MorphoMNIST
                    if opts.eval_dataset == "inner":
                        batch_c = next(c_iter)
                        if opts.dataset_kwargs.data_name == "mnist-thickness-intensity":
                            thickness, intensity = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                            classes = batch_c[:, 2:]
                            slant = torch.normal(mean=0, std=0.2, 
                                                size=(batch_c.shape[0], 1), device=opts.device)
                        elif opts.dataset_kwargs.data_name == "mnist-thickness-slant":
                            thickness, slant = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                            classes = batch_c[:, 2:]
                            intensity = torch.normal(mean=0, std=1.5,
                                                size=(batch_c.shape[0], 1), device=opts.device)
                            # intensity = torch.ones(size=(batch_c.shape[0], 1), device=opts.device)
                    elif opts.eval_dataset == "both":
                        if _i % 2 == 0:
                            batch_c = next(c_iter)
                            if opts.dataset_kwargs1.data_name == "mnist-thickness-slant":
                                thickness, intensity = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                                classes = batch_c[:, 2:]
                                slant = slant_estimation(thickness)
                            elif opts.dataset_kwargs1.data_name == "mnist-thickness-intensity":
                                thickness, slant = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                                classes = batch_c[:, 2:]
                                intensity = intensity_estimation(thickness)
                        else: 
                            batch_c = next(c_iter1)
                            if opts.dataset_kwargs.data_name == "mnist-thickness-slant":
                                thickness, intensity = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                                classes = batch_c[:, 2:]
                                slant = slant_estimation(thickness)
                            elif opts.dataset_kwargs.data_name == "mnist-thickness-intensity":
                                thickness, slant = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                                classes = batch_c[:, 2:]
                                intensity = intensity_estimation(thickness)
                    batch_c = torch.cat([thickness, intensity, slant, classes], dim=-1)
                elif opts.dataset_kwargs1.data_name == "ukb" or opts.dataset_kwargs1.data_name == "adni": ## MRI
                    if opts.eval_dataset == "inner":
                        batch_c = next(c_iter)
                        sex, age = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                        if opts.dataset_kwargs1.data_name == "adni":
                            ukb_model = opts.causal_samplers[0].get_causal_model()
                            adni_model = opts.causal_samplers[1].get_causal_model()
                            volumes_ukb = batch_c[:, 2:]
                            unnorm_age = ukb_model["min_age"] + (ukb_model["max_age"] - ukb_model["min_age"]) * age
                            cdr = cdr_estimation(sex=sex, age=unnorm_age, causalmodel=adni_model, normalize=True) ## sex, age -> cdr
                            cdr = torch.zeros(size=cdr.shape, device=opts.device) ## cdr -> 0
                            volumes_adni = volumes_estimation(sex=sex, age=unnorm_age, causalmodel=adni_model, cdr=cdr, normalize=True)
                            ### TODO
                            # volumes_adni = torch.random.normal(mean=volumes_adni, std=0.1) 
                        elif opts.dataset_kwargs1.data_name == "ukb":
                            ukb_model = opts.causal_samplers[1].get_causal_model()
                            adni_model = opts.causal_samplers[0].get_causal_model()
                            cdr = batch_c[:, 2:5]
                            volumes_adni = batch_c[:, 5:]
                            unnorm_age = adni_model["min_age"] + (adni_model["max_age"] - adni_model["min_age"]) * age
                            volumes_ukb = volumes_estimation(sex=sex, age=unnorm_age, causalmodel=ukb_model, normalize=True)
                    elif opts.eval_dataset == "both":
                        if _i % 2 == 0:
                            batch_c = next(c_iter)
                            sex, age = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                            if opts.dataset_kwargs1.data_name == "adni":
                                ukb_model = opts.causal_samplers[0].get_causal_model()
                                adni_model = opts.causal_samplers[1].get_causal_model()
                                volumes_ukb = batch_c[:, 2:]
                                unnorm_age = ukb_model["min_age"] + (ukb_model["max_age"] - ukb_model["min_age"]) * age
                                cdr = cdr_estimation(sex=sex, age=unnorm_age, causalmodel=adni_model, normalize=True) ## sex, age -> cdr
                                volumes_adni = volumes_estimation(sex=sex, age=unnorm_age, causalmodel=adni_model, cdr=cdr, normalize=True)
                            elif opts.dataset_kwargs1.data_name == "ukb":
                                ukb_model = opts.causal_samplers[1].get_causal_model()
                                adni_model = opts.causal_samplers[0].get_causal_model()
                                cdr = batch_c[:, 2:5]
                                volumes_adni = batch_c[:, 5:]
                                unnorm_age = adni_model["min_age"] + (adni_model["max_age"] - adni_model["min_age"]) * age
                                volumes_ukb = volumes_estimation(sex=sex, age=unnorm_age, causalmodel=ukb_model, normalize=True)
                        else:
                            batch_c = next(c_iter1)
                            sex, age = batch_c[:, 0].reshape(-1, 1), batch_c[:, 1].reshape(-1, 1)
                            if opts.dataset_kwargs.data_name == "adni":
                                ukb_model = opts.causal_samplers[1].get_causal_model()
                                adni_model = opts.causal_samplers[0].get_causal_model()
                                volumes_ukb = batch_c[:, 2:]
                                unnorm_age = ukb_model["min_age"] + (ukb_model["max_age"] - ukb_model["min_age"]) * age
                                cdr = cdr_estimation(sex=sex, age=unnorm_age, causalmodel=adni_model, normalize=True) ## sex, age -> cdr
                                volumes_adni = volumes_estimation(sex=sex, age=unnorm_age, causalmodel=adni_model, cdr=cdr, normalize=True)
                            elif opts.dataset_kwargs.data_name == "ukb":
                                ukb_model = opts.causal_samplers[0].get_causal_model()
                                adni_model = opts.causal_samplers[1].get_causal_model()
                                cdr = batch_c[:, 2:5]
                                volumes_adni = batch_c[:, 5:]
                                unnorm_age = adni_model["min_age"] + (adni_model["max_age"] - adni_model["min_age"]) * age
                                volumes_ukb = volumes_estimation(sex=sex, age=unnorm_age, causalmodel=ukb_model, normalize=True)
                    batch_c = torch.cat([sex, age, cdr, volumes_ukb, volumes_adni], dim=1)
                img = G(z=z, c=batch_c, **opts.G_kwargs)
            else: ## not hybrid
                batch_c = next(c_iter)
                img = G(z=z, c=batch_c, **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------
