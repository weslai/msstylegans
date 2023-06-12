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
from training.dataset import ConcatDataset
from latent_dist_morphomnist import normalize_samples
from training.random_variable_estimation import slant_estimation, intensity_estimation

INTERVENE_DICT_HYBRID = {
    ### For MorphoMNIST
    "thickness": 0,
    "intensity": 1, "slant": 2,
    ### For MRI
    "age": 0, 
    "brain": 1,
    "ventricle": 2
}
INTERVENE_DICT = {
    ### For MorphoMNIST
    "thickness": 0, 
    "intensity": 1, 
    "slant": 1, 
    ### For MRI
    "age": 0,
    "brain": 1,
    "ventricle": 1
}
#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, dataset_kwargs_1=None, samplers=None, 
                cur_bin: tuple = (0, 0), num_samples=None, 
                num_gpus=1, rank=0, device=None, progress=None, cache=False):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        if dataset_kwargs_1 is not None: ## hybrid
            self.dataset_kwargs_1 = dnnlib.EasyDict(dataset_kwargs_1)
            self.dataset_kwargs_1.update(max_size=None, xflip=False)
            self.sampler1, self.sampler2 = samplers
        self.cur_real_bin, self.cur_fake_bin = cur_bin
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
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        if opts.dataset_kwargs_1 is not None:
            dataset1 = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs_1)
            concat_dataset = ConcatDataset(dataset, dataset1)
            while True:
                c = []
                for _i in range(batch_size//2):
                    source1, source2 = concat_dataset[np.random.randint(len(concat_dataset))]
                    label1, label2 = source1[1], source2[1]
                    if dataset.data_name == "mnist-thickness-intensity":
                        ## estimation
                        c1 = label1[0] * dataset1.model["thickness_std"] + dataset1.model["thickness_mu"]
                        _, c3 = opts.sampler2.sampling_slant(torch.tensor(c1), normalize=True, model_=dataset1.model)
                        source1_labels = torch.concat([torch.tensor(label1).reshape(1, -1), c3], dim=1)
                        ## estimate intensities
                        c1 = label2[0] * dataset.model["thickness_std"] + dataset.model["thickness_mu"]
                        _, c2 = opts.sampler1.sampling_intensity(torch.tensor(c1), normalize=True, model_=dataset.model)
                        label2 = torch.tensor(label2)
                        source2_labels = torch.concat([label2[0].reshape(-1, 1), c2, label2[1].reshape(-1, 1)], dim=1)
                        concat_labels = torch.concat([source1_labels, source2_labels], dim=0)
                    elif dataset.data_name == "ukb":
                        ## estimation ventricle volumes
                        c1 = label1[0] * dataset1.model["age_std"] + dataset1.model["age_mu"]
                        label_w_c3 = opts.sampler2.sampling_given_age(torch.tensor(c1).reshape(-1, 1), normalize=True)
                        source1_labels = torch.concat([torch.tensor(label1).reshape(1, -1), label_w_c3[0, -1].reshape(-1, 1)], 
                                                      dim=1)
                        ## estimate brain volumes
                        c1 = label2[0] * dataset.model["age_std"] + dataset.model["age_mu"]
                        label_w_c2 = opts.sampler1.sampling_given_age(torch.tensor(c1).reshape(-1, 1), normalize=True)
                        label2 = torch.tensor(label2)
                        source2_labels = torch.concat(
                            [label2[0].reshape(1, -1), label_w_c2[0, -1].reshape(-1, 1), label2[1].reshape(1, -1)],
                            dim=1
                        )
                        concat_labels = torch.concat([source1_labels, source2_labels], dim=0)
                    else:
                        raise NotImplementedError("Unknown dataset")
                    c.append(concat_labels)
                c = torch.concat(c).pin_memory().to(opts.device)
                yield c
        else:
            while True:
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None, 
        cur_real_bin_min=0, cur_real_bin_max=1, cur_fake_bin_min=0, cur_fake_bin_max=1):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        
        ## real and fake bins
        self.cur_real_bin_min = cur_real_bin_min
        self.cur_real_bin_max = cur_real_bin_max
        self.cur_fake_bin_min = cur_fake_bin_min
        self.cur_fake_bin_max = cur_fake_bin_max

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
    
    def get_real_min_max_bin(self):
        return self.cur_real_bin_min, self.cur_real_bin_max
    def set_real_min_max_bin(self, cur_real_bin_min, cur_real_bin_max):
        self.cur_real_bin_min = cur_real_bin_min
        self.cur_real_bin_max = cur_real_bin_max
    def get_fake_min_max_bin(self):
        return self.cur_fake_bin_min, self.cur_fake_bin_max
    def set_fake_min_max_bin(self, cur_fake_bin_min, cur_fake_bin_max):
        self.cur_fake_bin_min = cur_fake_bin_min
        self.cur_fake_bin_max = cur_fake_bin_max

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
    if opts.dataset_kwargs_1 is not None:
        dataset1 = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs_1)
        concat_dataset = ConcatDataset(dataset, dataset1)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        # data_loader_kwargs = dict(pin_memory=True, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        if opts.dataset_kwargs_1 is not None:
            args = dict(dataset_kwargs=opts.dataset_kwargs, dataset_kwargs_1=opts.dataset_kwargs_1,
                        detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        else:
            args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        # cache_tag = f'{dataset.data_name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        if opts.dataset_kwargs_1 is not None:
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
    ## cov
    cov = opts.intervene
    cur_real_bin, cur_fake_bin = opts.cur_real_bin, opts.cur_fake_bin
    num_cov_bin = opts.num_intervene_bin
    cov_in_which_ds = opts.intervene_test_which_ds
    cov_in_which_index = INTERVENE_DICT[cov]
    ## load raw labels
    raw_labels = dataset._load_raw_labels()
    if opts.dataset_kwargs1 is not None:
        raw_labels1 = dataset1._load_raw_labels()
    if cov_in_which_ds == "both":
        assert opts.dataset_kwargs1 is not None
        if cov != "class": ## thickness, age, sex
            mean0, std0 = np.mean(raw_labels[:, cov_in_which_index]), np.std(raw_labels[:, cov_in_which_index])
            mean1, std1 = np.mean(raw_labels1[:, cov_in_which_index]), np.std(raw_labels1[:, cov_in_which_index])
            min_sec0, max_sec0 = mean0 - 2 * std0, mean0 + 2 * std0
            min_sec1, max_sec1 = mean1 - 2 * std1, mean1 + 2 * std1
            min_label = min(min_sec0, min_sec1)
            max_label = max(max_sec0, max_sec1)
    elif cov_in_which_ds == "first": ## intensity
        if cov != "class":
            mean0, std0 = np.mean(raw_labels[:, cov_in_which_index]), np.std(raw_labels[:, cov_in_which_index])
            min_label, max_label = mean0 - 2 * std0, mean0 + 2 * std0
    elif cov_in_which_ds == "second": ## slant
        assert opts.dataset_kwargs1 is not None
        if cov != "class":
            mean1, std1 = np.mean(raw_labels1[:, cov_in_which_index]), np.std(raw_labels1[:, cov_in_which_index])
            min_label, max_label = mean1 - 2 * std1, mean1 + 2 * std1
    else:
        raise NotImplementedError
    if cov == "class": ## class
        num_cov_bin = 9
        min_label, max_label = 0, 9
    elif cov == "cdr":
        num_cov_bin = 2
        min_label, max_label = 0, 1
    ## uniform bin
    cov_bin = np.linspace(min_label, max_label, num_cov_bin+1)
    if cov not in ["class", "sex", "cdr"]:
        if cov_in_which_ds == "both" or cov_in_which_ds == "first":
            cov_idxs = np.where((raw_labels[:, cov_in_which_index] >= cov_bin[cur_real_bin]) & (raw_labels[:, cov_in_which_index] < cov_bin[cur_real_bin+1]))[0]
        if cov_in_which_ds == "both" or cov_in_which_ds == "second":
            cov_idxs1 = np.where((raw_labels1[:, cov_in_which_index] >= cov_bin[cur_real_bin]) & (raw_labels1[:, cov_in_which_index] < cov_bin[cur_real_bin+1]))[0]
    elif cov in ["class", "cdr"]:
        if cov_in_which_ds == "both" or cov_in_which_ds == "first":
            cov_idxs = np.where(raw_labels[:, cov_in_which_index + cur_real_bin] == 1)[0]
        if cov_in_which_ds == "both" or cov_in_which_ds == "second":
            cov_idxs1 = np.where(raw_labels1[:, cov_in_which_index + cur_real_bin] == 1)[0]
    elif cov == "sex":
        if cov_in_which_ds == "both" or cov_in_which_ds == "first":
            cov_idxs = np.where(raw_labels[:, cov_in_which_index] == cur_real_bin)[0]
        if cov_in_which_ds == "both" or cov_in_which_ds == "second":
            cov_idxs1 = np.where(raw_labels1[:, cov_in_which_index] == cur_real_bin)[0]

    if cov_in_which_ds == "both":
        total_items = len(cov_idxs) + len(cov_idxs1)
        dataset2use = [dataset[i] for i in cov_idxs] 
        dataset2use1 = [dataset1[i] for i in cov_idxs1]
    elif cov_in_which_ds == "first":
        total_items = len(cov_idxs)
        dataset2use = [dataset[i] for i in cov_idxs]
    elif cov_in_which_ds == "second":
        total_items = len(cov_idxs1)
        dataset2use = [dataset1[i] for i in cov_idxs1]
    num_maxitems = min(total_items, max_items) if max_items is not None else total_items

    if cov_in_which_ds == "both":
        num_max_dataset = int(num_maxitems * (len(dataset2use) / (len(dataset2use) + len(dataset2use1))))
        num_max_dataset1 = num_maxitems - num_max_dataset
        print(f"num_max_dataset: {num_max_dataset}, num_max_dataset1: {num_max_dataset1}")
        print("total_items: ", num_maxitems)
    if cov not in ["class", "sex", "cdr"]:
        stats = FeatureStats(
            max_items=num_maxitems, 
            cur_real_bin_min=cov_bin[cur_real_bin], cur_real_bin_max=cov_bin[cur_real_bin+1],
            cur_fake_bin_min=cov_bin[cur_fake_bin], cur_fake_bin_max=cov_bin[cur_fake_bin+1],
            **stats_kwargs)
        stats.set_real_min_max_bin(cur_real_bin_min=cov_bin[cur_real_bin], cur_real_bin_max=cov_bin[cur_real_bin+1])
        stats.set_fake_min_max_bin(cur_fake_bin_min=cov_bin[cur_fake_bin], cur_fake_bin_max=cov_bin[cur_fake_bin+1])
    else:
        stats = FeatureStats(
            max_items=num_maxitems, 
            cur_real_bin_min=cov_bin[cur_real_bin], cur_real_bin_max=cov_bin[cur_real_bin],
            cur_fake_bin_min=cov_bin[cur_fake_bin], cur_fake_bin_max=cov_bin[cur_fake_bin],
            **stats_kwargs)
        stats.set_real_min_max_bin(cur_real_bin_min=cov_bin[cur_real_bin], cur_real_bin_max=cov_bin[cur_real_bin])
        stats.set_fake_min_max_bin(cur_fake_bin_min=cov_bin[cur_fake_bin], cur_fake_bin_max=cov_bin[cur_fake_bin])
    progress = opts.progress.sub(tag='dataset features', num_items=num_maxitems, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    if cov_in_which_ds != "both":
        item_subset = [(i * opts.num_gpus + opts.rank) % num_maxitems for i in range((num_maxitems - 1) // opts.num_gpus + 1)]
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset2use, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector(images.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.max_items)
    else:
        item_subset = [(i * opts.num_gpus + opts.rank) % num_max_dataset for i in range((num_max_dataset - 1) // opts.num_gpus + 1)]
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset2use, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector(images.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.max_items)
        item_subset1 = [(i * opts.num_gpus + opts.rank) % num_max_dataset1 for i in range((num_max_dataset1 - 1) // opts.num_gpus + 1)]
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset2use1, sampler=item_subset1, batch_size=batch_size, **data_loader_kwargs):
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

def compute_feature_stats_for_generator(opts,
    cur_fake_bin_min, cur_fake_bin_max, ## for stratified sampling
    detector_url, detector_kwargs,
    rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0
    ## init
    print("cur_fake_bin_min, cur_fake_bin_max", cur_fake_bin_min, cur_fake_bin_max)
    cov = opts.intervene
    num_cov_bin = opts.num_intervene_bin
    if opts.hybrid:
        cov_in_which_index = INTERVENE_DICT_HYBRID[cov]
    else:
        cov_in_which_index = INTERVENE_DICT[cov]
    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen) ## first causal model
    # if opts.dataset_kwargs1 is not None: ## hybrid
    #     c_iter1 = iterate_random_labels1(opts=opts, batch_size=batch_gen) ## second causal model

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
            batch_c = next(c_iter)
            # if opts.hybrid: ## two datasets
                # assert opts.dataset_kwargs1 is not None
                # if _i % 2 == 0:
                # else:
                #     batch_c = next(c_iter1)
            # else:
            #     batch_c = next(c_iter)
            ## random sampling in the bin
            if cov not in ["class", "cdr", "sex"]:
                rand_var = torch.tensor(np.random.choice(np.random.uniform(low=cur_fake_bin_min, high=cur_fake_bin_max, 
                                                                size=(int(stats.max_items//num_cov_bin),)), size=(batch_gen,))).to(opts.device)
            elif cov == "class":
                rand_var = torch.nn.functional.one_hot(torch.tensor(int(cur_fake_bin_min)), num_classes=10).to(opts.device).repeat(batch_gen, 1).float()
            elif cov == "cdr":
                rand_var = torch.nn.functional.one_hot(torch.tensor(int(cur_fake_bin_min)), num_classes=3).to(opts.device).repeat(batch_gen, 1).float()
            elif cov == "sex":
                rand_var = torch.tensor(np.random.choice([0, 1], size=(batch_gen,))).to(opts.device)
            else:
                raise NotImplementedError
            if opts.hybrid: ## hybrid also causal model
                ### random variable for Morpho-MNIST and MRI
                if opts.dataset_kwargs.data_name == "mnist-thickness-intensity" or opts.dataset_kwargs.data_name == "mnist-thickness-slant":
                    batch_c_hybrid = torch.zeros((batch_c.shape[0], batch_c.shape[1]+1)).to(opts.device)
                if cov not in ["class", "cdr"]:
                    batch_c_hybrid[:, cov_in_which_index] = rand_var
                # if cov in ["thickness", "sex", "age"]: ## thickness, sex, age
                #     batch_c[:, cov_in_which_index] = rand_var
                    # batch_c1[:, cov_in_which_index] = rand_var
                elif cov == "class": ## class
                    batch_c_hybrid[:, cov_in_which_index:] = rand_var
                    # batch_c1[:, cov_in_which_index:] = rand_var
                elif cov == "cdr": ## cdr in adni
                    batch_c_hybrid[:, cov_in_which_index:(cov_in_which_index+3)] = rand_var
                # elif cov in ["intensity", "ventricle", "brain", "intracranial"]: ## intensity and all vols in UKB
                #     batch_c[:, cov_in_which_index] = rand_var
                # elif cov in ["slant", "left_ventricle", "right_ventricle", "left_cortex", "right_cortex", "left_hippocampus", "right_hippocampus"]: ## slant, or all vols in adni
                #     batch_c1[:, cov_in_which_index] = rand_var
                else:
                    raise ValueError(f"cov {cov} not supported")
                if opts.dataset_kwargs.data_name == "mnist-thickness-intensity":
                    ### Common Variables set to the same value
                    # if _i % 2 == 0: ## thickness-intensity -> slant
                    if cov == "thickness":
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["intensity"]] = intensity_estimation(rand_var.reshape(-1, 1)).reshape(-1,) ## intensity
                        slant = torch.normal(mean=0, std=0.2, size=(batch_c.shape[0], 1))
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["slant"]] = slant.reshape(-1,).to(opts.device) ## slant
                        # batch_c_hybrid[:, INTERVENE_DICT_HYBRID["slant"]] = slant_estimation(rand_var.reshape(-1, 1)).reshape(-1,) ## slant
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["class"]:] = batch_c[:, 2:] ## class
                    elif cov == "intensity":
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["thickness"]] = batch_c[:, 0] ## thickness
                        slant = torch.normal(mean=0, std=0.2, size=(batch_c.shape[0], 1))
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["slant"]] = slant.reshape(-1,).to(opts.device) ## slant
                        # batch_c_hybrid[:, INTERVENE_DICT_HYBRID["slant"]] = slant_estimation(batch_c[:, 0].reshape(-1, 1)).reshape(-1,) ## slant
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["class"]:] = batch_c[:, 2:] ## class
                    elif cov == "slant":
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["thickness"]] = batch_c[:, 0] ## thickness
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["intensity"]] = batch_c[:, 1] ## intensity
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["class"]:] = batch_c[:, 2:] ## class
                elif opts.dataset_kwargs.data_name == "mnist-thickness-slant":
                    # else: ## thickness-slant -> intensity
                    if cov == "thickness":
                        intensity = torch.normal(mean=0, std=1, 
                                                size=(batch_c.shape[0], 1), device=opts.device)
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["intensity"]] = intensity.reshape(-1,) ## intensity
                        # batch_c_hybrid[:, INTERVENE_DICT_HYBRID["intensity"]] = intensity_estimation(rand_var.reshape(-1, 1)).reshape(-1,) ## intensity
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["slant"]] = slant_estimation(rand_var.reshape(-1, 1)).reshape(-1,) ## slant
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["class"]:] = batch_c[:, 2:] ## class
                    elif cov == "intensity":
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["thickness"]] = batch_c[:, 0] ## thickness
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["slant"]] = batch_c[:, 1] ## slant
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["class"]:] = batch_c[:, 2:] ## class
                    elif cov == "slant":
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["thickness"]] = batch_c[:, 0] ## thickness
                        intensity = torch.normal(mean=0, std=1, 
                                                size=(batch_c.shape[0], 1), device=opts.device)
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["intensity"]] = intensity.reshape(-1,) ## intensity
                        # batch_c_hybrid[:, INTERVENE_DICT_HYBRID["intensity"]] = intensity_estimation(batch_c[:, 0].reshape(-1, 1)).reshape(-1,) ## intensity
                        batch_c_hybrid[:, INTERVENE_DICT_HYBRID["class"]:] = batch_c[:, 2:] ## class
                else: ### MRI ## Sex and Age
                    if _i % 2 == 0:
                        if cov != "sex":
                            batch_c1[:, 0] = batch_c[:, 0]
                        if cov != "age":
                            batch_c1[:, 1] = batch_c[:, 1]
                    else:
                        if cov != "sex":
                            batch_c[:, 0] = batch_c1[:, 0]
                        if cov != "age":
                            batch_c[:, 1] = batch_c1[:, 1]

                # batch_c = torch.cat([batch_c, batch_c1], dim=-1)
                img = G(z=z, c=batch_c_hybrid, **opts.G_kwargs)
            else: ## not hybrid
                if opts.causal_samplers is not None: ## Causal
                    if cov == "thickness": ## change intensity or slant
                        if opts.dataset_kwargs.data_name == "mnist-thickness-intensity":
                            intensity = intensity_estimation(rand_var.reshape(-1, 1))
                            batch_c[:, INTERVENE_DICT["intensity"]] = intensity.reshape(-1,)
                        elif opts.dataset_kwargs.data_name == "mnist-thickness-slant":
                            slant = slant_estimation(rand_var.reshape(-1, 1))
                            batch_c[:, INTERVENE_DICT["slant"]] = slant.reshape(-1,)
                        batch_c[:, INTERVENE_DICT["thickness"]] = rand_var
                    elif cov == "intensity" or cov == "slant": ## change 
                        batch_c[:, cov_in_which_index] = rand_var
                    elif cov == "class":
                        batch_c[:, cov_in_which_index:] = rand_var
                else:
                    ## if cond then it just replace the covariate with the random variable
                    if cov not in ["class", "cdr"]:
                        batch_c[:, cov_in_which_index] = rand_var
                    elif cov == "class":
                        batch_c[:, cov_in_which_index:] = rand_var
                    elif cov == "cdr":
                        batch_c[:, cov_in_which_index:(cov_in_which_index+3)] = rand_var
                # which_lm = 0 if opts.dataset_kwargs.data_name == "mnist-thickness-intensity" else 1
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
