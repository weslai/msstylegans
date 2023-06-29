# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import torch.nn.functional as F
import dnnlib
import nibabel as nib

try:
    import pyspng
except ImportError:
    pyspng = None

### -----------
### --- Own ---
### -----------
from latent_mle import CausalSampling, preprocess_samples
#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW). (Num, Ch, H, W)
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
       return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        data_name,              # Name of the dataset.
        mode: str,               # ["train", "val", "test"] the mode of the dataset
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        excluded_files  = None, # List of files to exclude. 
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.data_name = data_name
        self._path = path
        self._zipfile = None
        self._check_mode(mode)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if data_name in ["adni", "ukb"]:
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in ".gz")
        else:
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        ## exclusive for adni
        ## remove the deleted files
        if excluded_files is not None:
            self.excluded_indices = []
            for excluded in excluded_files:
                self.excluded_indices.append(self._image_fnames.index(excluded))
                self._image_fnames.remove(excluded)

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape) ## [Size, C, W, H]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None
    
    def _check_mode(self, mode):
        assert mode in ["train", "val", "test"]
        if mode == "train":
            if not self._path.endswith("trainset") or not self._path.endswith("trainset/"):
                self._path = os.path.join(self._path, "trainset/")
        elif mode == "val":
            if not self._path.endswith("valset") or not self._path.endswith("valset/"):
                self._path = os.path.join(self._path, "valset/")
        elif mode == "test":
            if not self._path.endswith("testset") or not self._path.endswith("testset/"):
                self._path = os.path.join(self._path, "testset/")
        assert self._path.endswith("trainset/") or self._path.endswith("testset/") or self._path.endswith("valset/")

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        if self.data_name in ["adni", "ukb"]:
            path = os.path.join(self._path, fname)
            image = nib.load(path).get_fdata().astype(np.uint8)
        else:
            with self._open_file(fname) as f:
                if pyspng is not None and self._file_ext(fname) == '.png':
                    image = pyspng.load(f.read())
                else:
                    image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
## --------------------------------------------------------------------------
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
## --------------------------------------------------------------------------
## for images and labels (causal graph)
class OasisMRIDataset2D_Labels(ImageFolderDataset):
    def __init__(
        self, 
        path,
        resolution=None,
        data_name: str = "oasis3",
        **super_kwargs
    ):
        self.path = path

        ## load the causal graph and model
        self.sampler = CausalSampling(data_name)
        self.causal_graph = self.sampler.get_graph()
        self.causal_model = self.sampler.get_causal_model()
        self.vars = self.sampler.vars

        super().__init__(data_name, path, resolution, **super_kwargs)
    
    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames] ## a dict 
        

        new_labels = np.zeros(shape=(len(labels), len(labels[0])), dtype=np.float32)
        for i in range(len(labels)):
            new_labels[i, :] = [labels[i][VAR] for VAR in self.vars]
        new_labels = torch.tensor(new_labels, dtype=torch.float32)
        new_labels_ts = preprocess_samples(
            S = new_labels[:, 1].view(-1, 1),
            A = new_labels[:, 0].view(-1, 1),
            C = new_labels[:, 2].view(-1, 1),
            V = new_labels[:, 3:],
            model= self.causal_model,
            dataset=self.data_name
        )
        new_labels = new_labels_ts.detach().numpy().astype(np.float32)

        return new_labels ## return an array, each element is a dict {"mmse": ..., }
#----------------------------------------------------------------------------
## images and labels (UKB with DAG graph (causal))
class UKBiobankRetinalDataset(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "retinal",
        **super_kwargs
    ):
        self.mode = mode
        self.data_name = data_name
        ## which source
        self.which_source = [path.split("/")[-1], path.split("/")[-2]]
        for source in self.which_source:
            s = source.split("_")[-1]
            if s.startswith("source"):
                self.which_source = s
                break

        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
    def _get_mu_std(self, labels=None, which_source=None):
        model = dict()
        model.update(
            age_mu = np.mean(labels[:, 0]),
            age_std = np.std(labels[:, 0]),
            age_min = np.min(labels[:, 0]),
            age_max = np.max(labels[:, 0])
        )
        c_additional_mu = np.mean(labels[:, 1])
        c_additional_std = np.std(labels[:, 1])
        if which_source == "source1":
            model.update(
                systolic_bp_mu = c_additional_mu,
                systolic_bp_std = c_additional_std
            )
        elif which_source == "source2":
            model.update(
                cylindrical_power_left_mu = c_additional_mu,
                cylindrical_power_left_std = c_additional_std
            )
        return model

    def _normalise_labels(self, age, systolic_bp=None, cylindrical_power_left=None):
        ## zero mean normalisation
        age = (age - self.model["age_min"]) / (self.model["age_max"] - self.model["age_min"])
        if self.which_source == "source1":
            systolic_bp = (systolic_bp - self.model["systolic_bp_mu"]) / self.model["systolic_bp_std"]
            samples = np.concatenate([age, systolic_bp], 1)
        elif self.which_source == "source2":
            cylindrical_power_left = (cylindrical_power_left - self.model["cylindrical_power_left_mu"]) / self.model["cylindrical_power_left_std"]
            samples = np.concatenate([age, cylindrical_power_left], 1)
        return samples

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames] ## a dict 

        if self.which_source == "source1":
            self.vars = ["age", "systolic_bp"]
        elif self.which_source == "source2":
            self.vars = ["age", "cylindrical_power_left"]
        else:
            raise ValueError(f"No such source {self.which_source}")

        new_labels = np.zeros(shape=(len(labels), 2), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[var][str(i)] for var in self.vars]
            new_labels[num, :] = temp
        self.model = self._get_mu_std(new_labels, self.which_source)
        if self.which_source == "source1":
            new_labels = self._normalise_labels(
                age=new_labels[:, 0].reshape(-1, 1),
                systolic_bp=new_labels[:, 1].reshape(-1, 1)
            )
        elif self.which_source == "source2":
            new_labels = self._normalise_labels(
                age=new_labels[:, 0].reshape(-1, 1),
                cylindrical_power_left=new_labels[:, 1].reshape(-1, 1)
            )
        return new_labels
#----------------------------------------------------------------------------
## images and labels (UKB with DAG graph (causal))
class UKBiobankMRIDataset2D(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "ukb",
        **super_kwargs
    ):
        self.mode = mode
        self.data_name = data_name
        ## which source
        self.which_source = [path.split("/")[-1], path.split("/")[-2]]
        for source in self.which_source:
            s = source.split("_")[-1]
            if s.startswith("source"):
                self.which_source = s
                break

        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
    def _get_mu_std(self, labels=None, which_source=None):
        model = dict()
        model.update(
            age_mu = np.mean(labels[:, 0]),
            age_std = np.std(labels[:, 0]),
            age_min = np.min(labels[:, 0]),
            age_max = np.max(labels[:, 0])
        )
        c_additional_mu = np.mean(labels[:, 1])
        c_additional_std = np.std(labels[:, 1])
        if which_source == "source1":
            model.update(
                cortex_left_mu = c_additional_mu,
                cortex_left_std = c_additional_std
            )
        elif which_source == "source2":
            model.update(
                ventricle_mu = c_additional_mu,
                ventricle_std = c_additional_std
            )
        return model

    def _normalise_labels(self, age, cortex_left=None, ventricle=None):
        ## zero mean normalisation
        age = (age - self.model["age_min"]) / (self.model["age_max"] - self.model["age_min"])
        if self.which_source == "source1":
            cortex_left = (cortex_left - self.model["cortex_left_mu"]) / self.model["cortex_left_std"]
            samples = np.concatenate([age, cortex_left], 1)
        elif self.which_source == "source2":
            ventricle = (ventricle - self.model["ventricle_mu"]) / self.model["ventricle_std"]
            samples = np.concatenate([age, ventricle], 1)
        return samples

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames] ## a dict 

        if self.which_source == "source1":
            self.vars = ["age", "cortex_left"]
        elif self.which_source == "source2":
            self.vars = ["age", "ventricle"]
        else:
            raise ValueError(f"No such source {self.which_source}")

        new_labels = np.zeros(shape=(len(labels), 2), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[var][str(i)] for var in self.vars]
            new_labels[num, :] = temp
        self.model = self._get_mu_std(new_labels, self.which_source)
        if self.which_source == "source1":
            new_labels = self._normalise_labels(
                age=new_labels[:, 0].reshape(-1, 1),
                cortex_left=new_labels[:, 1].reshape(-1, 1)
            )
        elif self.which_source == "source2":
            new_labels = self._normalise_labels(
                age=new_labels[:, 0].reshape(-1, 1),
                ventricle=new_labels[:, 1].reshape(-1, 1)
            )

        return new_labels

#----------------------------------------------------------------------------
## images and labels (Adni with DAG graph (causal))
class AdniMRIDataset2D(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "adni",
        without_volumes: bool = False,
        **super_kwargs
    ):
        self.mode = mode
        self.path = path
        self.excluded_files = None
        # self.excluded_files = pd.read_csv("/dhc/home/wei-cheng.lai/causal-gan/data/adni/adni_excluded.csv")["delete"]
        ## load the causal graph and model
        self.sampler = CausalSampling(data_name,
            label_path=self.path
        )
        self.causal_graph = self.sampler.get_graph()
        self.causal_model = self.sampler.get_causal_model()
        self.vars = self.sampler.vars
        self.without_volumes = without_volumes

        super().__init__(data_name, self.mode, path, resolution, excluded_files=self.excluded_files, 
                         **super_kwargs)
    
    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames] ## a dict 

        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[VAR][str(i)] for VAR in self.vars]
            temp[1] = 1 if temp[1] == "M" else 0
            ## cdr 
            temp[2] = 1 if temp[2] >= 1.0 else temp[2]
            new_labels[num, :] = temp
        new_labels = torch.tensor(new_labels, dtype=torch.float32)
        new_labels_ts = preprocess_samples(
            S = new_labels[:, 1].view(-1, 1),
            A = new_labels[:, 0].view(-1, 1),
            C = new_labels[:, 2].view(-1, 1),
            V = new_labels[:, 3:],
            model= self.causal_model,
            dataset=self.data_name
        )
        new_labels = new_labels_ts.detach().numpy().astype(np.float32)
        if self.without_volumes:
            new_labels = new_labels[:, :5]
        return new_labels

#----------------------------------------------------------------------------
class MorphoMNISTDataset_causal(ImageFolderDataset):
    def __init__(
        self,
        path, 
        resolution=None,
        mode: str = "train", ##[train, val, test]
        data_name: str = "mnist-thickness-intensity", ## two datasets: "mnist-thickness-intensity", ""mnist-thickness-slant""
        include_numbers: bool = False,                ## include numbers as classes for labels
        **super_kwargs
    ):
        self.mode = mode
        self.include_numbers = include_numbers
        self.data_name = data_name
        ## which source 
        self.which_source = [path.split("/")[-1], path.split("/")[-2]]
        for source in self.which_source:
            s = source.split("_")[-1]
            if s.startswith("source"):
                self.which_source = s
                break

        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)

    def _get_mu_std(self, labels=None, data_name=None):
        model = dict()
        model.update(
            thickness_mu = np.mean(labels[:, 0]),
            thickness_std = np.std(labels[:, 0])
        )
        if data_name == "mnist-thickness-intensity" or (data_name == "mnist-thickness-intensity-slant" and self.which_source == "source1"):
            model.update(
                intensity_mu = np.mean(labels[:, 1]),
                intensity_std = np.std(labels[:, 1])
            )
        elif data_name == "mnist-thickness-slant" or (data_name == "mnist-thickness-intensity-slant" and self.which_source == "source2"):
            model.update(
                slant_mu = np.mean(labels[:, 1]),
                slant_std = np.std(labels[:, 1])
            )
        return model

    def _normalise_labels(self, thickness, intensity=None, slant=None, classes=None):
        ## gamma normalized dist
        thickness = (thickness - self.model["thickness_mu"]) / self.model["thickness_std"]
        if self.data_name == "mnist-thickness-intensity" or (self.data_name == "mnist-thickness-intensity-slant" and self.which_source == "source1"):
            intensity = (intensity - self.model["intensity_mu"]) / self.model["intensity_std"]
            samples = np.concatenate([thickness, intensity], 1)
        elif self.data_name == "mnist-thickness-slant" or (self.data_name == "mnist-thickness-intensity-slant" and self.which_source == "source2"):
            slant = (slant - self.model["slant_mu"]) / self.model["slant_std"]
            samples = np.concatenate([thickness, slant], 1)
        if classes is not None:
            if classes.shape[1] != 10:
                classes = F.one_hot(torch.tensor(classes, dtype=torch.long), num_classes=10).cpu().detach().numpy()
            samples = np.concatenate([samples, classes], 1)
        return samples

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames] ## a dict

        new_labels = np.zeros(shape=(len(labels), 2+10), 
            dtype=np.float32) if self.include_numbers else np.zeros(shape=(len(labels), 2), dtype=np.float32)
        if self.data_name == "mnist-thickness-intensity" or (self.data_name == "mnist-thickness-intensity-slant" and self.which_source == "source1"):
            for i in range(len(labels)):
                thickness = labels[i]["thickness"]
                intensity = labels[i]["intensity"]
                if self.include_numbers:
                    c = F.one_hot(torch.tensor(labels[i]["label"], dtype=torch.long), num_classes=10)
                    ll = np.concatenate([np.array([thickness, intensity]), c.cpu().detach().numpy()], axis=-1)
                new_labels[i, :] = ll if self.include_numbers else np.array([thickness, intensity], dtype=np.float32)
            self.model = self._get_mu_std(new_labels, self.data_name)
            new_labels = self._normalise_labels(
                thickness=new_labels[:, 0].reshape(-1, 1),
                intensity=new_labels[:, 1].reshape(-1, 1),
                slant=None,
                classes=new_labels[:, 2:] if self.include_numbers else None)

        elif self.data_name == "mnist-thickness-slant" or (self.data_name == "mnist-thickness-intensity-slant" and self.which_source == "source2"):
            for i in range(len(labels)):
                thickness = labels[i]["thickness"]
                slant = labels[i]["slant"]
                if self.include_numbers:
                    c = F.one_hot(torch.tensor(labels[i]["label"], dtype=torch.long), num_classes=10)
                    ll = np.concatenate([np.array([thickness, slant]), c.cpu().detach().numpy()], axis=-1)
                new_labels[i, :] = ll if self.include_numbers else np.array([thickness, slant], dtype=np.float32)
            self.model = self._get_mu_std(new_labels, self.data_name)
            new_labels = self._normalise_labels(
                thickness=new_labels[:, 0].reshape(-1, 1),
                intensity=None,
                slant=new_labels[:, 1].reshape(-1, 1),
                classes=new_labels[:, 2:] if self.include_numbers else None)
        else:
            raise ValueError(f"No such dataset {self.data_name}")
        return new_labels