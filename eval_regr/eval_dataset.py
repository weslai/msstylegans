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
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
import stylegan3.dnnlib as dnnlib
import nibabel as nib

try:
    import pyspng
except ImportError:
    pyspng = None
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

    def _get_norm_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            self._norm_labels = self.new_labels_norm if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._norm_labels, np.ndarray)
            assert self._norm_labels.shape[0] == self._raw_shape[0]
            assert self._norm_labels.dtype in [np.float32, np.int64]
            if self._norm_labels.dtype == np.int64:
                assert self._norm_labels.ndim == 1
                assert np.all(self._norm_labels >= 0)
        return self._norm_labels

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
        raise NotImplementedError

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()
    
    def get_norm_label(self, idx):
        label = self._get_norm_labels()[self._raw_idx[idx]]
        # if label.dtype == np.int64:
        #     onehot = np.zeros(self.label_shape, dtype=np.float32)
        #     onehot[label] = 1
        #     label = onehot
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
        self._mode = mode

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if data_name in ["adni", "ukb", "nacc"]:
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
        self._load_raw_image(0) # Validate image format.
        raw_shape = [len(self._image_fnames)] + list(self.image_arr.shape) ## [Size, C, W, H]
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
    
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        # self.image_arr = self.image_arr.transpose(1, 2, 0) # CHW => HWC
        image = image.transpose(1, 2, 0) # CHW => HWC
        # if self.image_arr.shape[-1] == 3:
        if image.shape[-1] == 3:
            image = PIL.Image.fromarray((image * 255).astype(np.uint8))
            if self._mode == "train":
                compose = transforms.Compose([
                    transforms.RandomRotation(degrees=90),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406],
                    #     std=[0.229, 0.224, 0.225]
                    # )
                ])
            else:
                # normalize = transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225]
                # )
                compose = transforms.Compose([
                    transforms.ToTensor(),
                    # normalize
                ])
        else:
            if image.ndim == 2:
                image = image[:, :, np.newaxis] # HW => HWC
            image = image.transpose(2, 0, 1) # HWC => HWC
            assert isinstance(image, np.ndarray)
            assert list(image.shape) == self.image_shape
            assert image.dtype == np.uint8
            if self._xflip[idx]:
                assert image.ndim == 3 # CHW
                image = image[:, :, ::-1]
            if self._mode == "train":
                compose = transforms.Compose([
                    # transforms.RandomRotation(degrees=15),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            else:
                compose = transforms.Compose([
                    transforms.ToTensor(),
                ])
        image = compose(image).cpu().detach().numpy()
        return image.copy(), self.get_norm_label(idx).copy()

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        if self.data_name in ["adni", "ukb", "nacc"]:
            path = os.path.join(self._path, fname)
            image = nib.load(path).get_fdata().astype(np.uint8)
            self.image_arr = image.copy()
        else:
            with self._open_file(fname) as f:
                if pyspng is not None and self._file_ext(fname) == '.png':
                    image = pyspng.load(f.read())
                    self.image_arr = image.copy()
                else:
                    # self.image_arr = np.array(PIL.Image.open(f))
                    image = PIL.Image.open(f)
                    self.image_arr = np.array(image)
                    if self.image_arr.ndim == 2:
                        self.image_arr = self.image_arr[:, :, np.newaxis] # HW => HWC
                    self.image_arr = self.image_arr.transpose(2, 0, 1) # HWC => CHW
        # return image
        return self.image_arr

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
#----------------------------------------------------------------------------
## images and labels (UKB with DAG graph (causal))
class UKBiobankMRIDataset2D(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "ukb",
        covariate: str = "age", ## ["age", "ventricle", "grey_matter"]
        **super_kwargs
    ):
        assert mode in ["train", "val", "test"]
        assert covariate in ["age", "ventricle", "grey_matter"]
        self.mode = mode
        self.data_name = data_name
        self.covariate = covariate
        ## which source
        self.which_source = [path.split("/")[-1], path.split("/")[-2]]
        for source in self.which_source:
            if len(source.split("_")) > 1:
                s = source.split("_")[-2]
                if s.startswith("source"):
                    self.which_source = s
                    break
        self.log_volumes_source1, self.log_volumes_source2 = get_settings(data_name)

        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
    def _get_mu_std(self, labels=None):
        model = dict()
        if self.mode != "train":
            assert self._path.endswith("valset/") or self._path.endswith("testset/")
            if self.mode == "val":
                self.train_path = os.path.join(self._path.split("valset")[0], "trainset/")
            elif self.mode == "test":
                self.train_path = os.path.join(self._path.split("testset")[0], "trainset/")
            self._type = 'dir'
            self.train_all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.train_path) for root, _dirs, files in os.walk(self.train_path) for fname in files}
            self.train_image_fnames = sorted(fname for fname in self.train_all_fnames if self._file_ext(fname) in ".gz")
            fname = "dataset.json"
            if fname not in self.train_all_fnames:
                return None
            with open(os.path.join(self.train_path, fname), 'rb') as f:
                trainlabels = json.load(f)["labels"]
            if trainlabels is None:
                return None
            trainlabels = dict(trainlabels)
            trainlabels = [trainlabels[fname.replace("\\", "/")] for fname in self.train_image_fnames] ## a dict

            new_labels = np.zeros(shape=(len(trainlabels), len(self.vars)), dtype=np.float32)
            for num, l in enumerate(trainlabels):
                i = list(l[self.vars[0]].items())[0][0]
                temp = [l[var][str(i)] for var in self.vars]
                new_labels[num, :] = temp
            labels = new_labels

        model.update(
            age_mu = np.mean(labels[:, 0]),
            age_std = np.std(labels[:, 0]),
            age_min = np.min(labels[:, 0]),
            age_max = np.max(labels[:, 0]),
            ventricle_mu = np.mean(np.log(labels[:, 1])) if self.log_volumes_source1 else np.mean(labels[:, 1]),
            ventricle_std = np.std(np.log(labels[:, 1])) if self.log_volumes_source1 else np.std(labels[:, 1]),
            grey_matter_mu = np.mean(np.log(labels[:, 2])) if self.log_volumes_source2 else np.mean(labels[:, 2]),
            grey_matter_std = np.std(np.log(labels[:, 2])) if self.log_volumes_source2 else np.std(labels[:, 2]),
        )
        return model

    def _normalise_labels(self, age, ventricle=None, grey_matter=None):
        ## zero mean normalisation
        age = (age - self.model["age_min"]) / (self.model["age_max"] - self.model["age_min"])
        if self.log_volumes_source1:
            ventricle = np.log(ventricle)
        ventricle = (ventricle - self.model["ventricle_mu"]) / self.model["ventricle_std"]
        if self.log_volumes_source2:
            grey_matter = np.log(grey_matter)
        grey_matter = (grey_matter - self.model["grey_matter_mu"]) / self.model["grey_matter_std"]
        samples = np.concatenate([age, ventricle, grey_matter], 1)
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
        self.vars = ["age", "ventricle", "grey_matter"]

        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[var][str(i)] for var in self.vars]
            new_labels[num, :] = temp
        self.model = self._get_mu_std(new_labels)
        self.new_labels_norm = self._normalise_labels(
            age=new_labels[:, 0].reshape(-1, 1),
            ventricle=new_labels[:, 1].reshape(-1, 1),
            grey_matter=new_labels[:, 2].reshape(-1, 1)
        )
        if self.covariate == "age":
            new_labels = new_labels[:, 0].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 0].reshape(-1, 1)
        elif self.covariate == "ventricle":
            new_labels = new_labels[:, 1].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 1].reshape(-1, 1)
        elif self.covariate == "grey_matter":
            new_labels = new_labels[:, 2].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 2].reshape(-1, 1)
        return new_labels
#----------------------------------------------------------------------------
class MorphoMNISTDataset_causal(ImageFolderDataset):
    def __init__(
        self,
        path, 
        resolution=None,
        mode: str = "train", ##[train, val, test]
        data_name: str = "mnist-thickness-intensity", ## two datasets: "mnist-thickness-intensity", "mnist-thickness-slant", "mnist-thickness-intensity-slant"
        covariate: str = "thickness", ## [thickness, intensity, slant]
        include_numbers: bool = False,                ## include numbers as classes for labels
        **super_kwargs
    ):
        assert mode in ["train", "val", "test"]
        assert covariate in ["thickness", "intensity", "slant"]
        self.mode = mode
        self.include_numbers = include_numbers
        self.data_name = data_name
        self.covariate = covariate
        ## which source
        self.which_source = [path.split("/")[-1], path.split("/")[-2]]
        for source in self.which_source:
            s = source.split("_")[-1]
            if s.startswith("source"):
                self.which_source = s
                break
        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)

    def _get_mu_std(self, labels=None):
        model = dict()
        if self.mode != "train":
            assert self._path.endswith("valset/") or self._path.endswith("testset/")
            if self.mode == "val":
                self.train_path = os.path.join(self._path.split("valset")[0], "trainset/")
            elif self.mode == "test":
                self.train_path = os.path.join(self._path.split("testset")[0], "trainset/")
            self._type = 'dir'
            self.train_all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.train_path) for root, _dirs, files in os.walk(self.train_path) for fname in files}
            PIL.Image.init()
            self.train_image_fnames = sorted(fname for fname in self.train_all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
            fname = "dataset.json"
            if fname not in self.train_all_fnames:
                return None
            with open(os.path.join(self.train_path, fname), 'rb') as f:
                trainlabels = json.load(f)["labels"]
            if trainlabels is None:
                return None
            trainlabels = dict(trainlabels)
            trainlabels = [trainlabels[fname.replace("\\", "/")] for fname in self.train_image_fnames] ## a dict

            new_labels = np.zeros(shape=(len(trainlabels), 3), dtype=np.float32)
            for num, l in enumerate(trainlabels):
                temp = [l[var] for var in self.vars]
                new_labels[num, :] = temp
            labels = new_labels
        model.update(
            thickness_mu = np.mean(labels[:, 0]),
            thickness_std = np.std(labels[:, 0]),
            intensity_mu = np.mean(labels[:, 1]),
            intensity_std = np.std(labels[:, 1]),
            slant_mu = np.mean(labels[:, 2]),
            slant_std = np.std(labels[:, 2])
        )
        return model

    def _normalise_labels(self, thickness, intensity=None, slant=None, classes=None):
        ## gamma normalized dist
        thickness = (thickness - self.model["thickness_mu"]) / self.model["thickness_std"]
        intensity = (intensity - self.model["intensity_mu"]) / self.model["intensity_std"]
        slant = (slant - self.model["slant_mu"]) / self.model["slant_std"]
        samples = np.concatenate([thickness, intensity, slant], 1)
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
        self.vars = ["thickness", "intensity", "slant"]

        new_labels = np.zeros(shape=(len(labels), 3+10), 
            dtype=np.float32) if self.include_numbers else np.zeros(shape=(len(labels), 3), dtype=np.float32)
        for num, l in enumerate(labels):
            temp = [l[var] for var in self.vars]
            if self.include_numbers:
                c = F.one_hot(torch.tensor(l["label"], dtype=torch.long), num_classes=10)
                ll = np.concatenate([np.array(temp), c.cpu().detach().numpy()], axis=-1)
            new_labels[num, :] = ll if self.include_numbers else temp
        self.model = self._get_mu_std(new_labels)
        self.new_labels_norm = self._normalise_labels(
            thickness=new_labels[:, 0].reshape(-1, 1),
            intensity=new_labels[:, 1].reshape(-1, 1),
            slant=new_labels[:, 2].reshape(-1, 1),
            classes=new_labels[:, 3:] if self.include_numbers else None)
        if self.covariate == "thickness":
            new_labels = new_labels[:, 0].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 0].reshape(-1, 1)
        elif self.covariate == "intensity":
            new_labels = new_labels[:, 1].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 1].reshape(-1, 1)
        elif self.covariate == "slant":
            new_labels = new_labels[:, 2].reshape(-1, 1)
            self.new_labels_norm =  self.new_labels_norm[:, 2].reshape(-1, 1)
        return new_labels

#----------------------------------------------------------------------------
class UKBiobankRetinalDataset2D(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "ukb",
        covariate: str = "age", ## ["age", "systolic", "diastolic", "spherical", "cylindrical"]
        **super_kwargs
    ):
        assert mode in ["train", "val", "test"]
        assert covariate in ["age", "systolic", "diastolic", "spherical", 
                             "cylindrical", "astigmatism_angle", "presbyopia", "cataract"]
        self.mode = mode
        self.data_name = data_name
        ## which source
        self.which_source = [path.split("/")[-1], path.split("/")[-2]]
        for source in self.which_source:
            if len(source.split("_")) > 1:
                s = source.split("_")[-2]
                if s.startswith("source"):
                    self.which_source = s
                    break
        self.log_volumes_source1, self.log_volumes_source2 = get_settings(data_name)
        self.covariate = covariate
        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
    def _get_mu_std(self, labels=None):
        model = dict()
        if self.mode != "train":
            assert self._path.endswith("valset/") or self._path.endswith("testset/")
            if self.mode == "val":
                self.train_path = os.path.join(self._path.split("valset")[0], "trainset/")
            elif self.mode == "test":
                self.train_path = os.path.join(self._path.split("testset")[0], "trainset/")
            self._type = 'dir'
            self.train_all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.train_path) for root, _dirs, files in os.walk(self.train_path) for fname in files}
            self.train_image_fnames = sorted(fname for fname in self.train_all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
            fname = "dataset.json"
            if fname not in self.train_all_fnames:
                return None
            with open(os.path.join(self.train_path, fname), 'rb') as f:
                trainlabels = json.load(f)["labels"]
            if trainlabels is None:
                return None
            trainlabels = dict(trainlabels)
            trainlabels = [trainlabels[fname.replace("\\", "/")] for fname in self.train_image_fnames] ## a dict

            new_labels = np.zeros(shape=(len(trainlabels), len(self.vars)), dtype=np.float32)
            for num, l in enumerate(trainlabels):
                i = list(l[self.vars[0]].items())[0][0]
                temp = [l[var][str(i)] for var in self.vars]
                new_labels[num, :] = temp
            labels = new_labels

        model.update(
            age_mu = np.mean(labels[:, 0]),
            age_std = np.std(labels[:, 0]),
            age_min = np.min(labels[:, 0]),
            age_max = np.max(labels[:, 0]),
            diastolic_bp_mu = np.mean(np.log(labels[:, 1])) if self.log_volumes_source1 else np.mean(labels[:, 1]),
            diastolic_bp_std = np.std(np.log(labels[:, 1])) if self.log_volumes_source1 else np.std(labels[:, 1]),
            # systolic_bp_mu = np.mean(np.log(labels[:, 2])) if self.log_volumes_source1 else np.mean(labels[:, 2]),
            # systolic_bp_std = np.std(np.log(labels[:, 2])) if self.log_volumes_source1 else np.std(labels[:, 2]),
            spherical_power_left_mu = np.mean(np.log(labels[:, 3] + 1e2)) if self.log_volumes_source2 else np.mean(labels[:, 3]),
            spherical_power_left_std = np.std(np.log(labels[:, 3] + 1e2)) if self.log_volumes_source2 else np.std(labels[:, 3]),
            cylindrical_power_left_mu = np.mean(np.log(labels[:, 4])) if self.log_volumes_source2 else np.mean(labels[:, 4]),
            cylindrical_power_left_std = np.std(np.log(labels[:, 4])) if self.log_volumes_source2 else np.std(labels[:, 4]),
            # astigmatism_angle_left_mu = np.mean(labels[:, 5]),
            # astigmatism_angle_left_std = np.std(labels[:, 5])
        )
        return model

    def _normalise_labels(self, age, diastolic_bp=None, systolic_bp=None, spherical_power_left=None, cylindrical_power_left=None):
                        #   astigmatism_angle_left=None):
        ## zero mean normalisation
        age = (age - self.model["age_min"]) / (self.model["age_max"] - self.model["age_min"])
        if self.log_volumes_source1:
            diastolic_bp = np.log(diastolic_bp)
            # systolic_bp = np.log(systolic_bp)
        bp = (diastolic_bp - self.model["diastolic_bp_mu"]) / self.model["diastolic_bp_std"]
        # systolic_encoder = OneHotEncoder().fit(systolic_bp)
        # systolic_bp = systolic_encoder.transform(systolic_bp).toarray()
        if self.log_volumes_source2:
            spherical_power_left = np.log(spherical_power_left + 1e2)
            cylindrical_power_left = np.log(cylindrical_power_left)
        spherical_power_left = (spherical_power_left - self.model["spherical_power_left_mu"]) / self.model["spherical_power_left_std"]
        cylindrical_power_left = (cylindrical_power_left - self.model["cylindrical_power_left_mu"]) / self.model["cylindrical_power_left_std"]
        # astigmatism_angle_left = astigmatism_angle_left / 90 - 1
        # astigmatism_angle_left = (astigmatism_angle_left - self.model["astigmatism_angle_left_mu"]) / self.model["astigmatism_angle_left_std"]
        samples = np.concatenate([age, bp, systolic_bp, spherical_power_left, cylindrical_power_left], 1)
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
        self.vars = ["age", "diastolic_bp", "systolic_bp", "spherical_power_left",
                     "cylindrical_power_left",
                      "astigmatism_angle_left", "presbyopia", "cataract"]

        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[var][str(i)] for var in self.vars]
            new_labels[num, :] = temp
        ## systolic_bp
        # normal_bp = (new_labels[:, 2] < 140)
        # # prehyper = np.where((140 > new_labels[:, 2]) & (new_labels[:, 2] >= 120))[0]
        # stage1 = np.where((160 > new_labels[:, 2]) & (new_labels[:, 2] >= 140))[0]
        # stage2 = (new_labels[:, 2] >= 160)
        # new_labels[normal_bp.reshape(-1), 2] = 0
        # # new_labels[prehyper.reshape(-1), 2] = 1
        # new_labels[stage1.reshape(-1), 2] = 1
        # new_labels[stage2.reshape(-1), 2] = 2

        self.model = self._get_mu_std(new_labels)
        self.new_labels_norm = self._normalise_labels(
            age=new_labels[:, 0].reshape(-1, 1),
            diastolic_bp=new_labels[:, 1].reshape(-1, 1),
            systolic_bp = new_labels[:, 2].reshape(-1, 1),
            spherical_power_left=new_labels[:, 3].reshape(-1, 1),
            cylindrical_power_left=new_labels[:, 4].reshape(-1, 1)
            # astigmatism_angle_left=new_labels[:, 5].reshape(-1, 1)
        )
        if self.covariate == "age":
            new_labels = new_labels[:, 0].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 0].reshape(-1, 1)
        elif self.covariate == "diastolic":
            new_labels = new_labels[:, 1].reshape(-1, 1).astype(np.float32)
            self.new_labels_norm = self.new_labels_norm[:, 1].reshape(-1, 1).astype(np.float32)
        elif self.covariate == "systolic":
            new_labels = new_labels[:, 2]
            self.new_labels_norm = self.new_labels_norm[:, 2].reshape(-1, 1).astype(np.float32)
            # self.new_labels_norm = self.new_labels_norm[:, 2:6]
        elif self.covariate == "spherical":
            new_labels = new_labels[:, 3].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 3].reshape(-1, 1).astype(np.float32)
        elif self.covariate == "cylindrical":
            new_labels = new_labels[:, 4].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 4].reshape(-1, 1)
        elif self.covariate == "astigmatism_angle":
            new_labels = new_labels[:, 5].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 5].reshape(-1, 1).astype(np.float32)
        elif self.covariate == "presbyopia":
            new_labels = new_labels[:, 6]
            self.new_labels_norm = new_labels.astype(np.int64)
        elif self.covariate == "cataract":
            new_labels = new_labels[:, 7]
            self.new_labels_norm = new_labels.astype(np.int64)
        return new_labels
#----------------------------------------------------------------------------
### Extra datasets
## ADNI
class AdniMRIDataset2D(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution = None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "adni",
        covariate: str = "age", ## ["age", "left_hippocampus", "right_hippocampus"]
        **super_kwargs
    ):
        assert mode in ["train", "val", "test"]
        assert covariate in ["age", "left_hippocampus", "right_hippocampus"]
        self.mode = mode
        self.path = path
        self.excluded_files = None
        self.data_name = data_name
        self.covariate = covariate
        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
    def _get_mu_std(self, labels=None):
        model = dict()
        if self.mode != "train":
            assert self._path.endswith("valset/") or self._path.endswith("testset/")
            if self.mode == "val":
                self.train_path = os.path.join(self._path.split("valset")[0], "trainset/")
            elif self.mode == "test":
                self.train_path = os.path.join(self._path.split("testset")[0], "trainset/")
            self._type = 'dir'
            self.train_all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.train_path) for root, _dirs, files in os.walk(self.train_path) for fname in files}
            self.train_image_fnames = sorted(fname for fname in self.train_all_fnames if self._file_ext(fname) in ".gz")
            fname = "dataset.json"
            if fname not in self.train_all_fnames:
                return None
            with open(os.path.join(self.train_path, fname), 'rb') as f:
                trainlabels = json.load(f)["labels"]
            if trainlabels is None:
                return None
            trainlabels = dict(trainlabels)
            trainlabels = [trainlabels[fname.replace("\\", "/")] for fname in self.train_image_fnames] ## a dict

            new_labels = np.zeros(shape=(len(trainlabels), len(self.vars)), dtype=np.float32)
            for num, l in enumerate(trainlabels):
                i = list(l[self.vars[0]].items())[0][0]
                temp = [l[var][str(i)] for var in self.vars]
                # ## cdr 
                # temp[1] = 1 if temp[1] >= 1.0 else temp[1]
                new_labels[num, :] = temp
            labels = new_labels
        model.update(
            age_mu = np.mean(labels[:, 0]),
            age_std = np.std(labels[:, 0]),
            age_min = np.min(labels[:, 0]),
            age_max = np.max(labels[:, 0]),
            # cdr_unique = np.unique(labels[:, 1]),
            left_hippo_mu = np.mean(labels[:, 1]),
            left_hippo_std = np.std(labels[:, 1]),
            right_hippo_mu = np.mean(labels[:, 2]),
            right_hippo_std = np.std(labels[:, 2])
        )
        return model

    def _normalise_labels(self, age, left_hippocampus, right_hippocampus):
        #cdr=None):
        ## zero mean normalisation
        age = (age - self.model["age_min"]) / (self.model["age_max"] - self.model["age_min"])
        left_hippocampus = (left_hippocampus - self.model["left_hippo_mu"]) / self.model["left_hippo_std"]
        right_hippocampus = (right_hippocampus - self.model["right_hippo_mu"]) / self.model["right_hippo_std"]
        # cdr_encoder = OneHotEncoder().fit(cdr)
        # cdr = cdr_encoder.transform(cdr).toarray()
        # cdr = cdr.astype(np.float32)
        samples = np.concatenate([age, left_hippocampus, right_hippocampus], 1)
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

        # self.vars = ["Age", "CDGLOBAL", "left_hippocampus", "right_hippocampus"]
        self.vars = ["Age", "left_hippocampus", "right_hippocampus"]
        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[VAR][str(i)] for VAR in self.vars]
            # ## cdr
            # temp[1] = 1 if temp[1] >= 1.0 else temp[1]
            new_labels[num, :] = temp
        self.model = self._get_mu_std(new_labels)
        self.new_labels_norm = self._normalise_labels(
            age=new_labels[:, 0].reshape(-1, 1),
            # cdr=new_labels[:, 1].reshape(-1, 1)
            left_hippocampus=new_labels[:, 1].reshape(-1, 1),
            right_hippocampus=new_labels[:, 2].reshape(-1, 1)
        )
        if self.covariate == "age":
            new_labels = new_labels[:, 0].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 0].reshape(-1, 1)
        elif self.covariate == "left_hippocampus":
            new_labels = new_labels[:, 1].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 1].reshape(-1, 1)
        elif self.covariate == "right_hippocampus":
            new_labels = new_labels[:, 2].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 2].reshape(-1, 1)
        return new_labels
## --------------------------------------------------------------------------
### NACC
class NACCMRIDataset2D(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "nacc",
        covariate: str = "age", ## ["age", "apoe"]
        **super_kwargs
    ):
        assert mode in ["train", "val", "test"]
        assert covariate in ["age", "apoe"]
        self.mode = mode
        self.path = path
        self.data_name = data_name
        self.covariate = covariate
        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)

    def _get_mu_std(self, labels=None):
        model = dict()
        if self.mode != "train" or labels is None:
            assert self._path.endswith("valset/") or self._path.endswith("testset/")
            if self.mode == "val":
                self.train_path = os.path.join(self._path.split("valset")[0], "trainset/")
            elif self.mode == "test":
                self.train_path = os.path.join(self._path.split("testset")[0], "trainset/")
            self._type = 'dir'
            self.train_all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.train_path) for root, _dirs, files in os.walk(self.train_path) for fname in files}
            self.train_image_fnames = sorted(fname for fname in self.train_all_fnames if self._file_ext(fname) in ".gz")
            fname = "dataset.json"
            if fname not in self.train_all_fnames:
                return None
            with open(os.path.join(self.train_path, fname), 'rb') as f:
                trainlabels = json.load(f)["labels"]
            if trainlabels is None:
                return None
            trainlabels = dict(trainlabels)
            trainlabels = [trainlabels[fname.replace("\\", "/")] for fname in self.train_image_fnames] ## a dict

            new_labels = np.zeros(shape=(len(trainlabels), len(self.vars)), dtype=np.float32)
            for num, l in enumerate(trainlabels):
                i = list(l[self.vars[0]].items())[0][0]
                temp = [l[var][str(i)] for var in self.vars]
                ## cdr 
                # temp[-1] = 1 if temp[-1] >= 1.0 else temp[-1]
                new_labels[num, :] = temp
            # self.excluded_files = np.where(new_labels[:, 1] == 9.0)[0]
            labels = new_labels[new_labels[:, 1] != 9.0]
        model.update(
            age_mu = np.mean(labels[:, 0]),
            age_std = np.std(labels[:, 0]),
            age_min = np.min(labels[:, 0]),
            age_max = np.max(labels[:, 0]),
            apoe_unique = np.unique(labels[:, 1]),
            # cdr_unique = np.unique(labels[:, -1]),
        )
        return model

    def _normalise_labels(self, age, apoe):
        # , cdr=None):
        ## zero mean normalisation
        age = (age - self.model["age_min"]) / (self.model["age_max"] - self.model["age_min"])
        apoe_encoder = OneHotEncoder().fit(apoe)
        apoe = apoe_encoder.transform(apoe).toarray()
        apoe = apoe.astype(np.float32)
        # cdr_encoder = OneHotEncoder().fit(cdr)
        # cdr = cdr_encoder.transform(cdr).toarray()
        # cdr = cdr.astype(np.float32)
        # samples = np.concatenate([age, apoe, cdr], 1)
        samples = np.concatenate([age, apoe], 1)
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

        self.vars = ["Age", "Apoe4"]
        # self.vars = ["Age", "Apoe4", "CDGLOBAL"]
        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[VAR][str(i)] for VAR in self.vars]
            ## cdr 
            # temp[-1] = 1 if temp[-1] >= 1.0 else temp[-1]
            new_labels[num, :] = temp
        idx_apoe4_9 = np.where(new_labels[:, 1] != 9.0)[0]
        new_labels = new_labels[new_labels[:, 1] != 9.0] ## remove missing APoe4
        self._image_fnames = [self._image_fnames[i] for i in idx_apoe4_9]
        self._raw_shape[0] = len(self._image_fnames) ## [Size, C, W, H]
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self.model = self._get_mu_std(new_labels)
        self.new_labels_norm = self._normalise_labels(
            age=new_labels[:, 0].reshape(-1, 1),
            apoe=new_labels[:, 1].reshape(-1, 1),
            # cdr=new_labels[:, -1].reshape(-1, 1)
        )
        if self.covariate == "age":
            new_labels = new_labels[:, 0].reshape(-1, 1)
            self.new_labels_norm = self.new_labels_norm[:, 0].reshape(-1, 1)
        elif self.covariate == "apoe":
            new_labels = new_labels[:, 1]
            self.new_labels_norm = new_labels.astype(np.int64)
        return new_labels
    
## --------------------------------------------------------------------------
## Extra Datasets with retinal images
## --------------------------------------------------------------------------
## images and labels
class KaggleEyepacsDataset(ImageFolderDataset):
    def __init__(
        self, 
        path, 
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "eyepacs",
        **super_kwargs
    ):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.data_name = data_name
        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
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
        self.vars = ["level"]

        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = l[self.vars[0]][str(i)]
            new_labels[num] = temp
        ## make it binary classification
        new_labels[new_labels > 0] = 1
        self.new_labels_norm = new_labels[:, 0].astype(np.int64)
        return new_labels.reshape(-1)
## --------------------------------------------------------------------------
## for RFMiD, we pick the three most common diseases
## This dataset is a multi-label problem
class RFMiDDataset(ImageFolderDataset):
    def __init__(
        self,
        path,
        resolution=None,
        mode: str = "train", ## ["train", "val", "test"]
        data_name: str = "rfmid",
        covariate: str = "MH", ## ["Disease_Risk", "MH", "TSLN"]
        **super_kwargs
    ):
        self.mode = mode
        self.data_name = data_name
        self.covariate = covariate
        super().__init__(data_name, self.mode, path, resolution, **super_kwargs)
    
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
        # self.vars = [
            #"Disease_Risk",
        #     "DR","ARMD","MH","DN","MYA",
        #     "BRVO","TSLN","ERM","LS","MS","CSR","ODC","CRVO",
        #     "TV","AH","ODP","ODE","ST","AION","PT","RT","RS","CRS",
        #     "EDN","RPEC","MHL","RP","CWS","CB","ODPM","PRH","MNF","HR",
        #     "CRAO","TD","CME","PTCR","CF","VH","MCA","VS","BRAO","PLQ","HPED","CL"
        # ]
        # self.vars = ["Disease_Risk", "DR", "ARMD", "MH", "MYA", "BRVO", "TSLN", "ODC", "ODP"]
        self.vars = ["Disease_Risk", "MH", "TSLN"]

        new_labels = np.zeros(shape=(len(labels), len(self.vars)), dtype=np.float32)
        for num, l in enumerate(labels):
            i = list(l[self.vars[0]].items())[0][0]
            temp = [l[VAR][str(i)] for VAR in self.vars]
            new_labels[num] = temp
        if self.covariate == "Disease_Risk":
            self.new_labels_norm = new_labels[:, 0].astype(np.int64)
            new_labels = new_labels[:, 0]
        elif self.covariate == "MH":
            self.new_labels_norm = new_labels[:, 1].astype(np.int64)
            new_labels = new_labels[:, 1]
        elif self.covariate == "TSLN":
            self.new_labels_norm = new_labels[:, 2].astype(np.int64)
            new_labels = new_labels[:, 2]
        return new_labels
## --------------------------------------------------------------------------
## Extra functions
## --------------------------------------------------------------------------
## get settings
def get_settings(dataset: str):
    if dataset == "ukb":
        log_volumes_source1, log_volumes_source2 = False, False
    elif dataset == "adni":
        log_volumes_source1, log_volumes_source2 = False, False
    elif dataset == "retinal":
        log_volumes_source1, log_volumes_source2 = False, False
    return log_volumes_source1, log_volumes_source2