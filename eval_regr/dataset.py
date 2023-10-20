import os
from typing import Optional
import pytorch_lightning as pl
import torch
from  torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

### --- own --- ###
from eval_regr.eval_dataset import MorphoMNISTDataset_causal, UKBiobankMRIDataset2D, UKBiobankRetinalDataset2D
from eval_regr.eval_dataset import ConcatDataset

### data loader for Morpho-MNIST dataset
class MorphoMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir,
                 data_name: str, 
                 use_labels: bool = True,
                 xflip: bool = False, 
                 batch_size: int = 128,
                 covariate: str = "thickness",
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.use_labels = use_labels
        self.xflip = xflip
        self.batch_size = batch_size
        self.covariate = covariate
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        if stage == "fit":
            self.data_train_s1 = MorphoMNISTDataset_causal(os.path.join(self.data_dir, "source1"),
                                                           mode="train",
                                                           data_name=self.data_name,
                                                           use_labels=self.use_labels,
                                                           covariate=self.covariate,
                                                           xflip=self.xflip,
                                                           include_numbers=False)
            self.data_val_s1 = MorphoMNISTDataset_causal(os.path.join(self.data_dir, "source1"),
                                                         mode="val",
                                                         data_name=self.data_name,
                                                         use_labels=self.use_labels,
                                                         covariate=self.covariate,
                                                         xflip=self.xflip,
                                                         include_numbers=False)
            
            self.data_train_s2 = MorphoMNISTDataset_causal(os.path.join(self.data_dir, "source2"),
                                                           mode="train",
                                                           data_name=self.data_name,
                                                           use_labels=self.use_labels,
                                                           covariate=self.covariate,
                                                           xflip=self.xflip,
                                                           include_numbers=False)
            self.data_val_s2 = MorphoMNISTDataset_causal(os.path.join(self.data_dir, "source2"),
                                                         mode="val",
                                                         data_name=self.data_name,
                                                         use_labels=self.use_labels,
                                                         covariate=self.covariate,
                                                         xflip=self.xflip,
                                                         include_numbers=False)
            self.data_train = ConcatDataset(self.data_train_s1, self.data_train_s2)
            self.data_val = ConcatDataset(self.data_val_s1, self.data_val_s2)
            
        if stage == "test":
            self.data_test_s1 = MorphoMNISTDataset_causal(os.path.join(self.data_dir, "source1"),
                                                       mode="test",
                                                       data_name=self.data_name,
                                                       use_labels=self.use_labels,
                                                       covariate=self.covariate,
                                                       xflip=self.xflip,
                                                       include_numbers=False)
            self.data_test_s2 = MorphoMNISTDataset_causal(os.path.join(self.data_dir, "source2"),
                                                       mode="test",
                                                       data_name=self.data_name,
                                                       use_labels=self.use_labels,
                                                       covariate=self.covariate,
                                                       xflip=self.xflip,
                                                       include_numbers=False)
            self.data_test = ConcatDataset(self.data_test_s1, self.data_test_s2)
            
        if stage == "predict":
            pass
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

class UKBiobankMRIDataModule(MorphoMNISTDataModule):
    def __init__(self, data_dir,
                 data_name: str, 
                 use_labels: bool = True,
                 xflip: bool = False, 
                 batch_size: int = 128,
                 covariate: str = "age",
                 num_workers: int = 4):
        super().__init__(data_dir, data_name, use_labels, xflip, batch_size, covariate, num_workers)
    def setup(self, stage: str):
        if stage == "fit":
            self.data_train_s1 = UKBiobankMRIDataset2D(os.path.join(self.data_dir, "source1"),
                                                           mode="train",
                                                           data_name=self.data_name,
                                                           use_labels=self.use_labels,
                                                           covariate=self.covariate,
                                                           xflip=self.xflip)
            self.data_val_s1 = UKBiobankMRIDataset2D(os.path.join(self.data_dir, "source1"),
                                                         mode="val",
                                                         data_name=self.data_name,
                                                         use_labels=self.use_labels,
                                                         covariate=self.covariate,
                                                         xflip=self.xflip)
            
            self.data_train_s2 = UKBiobankMRIDataset2D(os.path.join(self.data_dir, "source2"),
                                                           mode="train",
                                                           data_name=self.data_name,
                                                           use_labels=self.use_labels,
                                                           covariate=self.covariate,
                                                           xflip=self.xflip)
            self.data_val_s2 = UKBiobankMRIDataset2D(os.path.join(self.data_dir, "source2"),
                                                         mode="val",
                                                         data_name=self.data_name,
                                                         use_labels=self.use_labels,
                                                         covariate=self.covariate,
                                                         xflip=self.xflip)
            self.data_train = ConcatDataset(self.data_train_s1, self.data_train_s2)
            self.data_val = ConcatDataset(self.data_val_s1, self.data_val_s2)
            
        if stage == "test":
            self.data_test_s1 = UKBiobankMRIDataset2D(os.path.join(self.data_dir, "source1"),
                                                       mode="test",
                                                       data_name=self.data_name,
                                                       use_labels=self.use_labels,
                                                       covariate=self.covariate,
                                                       xflip=self.xflip)
            self.data_test_s2 = UKBiobankMRIDataset2D(os.path.join(self.data_dir, "source2"),
                                                       mode="test",
                                                       data_name=self.data_name,
                                                       use_labels=self.use_labels,
                                                       covariate=self.covariate,
                                                       xflip=self.xflip)
            self.data_test = ConcatDataset(self.data_test_s1, self.data_test_s2)
            
        if stage == "predict":
            pass

class UKBiobankRetinaDataModule(MorphoMNISTDataModule):
    def __init__(self, data_dir,
                 data_name: str, 
                 use_labels: bool = True,
                 xflip: bool = False, 
                 batch_size: int = 128,
                 covariate: str = "age",
                 num_workers: int = 4):
        super().__init__(data_dir, data_name, use_labels, xflip, batch_size, covariate, num_workers)
    
    def setup(self, stage: str):
        if stage == "fit":
            self.data_train_s1 = UKBiobankRetinalDataset2D(os.path.join(self.data_dir, "source1"),
                                                           mode="train",
                                                           data_name=self.data_name,
                                                           use_labels=self.use_labels,
                                                           covariate=self.covariate,
                                                           xflip=self.xflip)
            self.data_val_s1 = UKBiobankRetinalDataset2D(os.path.join(self.data_dir, "source1"),
                                                         mode="val",
                                                         data_name=self.data_name,
                                                         use_labels=self.use_labels,
                                                         covariate=self.covariate,
                                                         xflip=self.xflip)
            
            self.data_train_s2 = UKBiobankRetinalDataset2D(os.path.join(self.data_dir, "source2"),
                                                           mode="train",
                                                           data_name=self.data_name,
                                                           use_labels=self.use_labels,
                                                           covariate=self.covariate,
                                                           xflip=self.xflip)
            self.data_val_s2 = UKBiobankRetinalDataset2D(os.path.join(self.data_dir, "source2"),
                                                         mode="val",
                                                         data_name=self.data_name,
                                                         use_labels=self.use_labels,
                                                         covariate=self.covariate,
                                                         xflip=self.xflip)
            self.data_train = ConcatDataset(self.data_train_s1, self.data_train_s2)
            self.data_val = ConcatDataset(self.data_val_s1, self.data_val_s2)
            
        if stage == "test":
            self.data_test_s1 = UKBiobankRetinalDataset2D(os.path.join(self.data_dir, "source1"),
                                                       mode="test",
                                                       data_name=self.data_name,
                                                       use_labels=self.use_labels,
                                                       covariate=self.covariate,
                                                       xflip=self.xflip)
            self.data_test_s2 = UKBiobankRetinalDataset2D(os.path.join(self.data_dir, "source2"),
                                                       mode="test",
                                                       data_name=self.data_name,
                                                       use_labels=self.use_labels,
                                                       covariate=self.covariate,
                                                       xflip=self.xflip)
            self.data_test = ConcatDataset(self.data_test_s1, self.data_test_s2)
            
        if stage == "predict":
            pass

class MNISTDataModule(MorphoMNISTDataModule):
    def __init__(self, data_dir, data_name: str, use_labels: bool = True, 
                 xflip: bool = False, batch_size: int = 128, covariate: str = "thickness", 
                 num_workers: int = 4
    ):
        super().__init__(data_dir, data_name, use_labels, xflip, batch_size, covariate, num_workers)


    def setup(self, stage: str):
        if stage == "fit":
            self.data_train = MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
            self.data_train, self.data_val = torch.utils.data.random_split(self.data_train, [55000, 5000])
        if stage == "test":
            self.data_val = MNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())