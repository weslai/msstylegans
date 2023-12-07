import sys, os
sys.path.append("../")
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans/stylegan3")
import click
import json
import wandb
import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
import stylegan3.dnnlib as dnnlib

### --- own --- ###
from eval_regr.model import RegressionResnet, ClassificationResnet, MNISTClassificationResnet, MNISTRegressionResnet
from eval_regr.dataset import MorphoMNISTDataModule, UKBiobankMRIDataModule, UKBiobankRetinaDataModule
from eval_regr.dataset import AdniMRIDataModule, NACCMRIDataModule, EyepacsDataModule, RFMIDDataModule
from eval_regr.dataset import MNISTDataModule, CIFARDataModule
from eval_regr.eval_dataset import UKBiobankRetinalDataset2D, NACCMRIDataset2D, KaggleEyepacsDataset, RFMiDDataset
from stylegan3.utils import load_regression_model, load_single_source_regression_model

def train_model(opts):
    ## start logging
    wandb.init(
        project=opts.wandb_pj_v,
        name=opts.wandb_name,
        dir="/dhc/home/wei-cheng.lai/experiments/wandb/",
        config=locals()
    )
    ## initialize
    wandb_logger = WandbLogger(
        name=opts.wandb_name,
        project=opts.wandb_pj_v
    )

    seed_everything(42)
    BATCH_SIZE = opts.batch_size if torch.cuda.is_available() else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    class_weights = None
    if opts.data_name == "retinal":
        dm = UKBiobankRetinaDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )

        if opts.covariate in ["systolic", "presbyopia", "cataract"]:
            dataset1 = UKBiobankRetinalDataset2D(
                path=os.path.join(opts.data_dir, "source1"),
                mode="train",
                data_name=opts.data_name,
                use_labels=True,
                covariate=opts.covariate,
                xflip=False
            )
            dataset2 = UKBiobankRetinalDataset2D(
                path=os.path.join(opts.data_dir, "source2"),
                mode="train",
                data_name=opts.data_name,
                use_labels=True,
                covariate=opts.covariate,
                xflip=False
            )
            labels1 = dataset1._load_raw_labels()
            labels2 = dataset2._load_raw_labels()
            labels = np.concatenate([labels1, labels2], axis=0)
            class_counts = torch.bincount(torch.tensor(labels.astype("int64")))
            if opts.covariate in ["systolic", "presbyopia"]:
                total_samples = float(len(labels))
                class_weights = total_samples / (2.0 * class_counts)
                class_weights = torch.tensor(class_weights)
            elif opts.covariate == "cataract":
                class_weights = class_counts[0] / class_counts[1]

    elif opts.data_name == "ukb":
        dm = UKBiobankMRIDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
    elif opts.data_name == "mnist-thickness-intensity-slant":
        dm = MorphoMNISTDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
    elif opts.data_name == "adni":
        dm = AdniMRIDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
    elif opts.data_name == "nacc":
        dm = NACCMRIDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
        if opts.covariate == "apoe":
            dataset = NACCMRIDataset2D(
                path=opts.data_dir,
                mode="train",
                data_name=opts.data_name,
                use_labels=True,
                covariate=opts.covariate,
                xflip=False
            )
            labels = dataset._load_raw_labels()
            class_counts = torch.bincount(torch.tensor(labels.astype("int64")))
            total_samples = float(len(labels))
            class_weights = total_samples / (2.0 * class_counts)
            class_weights = torch.tensor(class_weights)

    elif opts.data_name == "eyepacs":
        dm = EyepacsDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
        # if opts.task == "classification": ## binary classification
        #     dataset = KaggleEyepacsDataset(
        #         path=opts.data_dir,
        #         mode="train",
        #         data_name=opts.data_name,
        #         use_labels=True,
        #         xflip=False
        #     )
        #     labels = dataset._load_raw_labels()
        #     class_counts = torch.bincount(torch.tensor(labels.astype("int64")))
        #     class_weights = class_counts[0] / class_counts[1]

    elif opts.data_name == "rfmid":
        dm = RFMIDDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
        if opts.covariate != "Disease_Risk":
            dataset = RFMiDDataset(
                path=opts.data_dir,
                mode="train",
                data_name=opts.data_name,
                use_labels=True,
                covariate=opts.covariate,
                xflip=False
            )
            labels = dataset._load_raw_labels()
            class_counts = torch.bincount(torch.tensor(labels.astype("int64")))
            class_weights = class_counts[0] / class_counts[1]

    elif opts.data_name == "mnist":
        dm = MNISTDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
    elif opts.data_name == "cifar":
        dm = CIFARDataModule(
            data_dir=opts.data_dir,
            data_name=opts.data_name,
            use_labels=True,
            covariate=opts.covariate,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            xflip=False,
        )
    else:
        raise ValueError(f"data_name {opts.data_name} not found")
    
    if opts.task == "regression":
        if opts.data_name in ["adni", "nacc", "eyepacs"]:
            if opts.continue_training:
                model = load_single_source_regression_model(
                    checkpoint_path=opts.regr_model,
                    which_model=opts.which_model,
                    ncls=opts.ncls,
                    task=opts.task,
                    continue_training=True
                )
            else:
                model = MNISTRegressionResnet(
                    ch_in=1 if opts.data_name != "eyepacs" else 3,
                    batch_size=BATCH_SIZE,
                    learning_rate=1e-2,
                    which_model=opts.which_model
                )
        else:
            if opts.continue_training:
                model = load_regression_model(
                    checkpoint_path=opts.regr_model,
                    which_model=opts.which_model,
                    ncls=opts.ncls,
                    task=opts.task,
                    continue_training=True
                )
            else:
                model = RegressionResnet(
                    ch_in=1 if opts.data_name != "retinal" else 3, 
                    batch_size=BATCH_SIZE,
                    learning_rate=1e-2,
                    which_model=opts.which_model
                )
    else:
        if opts.data_name in ["mnist", "cifar", "nacc", "eyepacs", "rfmid"]:
            model = MNISTClassificationResnet(
                ch_in=1 if opts.data_name not in ["eyepacs", "rfmid"] else 3,
                batch_size=BATCH_SIZE,
                learning_rate=1e-2,
                ncls=opts.ncls,
                class_weights=class_weights,
                which_model=opts.which_model
            )
        else:
            model = ClassificationResnet(
                ch_in=1 if opts.data_name != "retinal" else 3,
                batch_size=BATCH_SIZE,
                learning_rate=1e-2,
                ncls=opts.ncls,
                class_weights=class_weights,
                which_model=opts.which_model
            )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=60, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(opts.outdir, "checkpoints"),
            filename="best_model-{val_loss:.2f}",
            save_top_k=2,
            monitor="val_accuracy" if opts.task != "regression" else "val_loss",
            mode="max" if opts.task != "regression" else "min",
        ),
        RichProgressBar()
    ]
    trainer = Trainer(
        max_epochs=300,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm) 
    trainer.test(model, datamodule=dm)



### ---------------------
## click command
@click.command()

## required arguments
@click.option("--outdir", type=str, required=True)
@click.option("--data_dir", type=str, required=True)
@click.option("--data_name", type=str, required=True)
@click.option("--wandb_name",   help='wandb experiment name', metavar='STR', type=str,  default=None, required=False)
@click.option("--wandb_pj_v",   help='wandb project version', metavar='STR', type=str,  default=None, required=False)
@click.option("--batch_size", type=int, default=128)
@click.option("--covariate", type=str, default="thickness")
@click.option("--task", type=str, default="regression") # regression or classification
@click.option("--ncls", type=int, default=None)
@click.option("--which_model", type=str, default="resnet50") # resnet50, resnet101
@click.option("--continue_training", type=bool, default=False)
@click.option("--regr_model", type=str, default=None)


def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()

    c.outdir = opts.outdir
    c.data_name = opts.data_name
    c.data_dir = opts.data_dir
    c.batch_size = opts.batch_size
    c.covariate = opts.covariate
    c.task = opts.task
    c.ncls = opts.ncls
    c.which_model = opts.which_model
    c.continue_training = opts.continue_training
    c.wandb_name = opts.wandb_name
    c.wandb_pj_v = opts.wandb_pj_v

    if c.continue_training:
        assert opts.regr_model is not None
        c.regr_model = opts.regr_model
    
    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.outdir, exist_ok=True)
    with open(os.path.join(c.outdir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    train_model(opts)

## --- execute --- ##
if __name__ == "__main__":
    main()