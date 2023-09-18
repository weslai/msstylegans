import os
import click
import json
import wandb
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
# from pytorch_lightning.callbacks.progress import ProgressBarBase
# from torchmetrics import MeanSquaredError
import stylegan3.dnnlib as dnnlib

### --- own --- ###
from eval_regr.model import RegressionResnet
from eval_regr.dataset import MorphoMNISTDataModule, UKBiobankMRIDataModule, UKBiobankRetinaDataModule


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
    else:
        raise ValueError(f"data_name {opts.data_name} not found")
    
    model = RegressionResnet(
        ch_in=1 if opts.data_name != "retinal" else 3, 
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=50, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(opts.outdir, "checkpoints"),
            filename="best_model-{val_loss:.2f}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
        ),
        RichProgressBar()
    ]
    trainer = Trainer(
        max_epochs=250,
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


def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()

    c.outdir = opts.outdir
    c.data_name = opts.data_name
    c.data_dir = opts.data_dir
    c.batch_size = opts.batch_size
    c.covariate = opts.covariate
    c.wandb_name = opts.wandb_name
    c.wandb_pj_v = opts.wandb_pj_v
    
    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.outdir, exist_ok=True)
    with open(os.path.join(c.outdir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    train_model(opts)

## --- execute --- ##
if __name__ == "__main__":
    main()