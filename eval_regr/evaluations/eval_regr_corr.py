import sys, os
sys.path.append("../")
sys.path.append("/dhc/home/wei-cheng.lai/projects/msstylegans/stylegan3")
import click
import json
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import r2_score
from torchmetrics import Accuracy
### --- Own --- ###
import stylegan3.dnnlib as dnnlib
from stylegan3.utils import load_regression_model, load_single_source_regression_model
from eval_regr.dataset import UKBiobankRetinaDataModule, UKBiobankMRIDataModule, MorphoMNISTDataModule
from eval_regr.dataset import AdniMRIDataModule, NACCMRIDataModule
from eval_regr.dataset import EyepacsDataModule, RFMIDDataModule

def model_predict(opts):
    # Load the prediction model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opts.data_name in ["nacc", "adni", "eyepacs", "rfmid"]:
        regr_model = load_single_source_regression_model(opts.regr_model, opts.which_model, 
                                                         opts.ncls, opts.task).to(device)
    else:
        regr_model = load_regression_model(opts.regr_model, opts.which_model, opts.ncls, opts.task).to(device)

    # Load the data using a dataloader
    if opts.data_name == "retinal":
        data = UKBiobankRetinaDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size, covariate = opts.covariate,
            use_labels = True
        )
    elif opts.data_name == "ukb":
        data = UKBiobankMRIDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size, covariate = opts.covariate,
            use_labels = True
        )
    elif opts.data_name == "mnist-thickness-intensity-slant":
        data = MorphoMNISTDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size, covariate = opts.covariate,
            use_labels = True
        )
    elif opts.data_name == "adni":
        data = AdniMRIDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size, covariate = opts.covariate,
            use_labels = True
        )
    elif opts.data_name == "nacc":
        data = NACCMRIDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size, covariate = opts.covariate,
            use_labels = True
        )
    elif opts.data_name == "eyepacs":
        data = EyepacsDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size,
            use_labels = True
        )
    elif opts.data_name == "rfmid":
        data = RFMIDDataModule(
            opts.data_dir, data_name = opts.data_name, batch_size = opts.batch_size, covariate= opts.covariate,
            use_labels = True
        )
    else:
        raise NotImplementedError("Unknown dataset: {}".format(opts.data_name))
    data.setup("test")
    data_loader = data.test_dataloader()

    # Make a prediction
    if opts.data_name in ["adni", "nacc", "eyepacs", "rfmid"]:
        outputs_list = []
        gt_list = []
        with torch.no_grad():
            for batch in data_loader:
                imgs, labels = batch[0], batch[1]
                if opts.task == "regression":
                    outputs = regr_model(imgs.float().to(device))
                else:
                    if opts.ncls == 2:
                        outputs = torch.sigmoid(regr_model(imgs.float().to(device)))
                        outputs = torch.bernoulli(outputs)
                    else:
                        outputs = torch.softmax(regr_model(imgs.float().to(device)), dim=1)
                        outputs = torch.argmax(outputs, dim=1)
                if labels.dim() != 2:
                    labels = labels.unsqueeze(1)
                if outputs.dim() != 2:
                    outputs = outputs.unsqueeze(1)
                gt_list.append(labels)
                outputs_list.append(outputs.cpu().detach())
        outputs_tensor = torch.cat(outputs_list)
        gt_tensor = torch.cat(gt_list)
    else:
        outputs_s1_list = []
        outputs_s2_list = []
        gt_s1_list = []
        gt_s2_list = []
        with torch.no_grad():
            for batch in data_loader:
                img_s1, labels_s1 = batch[0][0], batch[0][1]
                img_s2, labels_s2 = batch[1][0], batch[1][1]
                if opts.task == "regression":
                    outputs_s1 = regr_model(img_s1.float().to(device))
                    outputs_s2 = regr_model(img_s2.float().to(device))
                else:
                    if opts.ncls == 2:
                        outputs_s1 = torch.sigmoid(regr_model(img_s1.float().to(device)))
                        outputs_s2 = torch.sigmoid(regr_model(img_s2.float().to(device)))
                        outputs_s1 = torch.round(outputs_s1)
                        outputs_s2 = torch.round(outputs_s2)
                    else:
                        outputs_s1 = torch.log_softmax(regr_model(img_s1.float().to(device)), dim=1)
                        outputs_s2 = torch.log_softmax(regr_model(img_s2.float().to(device)), dim=1)
                        outputs_s1 = torch.argmax(outputs_s1, dim=1)
                        outputs_s2 = torch.argmax(outputs_s2, dim=1)
                if labels_s1.dim() != 2:
                    labels_s1 = labels_s1.unsqueeze(1)
                if labels_s2.dim() != 2:
                    labels_s2 = labels_s2.unsqueeze(1)
                if outputs_s1.dim() != 2:
                    outputs_s1 = outputs_s1.unsqueeze(1)
                if outputs_s2.dim() != 2:
                    outputs_s2 = outputs_s2.unsqueeze(1)
                gt_s1_list.append(labels_s1)
                outputs_s1_list.append(outputs_s1.cpu().detach())
                gt_s2_list.append(labels_s2)
                outputs_s2_list.append(outputs_s2.cpu().detach())

        # Concatenate the outputs into a single tensor
        outputs_tensor = torch.cat(outputs_s1_list + outputs_s2_list)
        gt_tensor = torch.cat(gt_s1_list + gt_s2_list)

    # Convert the tensor to a pandas dataframe
    df_outputs = pd.DataFrame(outputs_tensor.numpy(), columns=["prediction"])
    gt_labels = pd.DataFrame(gt_tensor.numpy(), columns=["label"])

    df_outputs_labels = pd.concat([df_outputs, gt_labels], axis=1)
    # save the labels into a pandas dataframe
    filename = "regr_prediction_{}.csv".format(opts.covariate)
    df_outputs_labels.to_csv(os.path.join(opts.outdir, filename), 
                      index=False)

    # compute the accuracy of the prediction
    if opts.task != "regression":
        accuracy = Accuracy(task="binary" if opts.ncls == 2 else "multiclass", num_classes=opts.ncls)
        acc = accuracy(torch.tensor(df_outputs.values.flatten()), torch.tensor(gt_labels.values.flatten()))


    # Compute the Pearson correlation between the outputs and labels
    corr, _ = stats.pearsonr(df_outputs.values.flatten(), gt_labels.values.flatten())
    print(f"Pearson correlation: {corr}")

    r2 = r2_score(df_outputs.values.flatten(), gt_labels.values.flatten())
    print(f"R2 score: {r2}")
    if opts.task == "regression":
        corr_pd = pd.DataFrame({"Pearson correlation": [corr], 
                                "R2 score": [r2]})
    else:
        corr_pd = pd.DataFrame({"Pearson correlation": [corr], 
                                "R2 score": [r2],
                                "accuracy": [acc]})
    corr_pd.to_csv(os.path.join(opts.outdir, f"regr_pearson_corr_{opts.covariate}.csv"), index=False)

@click.command()
## required arguments
@click.option("--outdir", type=str, required=True)
@click.option("--data_dir", type=str, required=True)
@click.option("--data_name", type=str, required=True)
@click.option("--batch_size", type=int, default=128)
@click.option("--covariate", type=str, default="thickness")
@click.option("--regr_model", type=str, default="path/to/regression/model")
@click.option("--task", type=str, default="regression")
@click.option("--ncls", type=int, default=None)
@click.option("--which_model", type=str, default="resnet50")

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()

    c.outdir = opts.outdir
    c.data_name = opts.data_name
    c.data_dir = opts.data_dir
    c.batch_size = opts.batch_size
    c.covariate = opts.covariate
    c.regr_model = opts.regr_model
    c.task = opts.task
    c.ncls = opts.ncls
    c.which_model = opts.which_model
    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.outdir, exist_ok=True)
    with open(os.path.join(c.outdir, 'regr_predict_config.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    model_predict(opts)

## --- execute --- ##
if __name__ == "__main__":
    main()