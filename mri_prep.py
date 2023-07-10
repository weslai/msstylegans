## -------------------
## --- Third-Party ---
## -------------------
import os
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

PHENO_PTH = "/dhc/projects/ukbiobank/original/phenotypes/ukb49727.csv"
IMG_PTH = "/dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped/"

QCOVARIATES = {
    "age": 21022,
}
CCOVARIATES = {
    "standard": 25000,
    "peripheral_greymatter_norm": 25001,
    "peripheral_greymatter": 25002,
    "ventricular_cerebrospinal_fluid_norm": 25003, ## if we use this
    "ventricular_cerebrospinal_fluid": 25004,
    "grey_matter_norm": 25005,
    "grey_matter": 25006,
    "white_matter_norm": 25007,
    "white_matter": 25008,
    "brain_volume_norm": 25009,
    "brain_volume": 25010,
    "cortex_left": 26552, ## this one first
    "laterals_ventricles_left": 26554,
    "cortex_right": 26583, ## second
    "laterals_ventricles_right": 26585,
}
CCOVARIATE_CODING = {
    "standard": 0,
    "peripheral_greymatter_norm": 1,
    "peripheral_greymatter": 2,
    "ventricular_cerebrospinal_fluid_norm": 3,
    "ventricular_cerebrospinal_fluid": 4,
    "grey_matter_norm": 5,
    "grey_matter": 6,
    "white_matter_norm": 7,
    "white_matter": 8,
    "brain_volume_norm": 9,
    "brain_volume": 10,
    "cortex_left": 11,
    "laterals_ventricles_left": 12,
    "cortex_right": 13,
    "laterals_ventricles_right": 14,
}
NROWS = None

def get_img_indivs(out_file="tmp.csv"):
    pth = Path(IMG_PTH)
    pths = list(pth.glob(f"*_20252_*_*/T1/T1toMNIlin.nii.gz"))
    iids = [int(pth.parts[-3].split("_")[0]) for pth in pths]
    df = pd.DataFrame({"iid": iids, "path": pths}).sort_values(by="path", axis=0)
    # df.drop_duplicates(subset="iid", inplace=True, keep="last")
    df.index = df.iid
    df.drop(columns=["iid"], inplace=True)

    qdf, cdf, cdf2 = get_volumes(iids)
    df = df.join(qdf).join(cdf).join(cdf2)
    df.to_csv(out_file)

    return df

def get_volumes(iids):
    # quantitative:
    sniff = pd.read_csv(PHENO_PTH, nrows=2)

    qcols = ["eid"] + [
        col for col in sniff.columns if is_in_set(col, QCOVARIATES.values())
    ]
    qdf = pd.read_csv(PHENO_PTH, usecols=qcols, index_col=0, nrows=NROWS)
    I = sorted(np.intersect1d(qdf.index, iids))
    qdf = qdf.loc[I]
    
    ccols = ["eid"] + [
        col for col in sniff.columns if is_in_set_vols(col, CCOVARIATES.values())
    ]
    cdf = pd.read_csv(PHENO_PTH, usecols=ccols, index_col=0, nrows=NROWS)
    I = sorted(np.intersect1d(cdf.index, iids))
    cdf = cdf.loc[I]
    ccols2 = ["eid"] + [
        col for col in sniff.columns if is_in_set_vols2(col, CCOVARIATES.values())
    ]
    cdf2 = pd.read_csv(PHENO_PTH, usecols=ccols2, index_col=0, nrows=NROWS)
    I = sorted(np.intersect1d(cdf2.index, iids))
    cdf2 = cdf2.loc[I]
    return qdf, cdf, cdf2

def is_in_set(col, cols):
    return any([col.startswith(f"{c}-0.") for c in cols])
def is_in_set_vols(col, cols):
    return any([col.startswith(f"{c}-2.") for c in cols])
def is_in_set_vols2(col, cols):
    return any([col.startswith(f"{c}-3.") for c in cols])


def csv_extract_columns(csv_file: str, columns: list, outfile: str):
    df = pd.read_csv(csv_file)
    all_col = []
    for index, row in df.iterrows():
        suffix = row.loc["path"].split("/")[-3].split("_")[-2]
        for col in columns:
            if col == 21022:
                age = row.loc[str(col) + "-0.0"]
            else:
                temp = row.loc[str(col) + "-" + suffix + ".0"]
                if col == 25003:
                    ventricle = temp
                elif col == 26552:
                    cortex_left = temp
                elif col == 25005:
                    grey_matter = temp
        col_list = [row.loc["path"], age, ventricle, grey_matter, cortex_left]
        all_col.append(col_list)
    df = pd.DataFrame(all_col, columns=["path", "age", "ventricle", "grey_matter", "cortex_left"])
    df.to_csv(outfile)
    return df

def csv_annotation_creator(csv_file: str, id_file: str):
    eids = []
    suffixes = []
    brain_volumes = []
    ventricle_volumes = []
    intracranial_volumes = []
    sexes = []
    ages = []
    ## load the csv file (id file)
    id_suffix = pd.read_csv(id_file)
    id_needed = id_suffix["id"]
    ls_id = list(id_needed)
    suffix_needed = id_suffix["suffix"]
    print("done")
    del id_suffix
    ## load the csv file (annotation)
    chunksize = 10 ** 4
    ## sex, age, ventricle, brain
    use_columns = ["eid", "31-0.0", "21022-0.0", "25003-2.0", "25003-3.0",
                   "25009-2.0", "25009-3.0", "26521-2.0", "26521-3.0"
                   ]
    annotate = pd.read_csv(csv_file, 
                           chunksize=chunksize,
                           usecols=use_columns,
                           low_memory=False)
    df_annotate = pd.concat(annotate)
    # del annotate
    ## now create the csv file for the factors
    print("yeah, the annotation is loaded now")

    # for chunk in annotate:
    for i, eid in enumerate(df_annotate["eid"]):
        try:
            indices = ls_id.index(eid)
            for idx in [indices]:
                eids.append(id_needed[idx])
                suffixes.append(suffix_needed[idx])
                brain_volumes.append(df_annotate["25009-{}.0".format(suffix_needed[idx])][i])
                ventricle_volumes.append(df_annotate["25003-{}.0".format(suffix_needed[idx])][i])
                intracranial_volumes.append(df_annotate["26521-{}.0".format(suffix_needed[idx])][i])
                sexes.append(df_annotate["31-0.0"][i])
                ages.append(df_annotate["21022-0.0"][i])

        except ValueError:
            pass
    annotation = {"eid": eids,
                  "suffix": suffixes,
                  "age": ages,
                  "sex": sexes,
                  "ventricle_volume": ventricle_volumes,
                  "brain_volume": brain_volumes,
                  "intracranial_volume": intracranial_volumes}
    df_annotation = pd.DataFrame.from_dict(annotation)
    ## save the csv file
    path_to_save = "/dhc/home/wei-cheng.lai/causal-gan/data/ukbb/t1_brain/"
    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)
    path_to_save = os.path.join(path_to_save, "annotation_threevolumes.csv")
    df_annotation.to_csv(path_to_save)
    
def csv_id_creator(path: str, id_im: str = "20252"):
    """
    This is used for generating a csv file, containing a "path" column
    csv_file: the path to the annotation (default: ukb49727.csv)
    path: the path where contains the data (images)
    id_im: the id of data (ex: T1_structural_brain_mri: 20252)
    """
    ids_with_suffixes = []
    ids_only = []
    suffixes_only = []
    file_names = []

    for dir in os.listdir(path=path):
        if id_im in dir:
            temp_path = os.path.join(path, dir)
            if "T1" in os.listdir(temp_path) and "T1_unbiased_brain.nii.gz" in os.listdir(
                os.path.join(temp_path, "T1")):
                id = dir.split("_")[0]
                suffix = dir.split("_")[-2]
                id_path = temp_path + "/T1/T1_unbiased_brain.nii.gz"
                file_names.append(id_path)
                id_with_suffix = id + "_" + suffix
                ids_only.append(id)
                suffixes_only.append(suffix)
                ids_with_suffixes.append(id_with_suffix)
    print("ids has the length of {}".format(len(ids_with_suffixes)))
    id_dict = {"id_suffix": list(ids_with_suffixes), 
               "id": list(ids_only),
               "suffix": list(suffixes_only),
               "path": list(file_names)}
    df = pd.DataFrame.from_dict(id_dict)
    ## save the csv file
    path_to_save = "/dhc/home/wei-cheng.lai/causal-gan/data/ukbb/t1_brain/"
    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)
    path_to_save = os.path.join(path_to_save, "id_suffix_readfile.csv")
    df.to_csv(path_to_save)


def covs_pair_plot(data_path: str, 
    save_path: str,
    covs: list = ["age", "sex", "ventricle_volume", "brain_volume"]
):
    data = pd.read_csv(data_path)
    data_ext = data[covs].dropna()
    
    ## seaborn plot
    grid = sns.PairGrid(data_ext)
    grid.map_upper(sns.scatterplot)
    grid.map_lower(sns.kdeplot)
    grid.map_diag(sns.kdeplot, lw=3, legend=False)
    grid.savefig(save_path + "/pair_plot.png")



if __name__ == "__main__":
    csv_file = "/dhc/projects/ukbiobank/original/phenotypes/ukb49727.csv"
    path_to_data = "/dhc/projects/ukbiobank/original/imaging/brain_mri/T1_structural_brain_mri/unzipped"
    out_file = "/dhc/groups/fglippert/Ukbiobank/imaging/brain_mri/multisources/volumes.csv"
    # csv_id_creator(path=path_to_data)
    # id_file = "data/ukbb/t1_brain/id_suffix_readfile.csv"
    # csv_annotation_creator(csv_file=csv_file,
    #                        id_file=id_file)
    cdf = get_volumes()
    # df = df.join(qdf).join(cdf)
    cdf.to_csv(out_file)