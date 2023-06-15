## -------------------
## --- Third-Party ---
## -------------------
from operator import index
import os
from matplotlib import use
import seaborn as sns
import numpy as np
import pandas as pd

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
    csv_id_creator(path=path_to_data)
    id_file = "data/ukbb/t1_brain/id_suffix_readfile.csv"
    csv_annotation_creator(csv_file=csv_file,
                           id_file=id_file)