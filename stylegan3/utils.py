### -------------------
### --- Third-Party ---
### -------------------
import os
import pandas as pd
import numpy as np
import random
import seaborn as sns

## pairplot
def plot_pairplots(df, save_path: str = None):
    """
        df: (pd.DataFrame) with columns names (add type as a column)
    """
    # grid = sns.PairGrid(data=df, hue="type")
    grid = sns.pairplot(data=df, hue="type", diag_kind="hist", size=2)
    grid.savefig(save_path)
    grid.savefig(save_path.replace(".png", ".pdf"))
    return grid


### --- UKB Data preparation ---
### --- Datasets split ---
def annotations_split(
    annotation_file,
    save_path: str = None
):
    """
    This is for multi-source datasets preparation
    Here first for ukb, 
    we create in one source and split it into two sources
    the same id should be only in a single source
    """
    # read the annotation file
    df = pd.read_csv(annotation_file)
    # get the unique ids
    filepaths = df['filepath_MNIlin']
    filepaths_dict = {} ## id: [filepaths]
    ids = []
    for filepath in filepaths:
        id = filepath.split('/')[-3].split('_')[0]
        if id not in ids:
            ids.append(id)
            filepaths_dict[id] = [filepath]
        else:
            filepaths_dict[id].append(filepath)
    assert len(ids) == len(filepaths_dict)
    # shuffle the ids
    random.shuffle(ids)
    first_ids_list = ids[:len(ids)//2]
    # split the annotation file
    first_df, second_df = None, None
    for key, value in filepaths_dict.items(): ## key is id, value is filepaths with lists
        if key in first_ids_list:
            if first_df is None:
                first_df = df[df['filepath_MNIlin'].isin(value)]
            else:
                first_df = pd.concat([first_df, df[df['filepath_MNIlin'].isin(value)]], ignore_index=True)
        else:
            if second_df is None:
                second_df = df[df['filepath_MNIlin'].isin(value)]
            else:
                second_df = pd.concat([second_df, df[df['filepath_MNIlin'].isin(value)]], ignore_index=True)
    # save the annotation file
    if save_path is not None:
        first_df.to_csv(os.path.join(save_path, 'ukb_linear_freesurfer_first_annotation.csv'), index=False)
        second_df.to_csv(os.path.join(save_path, 'ukb_linear_freesurfer_second_annotation.csv'), index=False)
    return first_df, second_df