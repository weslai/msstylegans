# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tool for creating ZIP/PNG based datasets."""

import functools
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.util import crop
#----------------------------------------------------------------------------
def check_direction(img):
    ## crop the image
    img_crop = crop(img, ((50, 50), (50, 50), (0, 0)))
    intensity_middle = img_crop[img_crop.shape[0] //2, :, 0]
    max_intensity = np.max(intensity_middle)
    idx_max = np.where(intensity_middle == max_intensity)[0]
    if idx_max[0] < img_crop.shape[1] // 2:
        direction = "left"
    else:
        direction = "right"
    return direction
#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()
#----------------------------------------------------------------------------
def open_rfmid_retinal(
    annotation_path: str,
    max_images: Optional[int],
    which_dataset: str
):
    """
    use https://www.mdpi.com/2306-5729/6/2/14 as the dataset
    """
    assert which_dataset in ['train', 'val', 'test']
    ## load data
    assert annotation_path.split(".")[-1] == "csv"
    df = pd.read_csv(annotation_path)
    df = df.dropna()
    ## add paths 
    temp_path = os.path.join("/", *annotation_path.split("/")[:-1])
    if which_dataset == "train":
        temp_path = os.path.join(temp_path, "Training_cropped/")
    elif which_dataset == "val":
        temp_path = os.path.join(temp_path, "Validation_cropped/")
    else:
        temp_path = os.path.join(temp_path, "Test_cropped/")
    df["path"] = temp_path + df["ID"].astype("string") + ".jpg"

    input_images = list(df["path"])
    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            ## load image
            img = np.array(PIL.Image.open(fname))
            items = dict(
                (key, value)
                    for key, value in df.loc[df["path"] == fname].to_dict().items()
            )
            yield dict(img=img, label=items)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()
#----------------------------------------------------------------------------
def open_kaggle_eyepacs(
    annotation_path: str,
    max_images: Optional[int],
    which_dataset: str,
    covs: list = ["image", "level"],
    which_side: str = "left"
):
    """
    use https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview
    as the dataset
    """
    assert which_dataset in ['train', 'val', 'test']
    ## load data
    if annotation_path.split(".")[-1] == "zip":
        labels = zipfile.ZipFile(annotation_path, 'r')
        df = pd.read_csv(labels.open(labels.filelist[0]))
    else:
        df = pd.read_csv(annotation_path)
    df_ext = df[covs].dropna()
    ## pick the side
    split_df = df_ext["image"].str.split("_", n = 1, expand = True)
    df_side = df_ext[split_df[1] == which_side]
    ## add paths 
    if which_dataset in ["train", "val"]:
        temp_path = os.path.join("/", *annotation_path.split("/")[:-1])
        df_side["path"] =  temp_path + "/train/" + df_side["image"] + ".jpeg"
    else:
        temp_path = "/dhc/dsets/diabetic_retinopathy"
        df_side["path"] =  temp_path + "/test/" + df_side["image"] + ".jpeg"
    ## separate train/val/test sets
    if which_dataset in ["train", "val"]:
        trainset, valset = train_test_split(df_side, test_size=0.3, random_state=42, shuffle=True)
        if which_dataset == "train":
            dataset = trainset
        elif which_dataset == "val":
            dataset = valset
    else:
        # trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42, shuffle=True)
        dataset = df_side

    ## trainset/testset
    input_imgs = list(dataset["path"])
    max_idx = maybe_min(len(input_imgs), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_imgs):
            ## load image
            img = np.array(PIL.Image.open(fname))
            items = dict(
                (key, value)
                    for key, value in dataset.loc[dataset["path"] == fname][covs[1:]].to_dict().items()
            )
            yield dict(img=img, label=items)
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()
#----------------------------------------------------------------------------
def open_nacc(
    annotation_path: str,
    max_images: Optional[int],
    which_dataset: str,
    covs: list,
    normalize_img: bool = True
): ## brain MRI
    import nibabel as nib
    assert which_dataset in ['train', 'val', 'test']
    ## load data 
    df_3T = pd.read_csv(annotation_path)
    df_3T_ext = df_3T[covs].dropna()
    
    trainset, testset = train_test_split(df_3T_ext, test_size=0.15, random_state=42, shuffle=True)
    trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42, shuffle=True)
    
    if which_dataset == "train":
        dataset = trainset
    elif which_dataset == "val":
        dataset = valset
    elif which_dataset == "test":
        dataset = testset

    input_imgs = list(dataset["MNIlin_filepath"])
    max_idx = maybe_min(len(input_imgs), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_imgs):
            ## load image
            img = nib.load(fname)
            newimg = img.get_fdata()
            ## middle slice
            newimg = np.take(
                newimg,
                newimg.shape[1] // 2,
                1
            )
            ## rotate images
            newimg = np.rot90(newimg, k=1, axes=(0, 1))
            if normalize_img:
                newimg = (newimg - newimg.min()) / (newimg.max() - newimg.min())
                newimg = (255 * newimg).astype(np.uint8)
                # img = (255 * img).round().astype(np.uint8)
            items = dict(
                (key, value)
                    for key, value in dataset.loc[dataset["MNIlin_filepath"] == fname][covs[1:]].to_dict().items()
            )
            yield dict(img=newimg, label=items, org_img=img)
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()
#----------------------------------------------------------------------------
def open_adni(
    annotation_path: str,
    max_images: Optional[int],
    which_dataset: str,
    normalize_img: bool = True,
    covs: list = [
        "filepath_MNIlin",
        "Age", 
        "Sex",
        "CDGLOBAL",
        "left_lateral_ventricle",
        "right_lateral_ventricle",
        "left_cerebral_cortex",
        "right_cerebral_cortex",
        "left_hippocampus",
        "right_hippocampus"
    ]
):
    import nibabel as nib
    assert which_dataset in ['train', 'val', 'test']
    ## load data 
    df_3T = pd.read_csv(annotation_path)
    df_3T_ext = df_3T[covs].dropna()
    
    ## separate train/val/test sets
    ########### Downstream project ##########
    #### delete the replicated data
    # test_annotation = "/dhc/cold/groups/syreal/adni_test_annotation_downstream.csv"
    # df_test = pd.read_csv(test_annotation)
    # df_test = df_test[covs].dropna()
    # test_filepath = df_test[covs[0]]
    
    trainset, testset = train_test_split(df_3T_ext, test_size=0.20, random_state=42, shuffle=True)
    trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42, shuffle=True)

    ## here delete the replicated data from the trainset and concatenate on the valset
    # idxs = []
    # for i in range(len(test_filepath)):
    #     idx = trainset[covs[0]][trainset[covs[0]] == test_filepath.iloc[i]].index
    #     if len(idx) != 0:
    #         idxs.append(idx[0])
    # valset = pd.concat([valset, trainset.loc[idxs]])
    # trainset = trainset.drop(index=idxs)
    
    if which_dataset == "train":
        dataset = trainset
    elif which_dataset == "val":
        dataset = valset
    elif which_dataset == "test":
        dataset = testset

    input_imgs = list(dataset["filepath_MNIlin"])
    max_idx = maybe_min(len(input_imgs), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_imgs):
            ## load image
            img = nib.load(fname)
            newimg = img.get_fdata()
            ## middle slice
            newimg = np.take(
                newimg,
                newimg.shape[1] // 2,
                1
            )
            ## rotate images
            newimg = np.rot90(newimg, k=1, axes=(0, 1))
            if normalize_img:
                newimg = (newimg - newimg.min()) / (newimg.max() - newimg.min())
                newimg = (255 * newimg).astype(np.uint8)
                # img = (255 * img).round().astype(np.uint8)
            items = dict(
                (key, value)
                    for key, value in dataset.loc[dataset["filepath_MNIlin"] == fname][covs[1:]].to_dict().items()
            )
            yield dict(img=newimg, label=items, org_img=img)
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()
#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        # img = img.resize((ww, hh), PIL.Image.LANCZOS)
        img = img.resize((ww, hh))
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img)
        # img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, 
                 annotation_path: str = None,
                 which_dataset: str = None, max_images: Optional[int]):
    if os.path.isdir(source):
        ## retinal (discease dataset RFMid)
        if source.rstrip('/').endswith("RFMiD"):
            return open_rfmid_retinal(
                annotation_path=annotation_path,
                max_images=max_images,
                which_dataset=which_dataset
            )
        ## retinal (kaggle eyepacs dataset)
        elif source.rstrip('/').endswith("diabetic_retinopathy"):
            return open_kaggle_eyepacs(
                annotation_path=annotation_path,
                max_images=max_images,
                which_dataset=which_dataset,
                covs=["image", "level"],
                which_side="left"
            )
        ## adni
        elif source.rstrip('/').endswith("adni_t1_mprage"):
            return open_adni(
                annotation_path="/dhc/groups/fglippert/adni_t1_mprage/T1_3T_coronal_slice/T1_3T_coronal_mni_linear_hippo_resolution256/adni_linear_annotation.csv",
                max_images=max_images,
                which_dataset=which_dataset,
                covs=["filepath_MNIlin", 
                    "Age", 
                    #"Sex",
                    "CDGLOBAL"]
                    # "left_lateral_ventricle",
                    # "right_lateral_ventricle",
                    # "left_cerebral_cortex",
                    # "right_cerebral_cortex",
                    # "left_hippocampus",
                    # "right_hippocampus"]
            )
        ## nacc
        elif source.rstrip('/').endswith("NACC"):
            return open_nacc(
                annotation_path=annotation_path,
                max_images=max_images,
                which_dataset=which_dataset,
                covs=["MNIlin_filepath",
                    "Age", "CDGLOBAL", "MMSE", "Sex", "Apoe4"
                ]
            )
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--dataset_name', help='Name of the dataset', required=True, default='adni')
@click.option('--annotation_path', help='Path to the annotation file', default=None)
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--which_dataset', help='create data subset', required=True, default="train", type=click.Choice(['train', 'test', 'val']))
@click.option('--transform', help='Input crop/resize mode', default=None)
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    dataset_name: str,
    max_images: Optional[int],
    which_dataset: str,
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    annotation_path: str = None
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """
    import nibabel as nib
    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')
    num_files, input_iter = open_dataset(source=source,
        annotation_path=annotation_path,
        max_images=max_images, which_dataset=which_dataset
    )
    os.makedirs(dest, exist_ok=True)
    if which_dataset == "train":
        data_dir = dest + '/trainset'
    elif which_dataset == "val":
        data_dir = dest + '/valset'
    elif which_dataset == "test":
        data_dir = dest + '/testset'
    else:
        raise ValueError("which_dataset must be one of train, val, test")
    
    archive_root_dir, save_bytes, close_dest = open_dest(data_dir)
    
    if resolution is None: resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        if dataset_name in ["adni", "ukb", "nacc"]:
            archive_fname = f'{idx_str[:5]}/img{idx_str}.nii.gz'
        elif dataset_name in ["retinal", "eyepacs", "rfmid"]:
            archive_fname = f'{idx_str[:5]}/img{idx_str}.jpg'
        else:
            archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image['img'])
        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))
        
        if dataset_name in ["eyepacs"]: ## check the inverted (direction of the image)
            direction = check_direction(img)
            if direction == "right":
                img = np.fliplr(img)
                img = np.flipud(img)
        # Save the image as an uncompressed PNG.
        if dataset_name in ["adni", "ukb", "nacc"]:
            save_path = os.path.join(archive_root_dir, archive_fname)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img = img.reshape(*img.shape, 1)
            newimg = nib.Nifti1Image(img, image["org_img"].affine)
            nib.save(newimg, save_path)
        else:
            img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
            image_bits = io.BytesIO()
            if dataset_name in ["retinal", "eyepacs", "rfmid"]:
                img.save(image_bits, format='jpeg', compress_level=0, optimize=False)
            else:
                img.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
