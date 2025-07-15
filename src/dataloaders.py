import os
from typing import List, Union, Dict

import torch
from torch.utils.data import DataLoader

from src.datasets import ImgFolderDataset, collate_fn, collate_fn_tta
from src.transforms import build_train_transform, build_test_transfrom, \
    build_test_transfrom_tta


def build_train_loader(
    data_roots: List[Union[str, os.PathLike]],
    cat_name_id_dict: Dict[str, int],
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    eigen_vecs: torch.Tensor,
    eigen_vals: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    assert isinstance(data_roots, list)

    transforms = build_train_transform(
        channel_mean, channel_std, eigen_vecs, eigen_vals
    )

    train_set = ImgFolderDataset(
        data_roots, transforms, cat_name_id_dict
    )

    train_loader = DataLoader(
        train_set, batch_size, True, 
        num_workers = num_workers,
        collate_fn = collate_fn, 
        pin_memory = True,
        drop_last = True
    )

    return train_loader

def build_test_loader(
    data_roots: List[Union[str, os.PathLike]],
    cat_name_id_dict: Dict[str, int],
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    assert isinstance(data_roots, list)

    transforms = build_test_transfrom(channel_mean, channel_std)

    test_set = ImgFolderDataset(
        data_roots, transforms, cat_name_id_dict
    )

    test_loader = DataLoader(
        test_set, batch_size, False, 
        num_workers = num_workers,
        collate_fn = collate_fn, 
        pin_memory = True,
        drop_last = False
    )

    return test_loader

def build_test_loader_tta(
    data_roots: List[Union[str, os.PathLike]],
    cat_name_id_dict: Dict[str, int],
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    assert isinstance(data_roots, list)

    transforms = build_test_transfrom_tta(channel_mean, channel_std)

    test_set = ImgFolderDataset(
        data_roots, transforms, cat_name_id_dict
    )

    test_loader = DataLoader(
        test_set, batch_size, False, 
        num_workers = num_workers,
        collate_fn = collate_fn_tta, 
        pin_memory = True,
        drop_last = False
    )

    return test_loader