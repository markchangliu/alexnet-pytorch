import os
from typing import List, Union, Dict, Tuple, Callable

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


def collate_fn_train(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args
    - `batch`: `List[tuple]`, `[(imgs, cat_ids), ...]`
        - `img`: `torch.Tensor`, shape `(1, 3, h, w)`
        - `cat_id`: `int`
    
    Returns
    - `batch_imgs`: `torch.Tensor`, shape `(b, 3, h, w)`
    - `batch_cat_ids`: `torch.Tensor`, shape `(b, )`
    """
    batch_imgs = [d[0] for d in batch]
    batch_cat_ids = [d[1] for d in batch]
    batch_imgs = torch.concat(batch_imgs, dim = 0)
    batch_cat_ids = torch.as_tensor(batch_cat_ids, dtype = torch.int64)

    return batch_imgs, batch_cat_ids

def collate_fn_test(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args
    - `batch`: `List[tuple]`, `[(imgs, cat_id)]`
        - `imgs`: `torch.Tensor`, shape `(10, 3, h, w)`
        - `cat_id`: `int`
    
    Returns
    - `batch_imgs`: `torch.Tensor`, shape `(10 * b, 3, h, w)`
    - `batch_cat_ids`: `torch.Tensor`, shape `(10 * b, )`
    """
    batch_imgs = [d[0] for d in batch]
    batch_imgs = torch.concat(batch_imgs, dim = 0)

    batch_cat_ids = [d[1] for d in batch]
    batch_cat_ids = batch_cat_ids * 2 # itself and flip
    batch_cat_ids = batch_cat_ids * 5 # 5 crops
    batch_cat_ids = torch.as_tensor(batch_cat_ids, dtype = torch.int64)

    assert batch_imgs.shape[0] == batch_cat_ids.shape[0]

    return batch_imgs, batch_cat_ids


class ImgFolderDataset(Dataset):
    def __init__(
        self,
        data_roots: List[Union[str, os.PathLike]],
        transfroms: Callable[[np.ndarray, int], Tuple[torch.Tensor, torch.Tensor]],
        cat_name_id_dict: Dict[str, int]
    ) -> None:
        assert isinstance(data_roots, list)

        self.cat_name_id_dict = cat_name_id_dict
        self.cat_id_name_dict = {v: k for k, v in cat_name_id_dict.items()}
        self.transforms = transfroms

        dirnames = list(cat_name_id_dict.keys())
        dirnames.sort()

        self.data = []

        for data_root in data_roots:
            for dirname in dirnames:
                data_dir = os.path.join(data_root, dirname)
                img_dir = os.path.join(data_dir, "images")

                filenames = os.listdir(img_dir)
                filenames.sort()

                cat_id = self.cat_name_id_dict[dirname]

                for filename in filenames:
                    if not filename.endswith(".JPEG"):
                        continue

                    img_p = os.path.join(img_dir, filename)
                    self.data.append((img_p, cat_id))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_p, cat_id = self.data[idx]
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        return img, cat_id
    
