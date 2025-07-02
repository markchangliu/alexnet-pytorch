import os
from typing import List, Union, Dict, Tuple

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose


class ImgFolderDataset(Dataset):
    def __init__(
        self,
        data_roots: List[Union[str, os.PathLike]],
        transfrom: Compose,
        cat_name_id_dict: Dict[str, int]
    ) -> None:
        assert isinstance(data_roots, list)

        self.cat_name_id_dict = cat_name_id_dict
        self.cat_id_name_dict = {v: k for k, v in cat_name_id_dict.items()}
        self.transform = transfrom

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
        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img, dtype = torch.float32)
        img = self.transform(img)
        return img, cat_id