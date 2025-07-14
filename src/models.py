import os
from typing import Union, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(
        self, 
        num_classes: int
    ) -> None:
        super(AlexNet, self).__init__()
        self.lrn = nn.LocalResponseNorm(5, 1e-4, 0.75, 2)

        self.c1 = nn.Conv2d(3, 96, 11, 4, 2)
        self.pool1 = nn.MaxPool2d(3, 2, 0)

        self.c2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2, 0)

        self.c3 = nn.Conv2d(256, 384, 3, 1, 1)

        self.c4 = nn.Conv2d(384, 384, 3, 1, 1)

        self.c5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(3, 2, 0)

        self.fc6 = nn.Linear(9216, 4096)
        self.dropout6 = nn.Dropout()

        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout()

        self.fc8 = nn.Linear(4096, num_classes)
    
    def forward(
        self, 
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(n, 3, 224, 224)`
        
        Returns
        - `logits`: `torch.Tensor`, shape `(n, num_classes)`
        """
        y = self.c1(imgs)
        y = self.lrn(y)
        y = F.relu(y)
        y = self.pool1(y)

        y = self.c2(y)
        y = self.lrn(y)
        y = F.relu(y)
        y = self.pool2(y)

        y = self.c3(y)
        y = F.relu(y)

        y = self.c4(y)
        y = F.relu(y)

        y = self.c5(y)
        y = F.relu(y)
        y = self.pool5(y)

        y = torch.flatten(y, start_dim = 1, end_dim = -1)

        y = self.fc6(y)
        y = self.dropout6(y)

        y = self.fc7(y)
        y = self.dropout7(y)

        y = self.fc8(y)

        return y
    
def loss_func(
    gt_cat_ids: torch.Tensor,
    pred_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Args
    - `gt_cat_ids`: `torch.Tensor`, shape `(n, )`
    - `pred_logits`: `torch.Tensor`, shape `(n, c)`

    Returns
    - `loss`: `torch.Tensor`, shape `(1, )`
    """
    loss = F.cross_entropy(pred_logits, gt_cat_ids)
    return loss

@torch.no_grad()
def eval_func(
    gt_cat_ids: torch.Tensor,
    pred_logits: torch.Tensor
) -> torch.Tensor:
    """
    Args
    - `gt_cat_ids`: `torch.Tensor`, shape `(n, )`
    - `pred_logits`: `torch.Tensor`, shape `(n, c)`

    Returns
    - `acc`: `torch.Tensor`, shape `(1, )`
    """
    _, pred_cat_ids = torch.max(pred_logits, dim = 1)
    num_corrects = torch.sum(pred_cat_ids == gt_cat_ids)
    num_samples = len(gt_cat_ids)
    acc = num_corrects / num_samples
    return acc