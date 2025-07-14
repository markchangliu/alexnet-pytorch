from typing import List, Union, Dict

import torch.nn as nn
from torch.optim import SGD


def build_sgd(
    model_params: List[Union[nn.Parameter], Dict[str, nn.Parameter]],
    lr_gloabl: float,
    momentum_global: float,
    weight_decay_global: float
) -> SGD:
    """
    example
    ```
    # gloabl
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)

    # per layer
    optim.SGD([
            {'params': model.base.parameters(), 'lr': 1e-2},
            {'params': model.classifier.parameters()}
        ], lr=1e-3, momentum=0.9)
    ```
    """
    sgd = SGD(
        model_params, lr_gloabl, momentum_global, 
        weight_decay = weight_decay_global
    )
    return sgd

def get_lr(sgd: SGD) -> List[float]:
    lrs = []
    
    for group in sgd.param_groups:
        lrs.append(group["lr"])
    
    return lrs

def adjust_lr(
    sgd: SGD,
    gamma: float
) -> None:
    for group in sgd.param_groups:
        group["lr"] = group["lr"] * gamma

def set_lr(
    sgd: SGD,
    lrs: List[float]
) -> None:
    for lr, group in zip(lrs, sgd.param_groups):
        group["lr"] = lr