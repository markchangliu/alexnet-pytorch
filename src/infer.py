import os
from typing import Union, Callable, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_img(
    img_p: Union[str, os.PathLike],
    model: nn.Module,
    transforms: Callable[[np.ndarray], torch.Tensor],
    device: str
) -> Tuple[float, float]:
    assert device.startswith("cuda:")

    model.eval()
    model = model.to(device)

    img = cv2.imread(img_p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(img).to(device)

    pred_logits: torch.Tensor = model(img)
    pred_probs = F.softmax(pred_logits.flatten(), dim = 0)
    pred_prob, pred_cat_id = torch.max(pred_probs, dim = 0)

    pred_prob = pred_prob.item()
    pred_cat_id = pred_cat_id.item()

    return pred_prob, pred_cat_id