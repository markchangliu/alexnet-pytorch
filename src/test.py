import json
import os
from typing import Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import AlexNet, loss_func, eval_func
from src.dataloaders import build_test_loader_tta, build_test_loader
from src.utils import get_cat_name_id_dict


@torch.no_grad()
def test_model_tta(
    model: nn.Module,
    test_loader: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> Union[float, float]:
    if model.training:
        model.eval() 

    curr_loop = 1
    loss = 0
    acc = 0
    batch_size = test_loader.batch_size

    for imgs, gt_cat_ids in test_loader:
        gt_cat_ids = torch.chunk(gt_cat_ids, batch_size)
        gt_cat_ids = [g[0] for g in gt_cat_ids]
        gt_cat_ids = torch.stack(gt_cat_ids).to(device, non_blocking = True)

        imgs: torch.Tensor = imgs.to(device, non_blocking = True)
        pred_logits: torch.Tensor = model(imgs)

        pred_logits = torch.chunk(pred_logits, batch_size)
        pred_logits = [torch.mean(p, dim = 0, keepdim = True) for p in pred_logits]
        pred_logits = torch.concat(pred_logits)

        curr_loss = loss_func(gt_cat_ids, pred_logits).item()
        curr_acc = eval_func(gt_cat_ids, pred_logits).item()

        loss = loss + (curr_loss - loss) / curr_loop
        acc = acc + (curr_acc - acc) / curr_loop

        curr_loop += 1
    
    return loss, acc

@torch.no_grad()
def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> Union[float, float]:
    if model.training:
        model.eval() 

    curr_loop = 1
    loss = 0
    acc = 0

    for imgs, gt_cat_ids in test_loader:
        gt_cat_ids: torch.Tensor = gt_cat_ids.to(device, non_blocking = True)
        imgs: torch.Tensor = imgs.to(device, non_blocking = True)
        pred_logits: torch.Tensor = model(imgs)

        curr_loss = loss_func(gt_cat_ids, pred_logits).item()
        curr_acc = eval_func(gt_cat_ids, pred_logits).item()

        loss = loss + (curr_loss - loss) / curr_loop
        acc = acc + (curr_acc - acc) / curr_loop

        curr_loop += 1
    
    return loss, acc

def run_test(
    test_cfg_p: Union[str, os.PathLike],
    tta_flag: bool
) -> None:
    with open(test_cfg_p, "r") as f:
        cfg = json.load(f)
    
    runtime_cfg = cfg["runtimes"]
    device = runtime_cfg["device"]

    model_cfg = cfg["model"]
    model = AlexNet(model_cfg["num_classes"])
    state_dict = torch.load(model_cfg["ckpt_p"])
    model.load_state_dict(state_dict["model"])
    model = model.to(device)

    dataset_cfg = cfg["dataset"]
    channel_mean = torch.as_tensor(dataset_cfg["channel_mean"])
    channel_std = torch.as_tensor(dataset_cfg["channel_std"])
    test_data_root = dataset_cfg["test_data_root"]
    cat_name_id_dict = get_cat_name_id_dict(dataset_cfg["class_list"])
    
    dataloader_cfg = cfg["test_loader"]
    
    if tta_flag:
        test_loader = build_test_loader_tta(
            [test_data_root],
            cat_name_id_dict,
            channel_mean,
            channel_std,
            dataloader_cfg["batch_size"],
            dataloader_cfg["num_workers"]
        )
    else:
        test_loader = build_test_loader(
            [test_data_root],
            cat_name_id_dict,
            channel_mean,
            channel_std,
            dataloader_cfg["batch_size"],
            dataloader_cfg["num_workers"]
        )

    loss, acc = test_model(
        model, test_loader, loss_func, eval_func, device
    )

    print(f"loss: {loss:.3f}")
    print(f"accuracy: {acc:.3f}")