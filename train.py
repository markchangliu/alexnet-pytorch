import os
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loggers import build_logger
from optimizers import adjust_lr_multiplier, set_lr, get_lr


@torch.no_grad()
def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> Union[float, float]:
    model.eval()

    curr_batch = 0
    curr_loss_val = 0
    curr_acc_val = 0

    for imgs, gt_cat_ids in test_loader:
        imgs: torch.Tensor = imgs.to(device)
        gt_cat_ids: torch.Tensor = gt_cat_ids.to(device)


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    export_dir: Union[str, os.PathLike],
    num_epoches: int,
    print_iter_period: int,
    eval_save_epoch_period: int,
    warm_up_epoch_period: int,
    warm_up_lr_multiplier: float,
    adjust_lr_multiplier: float
) -> None:
    assert device.startswith("cuda")
    
    log_file_p = os.path.join(export_dir, "log_train.txt")
    ckpt_dir = os.path.join(export_dir, "ckpts")
    event_dir = os.path.join(export_dir, "events")
    os.makedirs(ckpt_dir, exist_ok = True)
    os.makedirs(event_dir, exist_ok = True)

    logger = build_logger(log_file_p)
    writer = SummaryWriter(event_dir)

    num_iters_per_epoch = len(train_loader.dataset)
    num_iters = num_epoches * num_iters_per_epoch
    batch_size = train_loader.batch_size

    model.train()

    curr_epoch = 0
    curr_iter = 0
    init_lrs = get_lr(optimizer)
    curr_lrs = [lr * warm_up_lr_multiplier for lr in init_lrs]
    set_lr(optimizer, curr_lrs)

    for i in range(num_epoches):

        curr_epoch += 1

        for j, (imgs, gt_cat_ids) in enumerate(train_loader):
            imgs: torch.Tensor = imgs.to(device, non_blocking = True)
            gt_cat_ids: torch.Tensor = gt_cat_ids.to(device, non_blocking = True)
            pred_logits: torch.Tensor = model(imgs)

            loss = loss_func(gt_cat_ids, pred_logits)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            curr_iter += batch_size

            if curr_iter % print_iter_period == 0:
                metric_val = eval_func(gt_cat_ids, pred_logits).item()
                loss_val = loss.item()

                msg = f"[Train] epoch {curr_epoch}/{num_epoches}, "
                msg += f"iter {curr_iter}/{num_iters}, "
                msg += f"loss {loss_val:.3f}, accuracy {metric_val:.3f}, lr ["

                for lr in curr_lrs:
                    msg += f"{lr:.3f}, "
                
                msg += "]"
            
                logger.info(msg)
                writer.add_scalar("Train/loss", loss_val)
                writer.add_scalar("Train/accuracy", metric_val)

                for _, lr in curr_lrs:
                    writer.add_scalar(f"Train/lr{_}", lr)
            
        if curr_epoch > warm_up_epoch_period:
            set_lr(optimizer, init_lrs)
        
        if curr_epoch % eval_save_epoch_period == 0:
