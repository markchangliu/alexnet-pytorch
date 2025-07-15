import os
import json
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataloaders import build_train_loader, build_test_loader
from src.loggers import build_logger
from src.models import AlexNet, loss_func, eval_func, init_weights
from src.optimizers import adjust_lr, set_lr, get_lr, build_sgd
from src.transforms import build_train_transform, build_test_transfrom


def save_ckpt(
    model: nn.Module,
    optimizer: Optimizer,
    curr_epoch: int,
    total_epoch: int,
    save_ckpt_p: Union[str, os.PathLike]
) -> None:
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curr_epoch": curr_epoch,
        "total_epoch": total_epoch
    }
    torch.save(state_dict, save_ckpt_p)
    

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

    curr_iter = 1
    loss = 0
    acc = 0
    batch_size = test_loader.batch_size

    for imgs, gt_cat_ids in test_loader:
        gt_cat_ids = gt_cat_ids[:batch_size]

        imgs: torch.Tensor = imgs.to(device, non_blocking = True)
        gt_cat_ids: torch.Tensor = gt_cat_ids.to(device, non_blocking = True)
        pred_logits: torch.Tensor = model(imgs)

        pred_logits = pred_logits.reshape(batch_size, 10, -1)
        pred_logits = torch.mean(pred_logits, dim = 1)

        curr_loss = loss_func(gt_cat_ids, pred_logits).item()
        curr_acc = eval_func(gt_cat_ids, pred_logits).item()

        loss = loss + (curr_loss - loss) / curr_iter
        acc = acc + (curr_acc - acc) / curr_iter
    
    return loss, acc


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
    warm_up_epoches: int,
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
    warm_up_lrs = [lr * warm_up_lr_multiplier for lr in init_lrs]
    set_lr(optimizer, warm_up_lrs)
    warm_up_flag = True

    last_print_iter = 0
    loss_last_epoch = float("inf")

    for i in range(num_epoches):

        curr_epoch += 1
        loss_curr_epoch = 0

        for j, (imgs, gt_cat_ids) in enumerate(train_loader):
            imgs: torch.Tensor = imgs.to(device, non_blocking = True)
            gt_cat_ids: torch.Tensor = gt_cat_ids.to(device, non_blocking = True)
            pred_logits: torch.Tensor = model(imgs)

            loss = loss_func(gt_cat_ids, pred_logits)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            curr_iter += batch_size

            loss_val = loss.item()
            loss_curr_epoch = loss_curr_epoch + (loss_val - loss_curr_epoch) / (j + 1)

            if curr_iter >= last_print_iter + print_iter_period:
                curr_lrs = get_lr(optimizer)
                metric_val = eval_func(gt_cat_ids, pred_logits).item()

                msg = f"[Train] epoch {curr_epoch}/{num_epoches}, "
                msg += f"iter {curr_iter}/{num_iters}, "
                msg += f"loss {loss_val:.3f}, accuracy {metric_val:.3f}, lr ["

                for lr in curr_lrs:
                    msg += f"{lr}, "
                
                msg += "]"
            
                logger.info(msg)
                writer.add_scalar("Train/loss", loss_val, curr_iter)
                writer.add_scalar("Train/accuracy", metric_val, curr_iter)

                for _, lr in enumerate(curr_lrs):
                    writer.add_scalar(f"Train/lr{_}", lr, curr_iter)
                
                last_print_iter = curr_iter
            
        if curr_epoch >= warm_up_epoches and warm_up_flag is True:
            set_lr(optimizer, init_lrs)
            msg = f"[Train] epoch {curr_epoch}/{num_epoches}, done warm up"
            logger.info(msg)
            warm_up_flag = False
        
        if curr_epoch % eval_save_epoch_period == 0:
            model.eval()

            test_loss, test_acc = test_model(
                model, test_loader, loss_func, eval_func, device
            )

            msg = f"[Test] epoch {curr_epoch}/{num_epoches}, "
            msg += f"loss {test_loss:.3f}, accuracy {test_acc:.3f}"

            logger.info(msg)
            writer.add_scalar("Test/loss", test_loss, curr_iter)
            writer.add_scalar("Test/accuracy", test_acc, curr_iter)

            save_ckpt_p = os.path.join(ckpt_dir, f"epoch{curr_epoch}.pth")
            save_ckpt(model, optimizer, curr_epoch, num_epoches, save_ckpt_p)

            model.train()
        
        if loss_last_epoch - loss_curr_epoch < 1e-6:
            adjust_lr(optimizer, adjust_lr_multiplier)
            
            msg = f"[Train] epoch {curr_epoch}/{num_epoches}, "
            msg += f"last_epoch_loss: {loss_last_epoch:.3f}, "
            msg += f"curr_epoch_loss: {loss_curr_epoch:.3f}, "
            msg += f"lower lr by: {adjust_lr_multiplier}"

            logger.info(msg)

        loss_last_epoch = loss_curr_epoch

def run_train(
    cfg_p: Union[str, os.PathLike],
) -> None:
    with open(cfg_p, "r") as f:
        cfg = json.load(f)

    dataset_cfg = cfg["dataset"]
    channel_mean = torch.as_tensor(dataset_cfg["channel_mean"])
    channel_std = torch.as_tensor(dataset_cfg["channel_std"])
    eigen_val = torch.as_tensor(dataset_cfg["eigen_val"])
    eigen_vec = torch.as_tensor(dataset_cfg["eigen_vec"])

    train_data_root = dataset_cfg["train_data_root"]
    test_data_root = dataset_cfg["test_data_root"]
    train_loader_cfg = cfg["train_loader"]
    test_loader_cfg = cfg["test_loader"]

    dirnames = os.listdir(train_data_root)
    dirnames = [d for d in dirnames if os.path.isdir(os.path.join(train_data_root, d))]
    dirnames.sort()
    cat_name_id_dict = {n: i for i, n in zip(range(len(dirnames)), dirnames)}

    train_loader = build_train_loader(
        [train_data_root],
        cat_name_id_dict,
        channel_mean,
        channel_std,
        eigen_vec,
        eigen_val,
        train_loader_cfg["batch_size"],
        train_loader_cfg["num_workers"]
    )

    test_loader = build_test_loader(
        [test_data_root],
        cat_name_id_dict,
        channel_mean,
        channel_std,
        test_loader_cfg["batch_size"],
        test_loader_cfg["num_workers"]
    )

    runtime_cfg = cfg["runtimes"]
    model_cfg = cfg["model"]
    optimizer_cfg = cfg["optimizer"]
    scheduler_cfg = cfg["scheduler"]

    model = AlexNet(model_cfg["num_classes"]).to(runtime_cfg["device"])
    init_weights(model)

    sgd = build_sgd(
        model.parameters(),
        optimizer_cfg["global"]["lr"],
        optimizer_cfg['global']["momentum"],
        optimizer_cfg['global']["weight_decay"]
    )

    train_model(
        model, 
        sgd,
        train_loader,
        test_loader,
        loss_func,
        eval_func,
        runtime_cfg["device"],
        runtime_cfg["export_dir"],
        runtime_cfg["num_epoches"],
        runtime_cfg["print_iter_period"],
        runtime_cfg["eval_save_epoch_period"],
        scheduler_cfg["warm_up_epoches"],
        scheduler_cfg["warm_up_lr_multiplier"],
        scheduler_cfg["adjust_lr_multiplier"]
    )
