import logging
import shutil
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
import wandb
import wandb.wandb_run
from PIL import Image
from torch import nn
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from becominglit.util.image import linear2srgb
from becominglit.util.inspection import filter_inputs
from becominglit.util.torchutils import save_checkpoint, to_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    config: Dict,
    batch_transform_fn: Optional[Callable] = None,
    device: Union[torch.device, str] = "cuda:0",
) -> Dict[str, float]:
    model.eval()
    psnr_fn = PeakSignalNoiseRatio(data_range=(0, 1)).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    for i, batch in enumerate(val_loader):
        batch = to_device(batch, device)
        if batch_transform_fn is not None:
            batch_transform_fn(batch)

        preds = model.forward(**filter_inputs(batch, model, required_only=False))

        pred_img = linear2srgb(preds["rgb"]).clamp(0.0, 1.0).nan_to_num(0.0)
        gt_img = linear2srgb(batch["image"]).clamp(0.0, 1.0).nan_to_num(0.0)

        psnr_fn.update(pred_img, gt_img)
        ssim_fn.update(pred_img, gt_img)
        lpips_fn.update(pred_img, gt_img)

    psnr = psnr_fn.compute().item()
    ssim = ssim_fn.compute().item()
    lpips = lpips_fn.compute().item()

    del psnr_fn
    del ssim_fn
    del lpips_fn

    torch.cuda.empty_cache()
    model.train()

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    config: Mapping[str, Any],
    loss_fn: Callable,
    last_iter: int,
    wandb_run: Optional[wandb.wandb_run.Run] = None,
    batch_transform_fn: Optional[Callable] = None,
    summary_fn: Optional[Callable] = None,
    iteration: int = 0,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    device: Optional[Union[torch.device, str]] = "cuda:0",
) -> None:
    trainloader_iter = iter(train_loader)

    while True:
        try:
            batch = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(train_loader)
            batch = next(trainloader_iter)

        batch = to_device(batch, device)
        if batch_transform_fn is not None:
            batch_transform_fn(batch)

        is_summary_step = summary_fn is not None and iteration % config.train.summary_every_n_steps == 0

        # train step
        preds = model.forward(**filter_inputs(batch, model, required_only=False), render_auxiliary=is_summary_step)

        loss, loss_dict = loss_fn(preds, batch, iteration=iteration)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        params = [p for pg in optimizer.param_groups for p in pg["params"]]
        for p in params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.data[torch.isnan(p.grad.data)] = 0
                p.grad.data[torch.isinf(p.grad.data)] = 0
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=True)
        optimizer.step()

        # clamp material to reasonable values
        if hasattr(model, "albedo"):
            model.albedo.data.clamp_(0.0, 1.0)

        # logging
        if iteration % config.train.log_every_n_steps == 0:
            loss_dict = {k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")}
            loss_str = " ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
            logger.info(f"iter={iteration}: {loss_str}")

            if wandb_run is not None:
                wandb_run.log({f"train/{name}": value for name, value in loss_dict.items()}, step=iteration)

        # summary
        if summary_fn is not None and iteration % config.train.summary_every_n_steps == 0:
            summary = summary_fn(preds, batch)
            if wandb_run is not None:
                wandb_run.log(
                    {f"images/{name}": wandb.Image(value, file_type="jpg") for name, value in summary.items()},
                    step=iteration,
                )
            else:
                for k, v in summary.items():
                    Image.fromarray(v).save(f"{config.train.run_dir}/images/{k}_{iteration:06d}.jpg")

        # checkpointing
        if iteration % config.train.ckpt_every_n_steps == 0:
            logger.info(f"iter={iteration}: saving checkpoint to `{config.train.ckpt_dir}`")
            save_checkpoint(
                f"{config.train.ckpt_dir}/latest.pt",
                {
                    "model": model,
                    "optimizer": optimizer,
                },
                iteration=iteration,
            )
            shutil.copyfile(f"{config.train.ckpt_dir}/latest.pt", f"{config.train.ckpt_dir}/{iteration:06d}.pt")

        # validation
        if val_loader is not None and iteration % config.train.validate_every_n_steps == 0:
            logger.info(f"iter={iteration}: running validation")
            val_metrics = validate(model, val_loader, config, batch_transform_fn, device)
            logger.info(f"iter={iteration}: validation metrics: {val_metrics}")

            if wandb_run is not None:
                wandb_run.log({f"val/{name}": value for name, value in val_metrics.items()}, step=iteration)

        iteration += 1
        if iteration >= last_iter:
            break

    return iteration
