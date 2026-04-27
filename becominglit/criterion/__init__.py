# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors

import copy

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from omegaconf import DictConfig
from torch import nn

from becominglit.criterion.registry import (
    get_loss,
    register_loss_by_fn,
)
from becominglit.util.module_loader import load_from_config, load_module


class MonotonicWeightSchedule:
    def __init__(self, start: int, end: int, init_value: float, target_value: float):
        self.start = start
        self.end = end
        self.init_value = init_value
        self.target_value = target_value
        self.delta = (target_value - init_value) / (end - start)

    def __call__(self, iteration: int):
        if iteration < self.start:
            return self.init_value
        elif iteration > self.end:
            return self.target_value
        t = min(iteration, self.end) - self.start
        return self.init_value + t * self.delta


class ModularLoss(nn.Module):
    def __init__(self, losses, extra_modules_loaded=None):
        super().__init__()

        if extra_modules_loaded is None:
            extra_modules_loaded = []

        # constructing
        self.weights = {}
        self.args = {}
        self.start_at = {}
        self.end_at = {}
        self.schedule = {}
        # TODO: should this be a module dict?
        self.fns = nn.ModuleDict()
        # load potential loss modules
        for extra_module in extra_modules_loaded:
            load_module(extra_module)
        # load losses from config `loss.losses`
        for loss_name, loss_def in losses.items():
            loss_def = copy.deepcopy(loss_def)

            # TODO: probably can get rid of this, keeping for compat-ty
            # get kwargs for arguments of loss class constructor
            if isinstance(loss_def, DictConfig):
                loss_init_kwargs = loss_def.pop("init_kwargs", {})
            else:
                loss_init_kwargs = {}

            # load the loss class
            loss_class_name = loss_def.pop("loss_fn", loss_name)
            if isinstance(loss_def, DictConfig) and "loss_type" in loss_def:
                loss_class_name = loss_def.pop("loss_type")

            # decide the weights, etc.
            if isinstance(loss_def, DictConfig):
                assert "weight" in loss_def or "schedule" in loss_def
                if "weight" in loss_def:
                    self.weights[loss_name] = float(loss_def.pop("weight"))
                elif "schedule" in loss_def:
                    self.schedule[loss_name] = load_from_config(loss_def.pop("schedule"))
                if "start_at" in loss_def:
                    self.start_at[loss_name] = loss_def.pop("start_at")
                if "end_at" in loss_def:
                    self.end_at[loss_name] = loss_def.pop("end_at")
                if loss_def:
                    loss_init_kwargs.update(**loss_def)
            elif isinstance(loss_def, (float, int)):
                self.weights[loss_name] = float(loss_def)
            else:
                raise ValueError("unsupported loss definition")

            self.fns[loss_name] = get_loss(loss_class_name, loss_init_kwargs)

    def forward(self, preds, targets, iteration=None):
        loss_total = 0.0
        losses_dict = {"loss_total": loss_total}
        for loss_name, loss_fn in self.fns.items():
            args = self.args.get(loss_name, {})

            weight = 1.0
            if loss_name in self.weights:
                weight = self.weights[loss_name]
            elif loss_name in self.schedule:
                assert iteration is not None, "Provide `iteration` when using schedules!"
                weight = self.schedule[loss_name](iteration)

            if weight == 0.0:
                losses_dict[f"loss_{loss_name}"] = torch.tensor(0.0)
                continue

            loss_value = loss_fn(preds, targets, **args)
            losses_dict[f"loss_{loss_name}"] = loss_value

            loss_total += weight * loss_value

        losses_dict.update(loss_total=loss_total)

        return loss_total, losses_dict


@register_loss_by_fn()
def rgb_l1(
    preds,
    targets,
    src_key: str = "rgb",
    tgt_key: str = "image",
):
    clamp_mask = targets[tgt_key] > 0.98
    preds_src = torch.where(clamp_mask, preds[src_key].clamp(0.0, 1.0), preds[src_key])
    return (preds_src - targets[tgt_key]).abs().mean()


@register_loss_by_fn()
def rgb_l2(
    preds,
    targets,
    src_key: str = "rgb",
    tgt_key: str = "image",
):
    return (preds[src_key] - targets[tgt_key]).square().mean()


@register_loss_by_fn()
def rgb_ssim(
    preds,
    targets,
    src_key: str = "rgb",
    tgt_key: str = "image",
):
    clamp_mask = targets[tgt_key] > 0.98
    preds_src = torch.where(clamp_mask, preds[src_key].clamp(0.0, 1.0), preds[src_key])
    return (1.0 - fused_ssim(preds_src, targets[tgt_key])).mean()


@register_loss_by_fn("bound_primscale_rgca")
def loss_bound_primscale(
    preds,
    targets=None,
    key: str = "scales_preclip",
    min_scale: float = 0.1,
    max_scale: float = 20.0,
):
    primscale = preds[key]
    return torch.where(
        primscale < min_scale,
        1.0 / primscale.clamp(1e-7, torch.inf),
        torch.where(primscale > max_scale, (primscale - max_scale) ** 2, 0.0),
    ).mean()


@register_loss_by_fn("l2_reg")
def loss_l2_reg(preds, targets=None, key="normal_offsets", threshold: float = 0.0):
    return F.relu(preds[key].square() - threshold).mean()


@register_loss_by_fn("negcolor")
def loss_negcolor(preds, batch=None, key: str = "diff_color"):
    return preds[key].clamp(max=0.0).pow(2).mean()
