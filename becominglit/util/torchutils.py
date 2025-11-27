import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from becominglit.util.inspection import filter_params

logger = logging.getLogger(__name__)


def to_device(
    things,
    device: torch.device,
    cache: Optional[Dict[str, torch.Tensor]] = None,
    key: Optional[str] = None,
    verbose: bool = False,
    max_bs: Optional[int] = None,
    non_blocking: bool = False,
):
    """Sends a potentially nested container of Tensors to the specified
    device. Non-tensors are preserved as-is.

    Args:
        things: Container with tensors or other containers of tensors to send
            to a GPU.

        device: Device to send the tensors to.

        cache: Optional dictionary to use as a cache for CUDAfied tensors. If
            passed, use this cache to allocate a tensor once and then resize /
            refill it on future calls to to_device() instead of reallocating
            it.

        key: If using the cache, store the tensor in this key, only for
            internal use.

        verbose: Print some info when a cached tensor is resized.

        max_bs: Maximum batch size allowed for tensors in cache

        non_blocking: if True and this copy is between CPU and GPU, the copy
            may occur asynchronously with respect to the host. For other cases,
            this argument has no effect.

    Returns:
        collection: The input collection with all tensors transferred to the given device.
    """
    device = torch.device(device)

    pr = print if verbose else lambda *args, **kwargs: None

    if isinstance(things, torch.Tensor) and things.device != device:
        if cache is not None:
            assert key is not None
            batch_size = things.shape[0]
            if key in cache:
                assert things.shape[1:] == cache[key].shape[1:]
                if batch_size > cache[key].shape[0]:
                    pr("Resized:", key, "from", cache[key].shape[0], "to", batch_size)
                    cache[key].resize_as_(things)
            else:
                buf_shape = list(things.shape)
                if max_bs is not None:
                    assert max_bs >= batch_size
                    buf_shape[0] = max_bs
                cache[key] = torch.zeros(*buf_shape, dtype=things.dtype, device=device)
                pr("Allocated:", key, buf_shape)
            cache[key][:batch_size].copy_(things, non_blocking=non_blocking)

            return cache[key][:batch_size]
        else:
            return things.to(device, non_blocking=non_blocking)
    elif isinstance(things, torch.nn.Module):
        return things.to(device, non_blocking=non_blocking)
    elif isinstance(things, dict):
        key = key + "." if key is not None else ""
        return {k: to_device(v, device, cache, key + k, verbose, max_bs, non_blocking) for k, v in things.items()}
    elif isinstance(things, Sequence) and not isinstance(things, str):
        key = key if key is not None else ""
        out = [to_device(v, device, cache, key + f"_{i}", verbose, max_bs, non_blocking) for i, v in enumerate(things)]
        if isinstance(things, tuple):
            out = tuple(out)
        return out
    elif isinstance(things, np.ndarray):
        return to_device(torch.from_numpy(things), device, cache, key, verbose, max_bs, non_blocking)
    else:
        return things


def save_checkpoint(ckpt_path, modules: Dict[str, Any], iteration: int):
    ckpt_dict = {"iteration": iteration}
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
    for name, mod in modules.items():
        if hasattr(mod, "module"):
            mod = mod.module
        ckpt_dict[name] = mod.state_dict()
    torch.save(ckpt_dict, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    modules: Dict[str, Any],
    iteration: int = None,
    strict: bool = False,
    map_location: Optional[str] = None,
    ignore_names: Optional[Dict[str, List[str]]] = None,
) -> int:
    """Load a checkpoint.
    Args:
        ckpt_path: directory or the full path to the checkpoint
    """
    if map_location is None:
        map_location = "cpu"
    # adding
    if os.path.isdir(ckpt_path):
        if iteration is None:
            ckpt_path = os.path.join(ckpt_path, "latest.pt")
        else:
            ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
    logger.info(f"loading checkpoint {ckpt_path}")
    ckpt_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    for name, mod in modules.items():
        params = ckpt_dict[name]
        if ignore_names is not None and name in ignore_names:
            logger.info(f"skipping: {ignore_names[name]}")
            params = filter_params(params, ignore_names[name])
        if isinstance(mod, nn.Module):
            mod.load_state_dict(params, strict=strict)
        elif isinstance(mod, torch.optim.Optimizer):
            mod.load_state_dict(params)

    return ckpt_dict["iteration"]
