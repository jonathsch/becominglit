import inspect
import re
from collections import OrderedDict
from typing import Dict, List, get_args

import torch
from torch import nn


def filter_params(params, ignore_names):
    return OrderedDict(
        [(k, v) for k, v in params.items() if not any([re.match(n, k) is not None for n in ignore_names])]
    )


def get_inputs(model: nn.Module, required_only: bool = True) -> List[str]:
    """Returns names of model inputs."""
    return [
        name
        for name, param in inspect.signature(model.forward).parameters.items()
        if not required_only or type(None) not in get_args(param.annotation)
    ]


def filter_inputs(
    inputs: Dict[str, torch.Tensor], model: nn.Module, required_only: bool = True
) -> Dict[str, torch.Tensor]:
    """Returns a subset of inputs for the model."""
    return {name: inputs[name] for name in get_inputs(model, required_only) if name in inputs}
