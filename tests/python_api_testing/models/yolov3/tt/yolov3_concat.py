import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

from python_api_testing.models.yolov3.reference.models.common import autopad
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from utility_functions_new import torch2tt_tensor, tt2torch_tensor


class TtConcat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, device, state_dict, base_address, dimension=1):
        super().__init__()
        self.device = device
        self.base_address = base_address

        self.d = dimension

    def forward(self, x):
        return fallback_ops.concat(x, self.d)
