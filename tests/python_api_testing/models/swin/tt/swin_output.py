from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


from tt_models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from python_api_testing.models.swin.swin_helper_funcs import linear as TtLinear
import tt_lib


class TtSwinOutput(nn.Module):
    def __init__(self, config, dim, state_dict, base_address, device, host):
        super().__init__()
        self.device = device

        self.dense_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.weight"], self.device
        )
        self.dense_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.bias"], self.device
        )

    def forward(self, hidden_states: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = TtLinear(hidden_states, self.dense_weight, self.dense_bias)
        return hidden_states
