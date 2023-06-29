from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


import torch
from torch import nn
from deit_config import DeiTConfig
from typing import Union, Optional, Tuple, Dict, Set, List

import tt_lib
# from deit_layer import TtDeiTLayer
from tt_lib.fallback_ops import fallback_ops
from deit_helper_funcs import make_linear
from utility_functions_new import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor


class TtDeiTSelfAttention(nn.Module):
    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="") -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.host = host
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.query.weight"], device)
        # self.query_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.query.bias"], device)

        # self.key_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.key.weight"], device)
        # self.key_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.key.bias"], device)

        # self.value_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.value.weight"], device)
        # self.value_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.value.bias"], device)
        # print('\nquery::::', f"{state_dict}.attention")

        self.query = make_linear(config.hidden_size, self.all_head_size, "query", state_dict, base_address, device)
        self.key = make_linear(config.hidden_size, self.all_head_size, "key", state_dict, base_address, device)
        self.value = make_linear(config.hidden_size, self.all_head_size, "value", state_dict, base_address, device)

    def transpose_for_scores(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        # x must be 4d originaly
        # 1 is appended to the beggining
        # so create tensor shape by ommiting the first dimension
        new_x_shape = list(x.shape()[1:-1]) + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = fallback_ops.reshape(x, *new_x_shape)
        x = tt_lib.tensor.permute(x, 0, 2, 1, 3)
        return x

    def forward(self,hidden_states,head_mask, output_attentions):
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer_transposed = tt_lib.tensor.transpose(key_layer)

        attention_scores = tt_lib.tensor.bmm(query_layer, key_layer_transposed)

        attention_head_size_tt = fallback_ops.full(
            attention_scores.shape(), self.attention_head_size
        )
        attention_head_size_tt = tt_lib.tensor.sqrt(attention_head_size_tt)
        attention_head_size_tt = tt_lib.tensor.recip(attention_head_size_tt)

        attention_scores = tt_lib.tensor.mul(attention_scores, attention_head_size_tt)

        # Normalize the attention scores to probabilities.
        attention_probs = fallback_ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tt_lib.tensor.bmm(attention_probs, value_layer)
        context_layer = tt_lib.tensor.permute(context_layer, 0, 2, 1, 3)
        new_context_layer_shape = (1, ) + tuple(context_layer.shape()[:-2]) + (self.all_head_size,)
        context_layer = fallback_ops.reshape(context_layer, *new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
