# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import math
import torch

from typing import Optional
import tt_lib
from tt_lib.utils import pad_weight
from models.utility_functions import torch2tt_tensor
from functools import partial


def mha(qkv_weight, qkv_bias, hidden_dim, num_heads, device, model_config):
    assert isinstance(num_heads, int) and num_heads > 0

    # Used to scale down the input to the softmax
    freciprocal_of_sqrt_hidden_dim = 1 / math.sqrt(hidden_dim // num_heads)

    reserve_split_heads_shape = model_config.get("RESERVE_SPLIT_HEADS_SHAPE", None)

    if "OP1_FUSED_QKV_MM_CONFIG" in model_config:
        qkv_matmul = partial(
            tt_lib.operations.primary.matmul,
            program_config=model_config["OP1_FUSED_QKV_MM_CONFIG"],
            output_mem_config=model_config["OP1_FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP1_FUSED_QKV_MM_OUTPUT_DTYPE"],
        )
    else:
        qkv_matmul = partial(
            tt_lib.tensor.bert_large_fused_qkv_matmul,
            output_mem_config=model_config["OP1_FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP1_FUSED_QKV_MM_OUTPUT_DTYPE"],
        )
    if "OP7_PRE_SOFTMAX_BMM_CONFIG" in model_config:
        pre_softmax_bmm = partial(
            tt_lib.operations.primary.matmul,
            program_config=model_config["OP7_PRE_SOFTMAX_BMM_CONFIG"],
            output_mem_config=model_config["OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE"],
        )
    else:
        pre_softmax_bmm = partial(
            tt_lib.tensor.bert_large_pre_softmax_bmm,
            output_mem_config=model_config["OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE"],
        )
    if "OP9_POST_SOFTMAX_BMM_CONFIG" in model_config:
        post_softmax_bmm = partial(
            tt_lib.operations.primary.matmul,
            program_config=model_config["OP9_POST_SOFTMAX_BMM_CONFIG"],
            output_mem_config=model_config["OP9_POST_SOFTMAX_BMM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP9_POST_SOFTMAX_BMM_OUTPUT_DTYPE"],
        )
    else:
        post_softmax_bmm = partial(
            tt_lib.tensor.bert_large_post_softmax_bmm,
            output_mem_config=model_config["OP9_POST_SOFTMAX_BMM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP9_POST_SOFTMAX_BMM_OUTPUT_DTYPE"],
        )

    if "OP8_SOFTMAX_CONFIG" in model_config:
        softmax = partial(
            tt_lib.operations.primary.transformers.scale_mask_softmax_in_place,
            program_config=model_config["OP8_SOFTMAX_CONFIG"],
        )
    else:
        softmax = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place

    split_fused_qkv_and_split_heads = partial(
        tt_lib.operations.primary.transformers.split_fused_qkv_and_split_heads,
        compute_with_storage_grid_size=model_config.get("GRID_SIZE", device.compute_with_storage_grid_size()),
        output_mem_config=model_config["OP2TO6_SPLIT_QKV_HEADS_OUTPUT_MEMCFG"],
    )

    def op1_qkv_fused(activation, qkv_weight, qkv_bias):
        qkv = qkv_matmul(
            activation,
            qkv_weight,
            bias=qkv_bias,
        )
        return qkv

    def op2to6_create_qkv_heads(qkv):
        (
            q_heads,
            kt_heads,
            v_heads,
        ) = split_fused_qkv_and_split_heads(
            qkv,
        )
        return q_heads, kt_heads, v_heads

    def op7_bmm(Q_heads, K_T_heads):
        qkt = pre_softmax_bmm(
            Q_heads,
            K_T_heads,
        )
        return qkt

    def op8_scale_mask_softmax(qkt, attention_mask):
        # Attention scores computation

        # Input and output tensors of this fused op is: [9, 1, 6144, 384] instead of [9, 16, 384, 384]
        # No-op reshapes are handled within pre-softmax (op 7) and post-softmax bmms (op 9)
        shape = qkt.shape()
        qkt = qkt.reshape(shape[0], 1, shape[1] * shape[2], shape[3])
        attention_scores = softmax(qkt, freciprocal_of_sqrt_hidden_dim, attention_mask)
        attention_scores = attention_scores.reshape(shape)

        return attention_scores

    def op9_bmm(attention_scores, V_heads):
        weighted_activation = post_softmax_bmm(
            attention_scores,
            V_heads,
        )

        return weighted_activation

    def op10_unmake_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            retval = tt_lib.tensor.nlp_concat_heads(
                x,
                output_mem_config=model_config["OP10_CONCATENATE_ATTENTION_HEADS_OUTPUT_MEMCFG"],
            )
            return retval

    def mha_(activation, attention_mask):
        # TODO: Remove hardcoded shape hack
        if reserve_split_heads_shape is not None:
            temp = tt_lib.tensor.empty(
                reserve_split_heads_shape,
                tt_lib.tensor.DataType.BFLOAT16,
                tt_lib.tensor.Layout.ROW_MAJOR,
                activation.device(),
                tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1),
            )
        qkv = op1_qkv_fused(activation, qkv_weight, qkv_bias)
        if reserve_split_heads_shape is not None:
            temp.deallocate()
        # activation.deallocate()

        Q_heads, K_T_heads, V_heads = op2to6_create_qkv_heads(qkv)
        qkv.deallocate()

        qkt = op7_bmm(Q_heads, K_T_heads)
        Q_heads.deallocate()
        K_T_heads.deallocate()

        attention_scores = op8_scale_mask_softmax(qkt, attention_mask)
        # Should be a no-op deallocate since it was moved?
        # qkt.deallocate()
        weighted_activation = op9_bmm(attention_scores, V_heads)
        attention_scores.deallocate()
        V_heads.deallocate()

        res = op10_unmake_attention_heads(
            weighted_activation
        )  # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]
        weighted_activation.deallocate()

        return res

    return mha_


class TtMultiHeadAttentionModel:
    def __init__(self, config, encoder_idx, state_dict, device, model_config=None, tt_cache_path=None):
        layer_name = f"bert.encoder.layer.{encoder_idx}.attention.self"

        if tt_cache_path is not None:
            interleaved_str = ""
            if "QKV_INTERLEAVED" in model_config:
                interleaved_str = f"interleaved_{model_config['QKV_INTERLEAVED']}_"
            qkv_weight = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.qkv.weight_{interleaved_str}{model_config['OP1_FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, model_config["OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG"])
            qkv_bias = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.qkv.bias_{interleaved_str}{model_config['OP1_FUSED_QKV_MM_BIAS_DTYPE'].name}.bin"
                )
            ).to(device, model_config["OP1_FUSED_QKV_MM_BIAS_MEMCFG"])
        else:
            qw = state_dict[f"{layer_name}.query.weight"]
            qb = state_dict[f"{layer_name}.query.bias"]
            kw = state_dict[f"{layer_name}.key.weight"]
            kb = state_dict[f"{layer_name}.key.bias"]
            vw = state_dict[f"{layer_name}.value.weight"]
            vb = state_dict[f"{layer_name}.value.bias"]

            # Weights pre-transposed on host​. No on-the fly transpose of W​
            qw = torch.transpose(qw, -1, -2)
            kw = torch.transpose(kw, -1, -2)
            vw = torch.transpose(vw, -1, -2)
            if "QKV_INTERLEAVED" in model_config:
                const_w_dims = qw.shape[:-1]
                const_b_dims = qb.shape[:-1]
                qw = qw.reshape([*const_w_dims, model_config["QKV_INTERLEAVED"], -1])
                qb = qb.reshape([*const_b_dims, model_config["QKV_INTERLEAVED"], -1])
                kw = kw.reshape(qw.shape)
                kb = kb.reshape(qb.shape)
                vw = vw.reshape(qw.shape)
                vb = vb.reshape(qb.shape)
                qkv_weight = torch.cat((qw, kw, vw), -1).reshape([*const_w_dims, -1])
                qkv_bias = torch.cat((qb, kb, vb), -1).reshape([*const_b_dims, -1])
            else:
                qkv_weight = torch.cat((qw, kw, vw), -1)
                qkv_bias = torch.cat((qb, kb, vb), -1)

            qkv_weight = pad_weight(qkv_weight)
            qkv_bias = pad_weight(qkv_bias)

            qkv_weight = torch2tt_tensor(
                qkv_weight,
                device,
                tt_layout=tt_lib.tensor.Layout.TILE,
                tt_memory_config=model_config["OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                tt_dtype=model_config["OP1_FUSED_QKV_MM_WEIGHTS_DTYPE"],
            )

            qkv_bias = torch2tt_tensor(
                qkv_bias,
                device,
                tt_layout=tt_lib.tensor.Layout.TILE,
                tt_memory_config=model_config["OP1_FUSED_QKV_MM_BIAS_MEMCFG"],
                tt_dtype=model_config["OP1_FUSED_QKV_MM_BIAS_DTYPE"],
            )

        # Hidden dim
        hidden_dim = qkv_weight.shape()[-1] // 3

        self.mha = mha(
            qkv_weight,
            qkv_bias,
            hidden_dim,
            config.num_attention_heads,
            device,
            model_config,
        )

    def __call__(
        self, activation: tt_lib.tensor.Tensor, attention_mask: Optional[tt_lib.tensor.Tensor] = None
    ) -> tt_lib.tensor.Tensor:
        result = self.mha(activation, attention_mask)
        return result
