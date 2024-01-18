# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import pytest

import torch
import tt_lib as ttl
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


def manual_group_norm(input_tensor, num_groups, eps=1e-2):
    N, C, H, W = input_tensor.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    # print("input_tensor")
    # print(input_tensor.shape)

    # Reshape into groups
    group_channels = C // num_groups
    input_tensor = input_tensor.view(N, num_groups, group_channels, H, W)

    # print("input_tensor_regroup")
    # print(input_tensor)

    # Calculate mean and variance
    mean = input_tensor.mean(dim=(2, 3, 4), keepdim=True)

    # print("mean")
    # print(mean)

    var = input_tensor.var(dim=(2, 3, 4), keepdim=True)

    # print("var")
    # print(var)

    # print("input_tensor - mean")
    # print((input_tensor - mean))

    # Normalize
    input_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # print("Normalize")
    # print(input_tensor)

    # Reshape back to original dimensions
    input_tensor = input_tensor.view(N, C, H, W)

    # print("input_tensor_regroup")
    # print(input_tensor)

    return input_tensor


def ref_groupnorm(x, group_size, eps, **kwargs):
    n_channels = x.shape[1]
    lnorm = torch.nn.GroupNorm(group_size, n_channels, eps, **kwargs)
    return lnorm(x)


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0,),
    ids=[
        "GN",
    ],
)
def test_groupnorm_sharded(test_id, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    epsf = 1e-2

    num_groups = 4
    num_batches = 6

    C = 128
    H = 16
    W = 32
    grid_size = (2, 2)
    in0_shape = (1, 1, num_batches * W * H, C)

    pyt_in0_shape = (num_batches, C, H, W)
    pyt_in0 = torch.rand(pyt_in0_shape)

    print("in0_shape ")
    print(in0_shape)

    shard_shape = [int(num_batches * W * H / 2), int(C / 2)]

    print("shard_shape")
    print(shard_shape)

    in0 = pyt_in0.transpose(1, -1).contiguous().view(1, 1, -1, C)

    print(pyt_in0)
    print(in0)

    # shard_spec = ttl.tensor.ShardSpec(
    #     ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), grid_size)}),
    #     (in0_shape[2] // grid_size[1], in0_shape[3] // grid_size[0]),
    #     ttl.tensor.ShardOrientation.ROW_MAJOR,
    #     False)

    # in0_t = ttl.tensor.Tensor(in0, ttl.tensor.DataType.BFLOAT16)
    # in0_t_sharded = in0_t.to(device, in0_mem_config, shard_spec)

    # layout = ttl.tensor.Layout.ROW_MAJOR
    layout = ttl.tensor.Layout.TILE

    in0_t = torch2tt_tensor(
        in0, device, tt_memory_config=in0_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16, tt_layout=layout
    )
    in0_t_sharded = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        shard_shape,
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.GroupNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
    )

    out_t = ttl.operations.primary.groupnorm(
        in0_t_sharded, num_groups, num_batches, epsf, output_mem_config=out_mem_config, program_config=program_config
    )

    out_t = ttl.tensor.sharded_to_interleaved(out_t, in0_mem_config)
    out = tt2torch_tensor(out_t)

    manual_out = manual_group_norm(pyt_in0, num_groups=num_groups, eps=epsf)
    manual_out = manual_out.transpose(1, -1).contiguous().view(1, 1, -1, C)

    pyt_groupnorm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=C, eps=epsf)
    pyt_out = pyt_groupnorm(pyt_in0)
    pyt_out = pyt_out.transpose(1, -1).contiguous().view(1, 1, -1, C)

    print(pyt_out[0][0][0])
    print(manual_out[0][0][0])
    print(out[0][0][0])

    passing, output = comp_pcc(pyt_out, out)
    logger.info(output)
