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

from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


def ref_groupnorm(x, group_size, eps, **kwargs):
    n_channels = x.shape[1]
    lnorm = torch.nn.GroupNorm(group_size, n_channels, eps, **kwargs)
    return lnorm(x)


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
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

    N = 1
    C = 32
    H = 32
    W = 32
    grid_size = (1, 1)
    in0_shape = (1, 1, N * C * H, W)
    group_size = 1

    shard_spec = ttl.tensor.ShardSpec(
        ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), grid_size)}),
        (in0_shape[2] // grid_size[1], in0_shape[3] // grid_size[0]),
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = ttl.tensor.Tensor(in0, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE)
    in0_t_sharded = in0_t.to(device, in0_mem_config, shard_spec)

    program_config = ttl.operations.primary.GroupNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=N,
        block_w=4,
        math_fidelity=fidelity,
        im_data_format=cb_dtype,
        out_data_format=out_dtype,
        inplace=True,
    )
