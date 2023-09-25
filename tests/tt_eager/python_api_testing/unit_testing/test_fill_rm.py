# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib
from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.common import (
    is_wormhole_b0,
    skip_for_wormhole_b0,
)


def test_fill_rm(device):
    N = 2
    C = 3
    H = 64
    W = 96

    fillH = 33
    fillW = 31

    if is_wormhole_b0():
        N, C, H, W = [1, 1, 32, 32]
        fillH = 31
        fillW = 31

    x = torch.zeros((N, C, H, W))
    xp = torch.clone(x)
    xp[:, :, :fillH, :fillW] = 1.0

    xt = (
        tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_lib.tensor.Layout.TILE)
        .to(device)
    )
    xtt = tt_lib.tensor.fill_ones_rm(N, C, H, W, fillH, fillW, xt)
    assert xtt.shape() == [N, C, H, W]

    tt_got_back = xtt.cpu().to_torch()

    # x[1,1,2,2] = 2.0
    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, xp)
    assert torch.equal(tt_got_back, xp)
