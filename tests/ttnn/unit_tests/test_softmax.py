# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch_random


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_softmax(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((1, 16, h, w), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_moreh_softmax(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((1, 16, h, w), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.experimental.moreh_softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("num_columns_to_keep", [10])
def test_moreh_softmax_with_mask(device, h, w, num_columns_to_keep):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((1, 16, h, w), -10, 10, dtype=torch.bfloat16)
    torch_input_tensor[..., num_columns_to_keep:] -= 200
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.experimental.moreh_softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
