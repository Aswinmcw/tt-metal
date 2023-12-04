# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_layer_norm(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.experimental.layer_norm(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_layer_norm_with_weight_and_bias(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    weight = ttnn.from_torch(torch_weight)
    bias = ttnn.from_torch(torch_bias)

    input_tensor = ttnn.to_device(input_tensor, device)
    weight = ttnn.to_device(weight, device)
    bias = ttnn.to_device(bias, device)

    output_tensor = ttnn.experimental.layer_norm(input_tensor, weight=weight, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_layer_norm_with_weight_bias_and_residual_input(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor)
    weight = ttnn.from_torch(torch_weight)
    bias = ttnn.from_torch(torch_bias)

    input_tensor = ttnn.to_device(input_tensor, device)
    residual_input_tensor = ttnn.to_device(residual_input_tensor, device)
    weight = ttnn.to_device(weight, device)
    bias = ttnn.to_device(bias, device)

    output_tensor = ttnn.experimental.layer_norm(
        input_tensor, residual_input=residual_input_tensor, weight=weight, bias=bias
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_moreh_layer_norm_with_weight_bias_and_residual_input(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor)
    weight = ttnn.from_torch(torch_weight)
    bias = ttnn.from_torch(torch_bias)

    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    residual_input_tensor = ttnn.to_device(residual_input_tensor, device)
    residual_input_tensor = ttnn.to_layout(residual_input_tensor, ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    bias = ttnn.to_device(bias, device)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)

    output_tensor = ttnn.experimental.moreh_layer_norm(
        input_tensor, residual_input=residual_input_tensor, weight=weight, bias=bias
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)
