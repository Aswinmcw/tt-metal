# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
def test_argmin(input_shapes, device):
    torch.manual_seed(0)
    input_data = torch.randn(input_shapes).bfloat16()
    input_data[0, 0, 0, 0] = -100
    input_tensor = (
        tt_lib.tensor.Tensor(input_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.argmin(input_tensor, 0)
    # print("minimum value ")
    # print(torch.min(input_data))
    golden_tensor = torch.argmin(input_data)
    tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    pt_out_tensor = golden_tensor
    torch.set_printoptions(sci_mode=False, threshold=10000)
    # print(input_data)
    # print(pt_out_tensor, tt_out_tensor)
    print(pt_out_tensor, tt_out_tensor[0, 0, 0, 0])
    comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor[0, 0, 0, 0], pcc=0.99)
    comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
    logger.info(comp_pass)
    logger.info(comp_all)
    logger.info(comp_out)
    status = comp_pass | comp_all
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
def test_argmin_w(input_shapes, device):
    torch.manual_seed(0)
    input_data = torch.randn(input_shapes).bfloat16()
    input_data[0, 0, 0, 0] = -100
    input_tensor = (
        tt_lib.tensor.Tensor(input_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.argmin(input_tensor, dim=3)
    # print("Maximum value ")
    # print(torch.max(input_data))
    golden_tensor = torch.argmin(input_data, dim=3)

    tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    pt_out_tensor = golden_tensor
    torch.set_printoptions(sci_mode=False, threshold=10000)
    print(pt_out_tensor, tt_out_tensor[0, 0, 0])
    comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor[0, 0, 0], pcc=0.99)
    comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor[0, 0, 0], atol=4, rtol=1e-1)
    logger.info(comp_pass)
    logger.info(comp_all)
    logger.info(comp_out)
    status = comp_pass | comp_all
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
def test_argmin_h(input_shapes, device):
    torch.manual_seed(0)

    input_data = torch.randn(input_shapes).bfloat16()
    input_data[0, 0, 0, 0] = -100
    input_tensor = (
        tt_lib.tensor.Tensor(input_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.argmin(input_tensor, dim=2)
    # print("Maximum value ")
    # print(torch.max(input_data, dim =2))
    golden_tensor = torch.argmin(input_data, dim=2)

    tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    pt_out_tensor = golden_tensor
    torch.set_printoptions(sci_mode=False, threshold=10000)
    # # print(input_data)
    # print(pt_out_tensor, tt_out_tensor)
    print(pt_out_tensor, tt_out_tensor[0, 0, 0])
    comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor[0, 0, 0], pcc=0.99)
    comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor[0, 0, 0], atol=4, rtol=1e-1)
    logger.info(comp_pass)
    logger.info(comp_all)
    logger.info(comp_out)
    status = comp_pass | comp_all
    assert status
