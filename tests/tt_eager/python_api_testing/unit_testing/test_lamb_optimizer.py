# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.reference_optimizer import (
    lamb_optimizer_kernel,
)
from loguru import logger


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("step_size", [1e-3])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("weight_decay", [0.01])
def test_lamb_kernel(input_shapes, beta1, beta2, step_size, eps, weight_decay, device):
    torch.manual_seed(0)
    param_data = torch.Tensor(size=input_shapes).uniform_(1, 100)
    grad_data, exp_avg_data, exp_avg_sq_data = param_data, param_data, param_data

    param = (
        tt_lib.tensor.Tensor(param_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    grad = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    exp_avg = (
        tt_lib.tensor.Tensor(exp_avg_data, tt_lib.tensor.DataType.BFLOAT16)
        .to(tt_lib.tensor.Layout.ROW_MAJOR)
        .to(device)
    )

    exp_avg_sq = (
        tt_lib.tensor.Tensor(exp_avg_sq_data, tt_lib.tensor.DataType.BFLOAT16)
        .to(tt_lib.tensor.Layout.ROW_MAJOR)
        .to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.lamb_optimizer(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        beta1=beta1,
        beta2=beta2,
        step_size=step_size,
        eps=eps,
        weight_decay=weight_decay,
    )
    tt_output_tensor_a = tt_output_tensor_on_device[0].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    tt_output_tensor_b = tt_output_tensor_on_device[1].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    tt_output_tensor_c = tt_output_tensor_on_device[2].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    exp_avg_out, exp_avg_sq_out, param = lamb_optimizer_kernel.lamb_kernel(
        param_data,
        grad_data,
        exp_avg_data,
        exp_avg_sq_data,
        beta1=beta1,
        beta2=beta2,
        step_size=step_size,
        eps=eps,
        weight_decay=weight_decay,
    )

    comp_pass_a, _ = comparison_funcs.comp_pcc(exp_avg_out, tt_output_tensor_a, 0.99)
    _, comp_out_a = comparison_funcs.comp_allclose_and_pcc(exp_avg_out, tt_output_tensor_a)

    comp_pass_b, _ = comparison_funcs.comp_pcc(exp_avg_sq_out, tt_output_tensor_b, 0.99)
    _, comp_out_b = comparison_funcs.comp_allclose_and_pcc(exp_avg_sq_out, tt_output_tensor_b)

    comp_pass_c, _ = comparison_funcs.comp_pcc(param, tt_output_tensor_c, 0.99)
    _, comp_out_c = comparison_funcs.comp_allclose_and_pcc(param, tt_output_tensor_c)

    logger.info(comp_out_a)
    logger.info(comp_out_b)
    logger.info(comp_out_c)
    assert comp_pass_a & comp_pass_b & comp_pass_c
