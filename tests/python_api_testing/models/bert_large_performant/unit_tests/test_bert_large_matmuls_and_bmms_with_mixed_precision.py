from pathlib import Path
import sys
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

import numpy as np

import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    comp_pcc,
)
import torch
import pytest


def run_bert_large_matmul_test(
    bert_large_op,
    batch,
    in0_dtype,
    in1_dtype,
    bias_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
):
    gelu_activation = False

    if bert_large_op == ttl.tensor.bert_large_fused_qkv_matmul:
        a_shape = [batch, 1, 384, 1024]
        b_shape = [1, 1, 1024, 3072]
        bias_shape = [1, 1, 1, 3072]
        bias_pad_shape = [1, 1, 32, 3072]
        expected_output_shape = [batch, 1, 384, 3072]

    elif bert_large_op == ttl.tensor.bert_large_ff1_matmul:
        if (
            in0_dtype == ttl.tensor.DataType.BFLOAT16
            and in1_dtype == ttl.tensor.DataType.BFLOAT16
            and out_dtype == ttl.tensor.DataType.BFLOAT16
            and out_mem_config.buffer_type == ttl.tensor.BufferType.L1
            and (
                in0_mem_config.buffer_type == ttl.tensor.BufferType.L1
                or in1_mem_config.buffer_type == ttl.tensor.BufferType.L1
            )
        ):
            pytest.skip("Skipping test since these tensors won't fit on device!")

        gelu_activation = True
        a_shape = [batch, 1, 384, 1024]
        b_shape = [1, 1, 1024, 4096]
        bias_shape = [1, 1, 1, 4096]
        bias_pad_shape = [1, 1, 32, 4096]
        expected_output_shape = [batch, 1, 384, 4096]

    elif bert_large_op == ttl.tensor.bert_large_ff2_matmul:
        a_shape = [batch, 1, 384, 4096]
        b_shape = [1, 1, 4096, 1024]
        bias_shape = [1, 1, 1, 1024]
        bias_pad_shape = [1, 1, 32, 1024]
        expected_output_shape = [batch, 1, 384, 1024]

    elif bert_large_op == ttl.tensor.bert_large_selfout_matmul:
        a_shape = [batch, 1, 384, 1024]
        b_shape = [1, 1, 1024, 1024]
        bias_shape = [1, 1, 1, 1024]
        bias_pad_shape = [1, 1, 32, 1024]
        expected_output_shape = [batch, 1, 384, 1024]

    else:
        raise NotImplementedError(f"bert_large matmul op is undefined!")

    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    BIAS = torch.randint(-20, 20, bias_shape, dtype=torch.float)

    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            in0_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            in1_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )

    if bias_mem_config is not None:
        bias_t = (
            ttl.tensor.Tensor(
                BIAS.flatten().tolist(),
                bias_shape,
                bias_dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .pad(bias_pad_shape, [0, 0, 0, 0], 0)
            .to(ttl.tensor.Layout.TILE)
            .to(device, bias_mem_config)
        )
    else:
        bias_t = None

    if bert_large_op == ttl.tensor.bert_large_ff1_matmul:
        t2 = bert_large_op(a_t, b_t, bias_t, gelu_activation, out_mem_config, out_dtype)
    else:
        t2 = bert_large_op(a_t, b_t, bias_t, out_mem_config, out_dtype)

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.dtype() == in1_dtype
    if bias_mem_config is not None:
        assert bias_t.memory_config().buffer_type == bias_mem_config.buffer_type
        assert bias_t.dtype() == bias_dtype
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    assert t2.dtype() == out_dtype
    logger.debug(
        f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.dtype()}"
    )
    logger.debug(
        f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.dtype()}"
    )
    if bias_mem_config is not None:
        logger.debug(
            f"bias ({bias_shape}): {bias_t.memory_config().buffer_type} and {bias_t.dtype()}"
        )
    logger.debug(
        f"out ({expected_output_shape}): {t2.memory_config().buffer_type} and {t2.dtype()}"
    )

    assert t2.shape() == expected_output_shape
    tt_host_rm = t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = torch.Tensor(tt_host_rm.data()).reshape(tt_host_rm.shape())

    ref_bmm = torch.matmul(A, B)
    if bias_mem_config is not None:
        ref_bmm = ref_bmm + BIAS
    if gelu_activation:
        ref_bmm = torch.nn.functional.gelu(ref_bmm)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")
    ttl.device.CloseDevice(device)
    assert passing_pcc


def run_bert_large_bmm_test(
    bert_large_op,
    batch,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
):
    if bert_large_op == ttl.tensor.bert_large_pre_softmax_bmm:
        a_shape = [batch, 16, 384, 64]
        b_shape = [batch, 16, 64, 384]
        expected_output_shape = [
            batch,
            1,
            16 * 384,
            384,
        ]  # No-op reshape from [batch, 16, 384, 384] in pre_softmax_bmm

    elif bert_large_op == ttl.tensor.bert_large_post_softmax_bmm:
        a_shape = [
            batch,
            1,
            16 * 384,
            384,
        ]  # No-op reshape to [batch, 16, 384, 384] in post_softmax_bmm
        b_shape = [batch, 16, 384, 64]
        expected_output_shape = [batch, 16, 384, 64]

    else:
        raise NotImplementedError(f"bert_large bmm op is undefined!")

    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            in0_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            in1_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )

    t2 = bert_large_op(a_t, b_t, out_mem_config, out_dtype)

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.dtype() == in1_dtype
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    assert t2.dtype() == out_dtype
    logger.debug(
        f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.dtype()}"
    )
    logger.debug(
        f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.dtype()}"
    )
    logger.debug(
        f"out ({expected_output_shape}): {t2.memory_config().buffer_type} and {t2.dtype()}"
    )

    assert t2.shape() == expected_output_shape
    tt_host_rm = t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = torch.Tensor(tt_host_rm.data()).reshape(tt_host_rm.shape())

    if bert_large_op == ttl.tensor.bert_large_pre_softmax_bmm:
        ref_bmm = torch.matmul(A, B).reshape(expected_output_shape)

    elif bert_large_op == ttl.tensor.bert_large_post_softmax_bmm:
        ref_bmm = torch.matmul(A.reshape([a_shape[0], 16, 384, 384]), B)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")
    ttl.device.CloseDevice(device)
    assert passing_pcc


# TODO: We could parametrize these separately for comprehensive testing
@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config",
    (
        (
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ),
        (
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
        ),
    ),
    ids=["DRAM", "L1"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "bias_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["bias_BFLOAT8_B", "bias_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in1_BFLOAT8_B", "in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in0_BFLOAT8_B", "in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch",
    (9, 8, 7),
    ids=[
        "batch_9",
        "batch_8",
        "batch_7",
    ],
)
@pytest.mark.parametrize(
    "bert_large_op",
    (
        ttl.tensor.bert_large_fused_qkv_matmul,
        ttl.tensor.bert_large_ff1_matmul,
        ttl.tensor.bert_large_ff2_matmul,
        ttl.tensor.bert_large_selfout_matmul,
    ),
    ids=["fused_qkv_bias", "ff1_bias_gelu", "ff2_bias", "selfout_bias"],
)
def test_bert_large_matmul(
    bert_large_op,
    batch,
    in0_dtype,
    in1_dtype,
    bias_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    request,
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_{request.node.callspec.id}"
    )
    run_bert_large_matmul_test(
        bert_large_op,
        batch,
        in0_dtype,
        in1_dtype,
        bias_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        bias_mem_config,
        out_mem_config,
    )


# TODO: We could parametrize these separately for comprehensive testing
@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, out_mem_config",
    (
        (
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ),
        (
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
        ),
    ),
    ids=["DRAM", "L1"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in1_BFLOAT8_B", "in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in0_BFLOAT8_B", "in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch",
    (9, 8, 7),
    ids=[
        "batch_9",
        "batch_8",
        "batch_7",
    ],
)
@pytest.mark.parametrize(
    "bert_large_op",
    (ttl.tensor.bert_large_pre_softmax_bmm, ttl.tensor.bert_large_post_softmax_bmm),
    ids=["pre_softmax_bmm", "post_softmax_bmm"],
)
def test_bert_large_bmm(
    bert_large_op,
    batch,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    request,
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_{request.node.callspec.id}"
    )
    run_bert_large_bmm_test(
        bert_large_op,
        batch,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
    )
