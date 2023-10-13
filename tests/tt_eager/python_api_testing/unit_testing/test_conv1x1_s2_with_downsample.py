# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
import sys
import math
import numpy as np

import tt_lib as ttl
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import print_diff_argmax, is_close, comp_pcc, comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
    create_conv_bias_tensor,
    create_conv_weight_tensor_special_padding,
)
import torch

@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, stride_h, stride_w, num_cores",
    (
        #(20, 64, 64, 16, 16, 2, 2, 20),
        #(8, 64, 64, 56, 56, 1, 1, 98),
        (8, 64, 64, 56, 56, 2, 2, 98),
    ),
)
def test_run_downsample(
    use_program_cache,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    stride_h,
    stride_w,
    num_cores,
    device,
):

    assert(input_channels % 32 == 0)
    assert(output_channels % 32 == 0)
    assert(stride_h == stride_w)

    # torch.set_printoptions(threshold=10000)
    torch.manual_seed(0)
    a_activation_shape = [batch_size, input_channels, input_height, input_width]
    A_pyt = torch.normal(mean=0, std=0.1, size=a_activation_shape)
    b_weights_shape = [output_channels, input_channels, 1, 1]
    B_pyt = torch.normal(mean=0, std=0.1, size=b_weights_shape)

    output_height = math.ceil(input_height / stride_h)
    output_width = math.ceil(input_width / stride_w)

    conv_output_shape = [batch_size, output_height, output_width, output_channels]

    # Convert NCHW to NHWC shape
    A_pyt_nhwc = torch.permute(A_pyt, (0, 2, 3, 1))
    A_pyt_nhwc = A_pyt_nhwc.reshape(1, 1, batch_size*input_height*input_width, input_channels)
    #for i in range(2):
    #    for j in range(32):
    #        print(f"A_pyt_nhwc_2d[{i}][{j}]={A_pyt_nhwc[0][0][i][j]}")
    #print("A_pyt_nhwc_2d[32][0]=", A_pyt_nhwc[0][0][32][0])
    a_activation_shape_nhwc = [batch_size, input_height, input_width, input_channels]
    A_cl_host = ttl.tensor.Tensor(A_pyt_nhwc, ttl.tensor.DataType.BFLOAT16).reshape(1, 1, batch_size*input_height*input_width, input_channels)

    A_interleaved = A_cl_host.to(ttl.tensor.Layout.TILE).to(device, ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
        ))
    assert A_interleaved.shape()[0] == 1 and A_interleaved.shape()[1] == 1
    input_2d_height = A_interleaved.shape()[2]
    input_2d_width = A_interleaved.shape()[3]
    print("input_2d_height=", input_2d_height)
    print("input_2d_width=", input_2d_width)

    A_sharded = ttl.tensor.interleaved_to_sharded(
        A_interleaved, num_cores, [(int) (input_2d_height / num_cores), input_2d_width], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )
    # Prepare weights for simple matmul
    B_tiled_host = create_conv_weight_tensor(
        B_pyt, output_channels, input_channels, 1, 1, 1, 1
    )
    B_tiled = B_tiled_host.to(device)

    # downsample golden output using maxpool
    out_golden = torch.nn.functional.max_pool2d(A_pyt, 1, stride=stride_h)
    out_golden_2d_nhwc = torch.permute(out_golden, (0, 2, 3, 1)).reshape(1, 1, batch_size*output_height*output_width, input_channels)
    # for i in range(1):
    # #i = 1
    #     for j in range(32):
    #         print(f"out_golden_2d_nhwc[{i}][{j}]={out_golden_2d_nhwc[0][0][i][j]}")

    downsample_params = [batch_size, input_height, input_width, stride_h, stride_w]
    sharded_memory_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
            )
    # Run downsample op
    A_downampled_sharded = ttl.tensor.downsample(A_sharded, downsample_params, sharded_memory_config)
    A_downsampled = ttl.tensor.sharded_to_interleaved(A_downampled_sharded, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1))
    out = A_downsampled
    out_shape = [1, 1, batch_size*output_height*output_width, input_channels]
    assert out_shape == out.shape()
    out = ttl.tensor.format_output_tensor(out, out.shape(), device, ttl.tensor.Layout.ROW_MAJOR)
    out = out.reshape(batch_size, output_height, output_width, input_channels)
    out = out.cpu()
    assert out.layout() == ttl.tensor.Layout.ROW_MAJOR

    # Copy output to host and convert tt tensor to pytorch tensor
    out_result = out.to_torch().float()
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    # # Compare against golden
    # assert out_result.shape == out_golden.shape
    # [output_N, output_C, output_H, output_W] = out_result.shape
    # print("Golden - ")
    # print(out_golden.flatten())
    # print("Result - ")
    # print(out_result.flatten())
    # num_errors = 0
    # for n in range(output_N):
    #     for c in range(output_C):
    #         for h in range(output_H):
    #             for w in range(output_W):
    #                 calculated = torch.tensor(out_result[n][c][h][w])
    #                 golden = torch.tensor(out_golden[n][c][h][w])
    #                 atol_delta = torch.abs(golden - calculated).item()
    #                 rtol_delta = torch.abs(golden - calculated) / torch.abs(calculated)
    #                 if atol_delta > 0.1 or rtol_delta > 0.1:
    #                     print(f"Bad value at {n},{c},{h},{w} with ATOL={atol_delta} and RTOL={rtol_delta}")
    #                     print(f"    result={calculated}, golden={golden}")
    #                     num_errors += 1

    # print("Num of errors - ", num_errors)

    passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(out_golden, out_result, rtol=1e-1, atol=1e-3, pcc=0.9999)  # For LowFi we need 0.99976
    print("Passing=", passing_allclose_and_pcc)
    print("Output info=", output_info)
    passing_pcc_ds, _ = comp_pcc(out_golden, out_result, pcc=0.9998) # For LowFi we need 0.99976
    return
    assert passing_pcc_ds

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(
        A_pyt, B_pyt, stride=(stride_h, stride_w)
    )

    # Run regular matmul
    out = ttl.tensor.matmul(A_downsampled, B_tiled)
    out_shape = [1, 1, batch_size*output_height*output_width, output_channels]
    assert out_shape == out.shape()
    out = ttl.tensor.format_output_tensor(out, out.shape(), device, ttl.tensor.Layout.ROW_MAJOR)
    out = out.reshape(batch_size, output_height, output_width, output_channels)
    out = out.cpu()
    assert out.layout() == ttl.tensor.Layout.ROW_MAJOR

    # Copy output to host and convert tt tensor to pytorch tensor
    out_result = out.to_torch().float()
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    torch.set_printoptions(
        precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32
    )

    # print(f'OUT: {out_result}')
    # print(f'GLD: {out_golden}')

    # Compare against golden
    assert out_result.shape == out_golden.shape
    [output_N, output_C, output_H, output_W] = out_result.shape
    #print("Golden - ")
    #print(out_golden.flatten())
    #print("Result - ")
    #print(out_result.flatten())
    # for n in range(output_N):
    #     for c in range(output_C):
    #         for h in range(output_H):
    #             for w in range(output_W):
    #                 calculated = torch.tensor(out_result[n][c][h][w])
    #                 golden = torch.tensor(out_golden[n][c][h][w])
    #                 atol_delta = torch.abs(golden - calculated).item()
    #                 rtol_delta = torch.abs(golden - calculated) / torch.abs(calculated)
    #                 if atol_delta > 0.1 or rtol_delta > 0.1:
    #                     print(f"Bad value at {n},{c},{h},{w} with ATOL={atol_delta} and RTOL={rtol_delta}")
    #                     print(f"    result={calculated}, golden={golden}")

    passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(out_golden, out_result, rtol=1e-1, atol=1e-3, pcc=0.9999)  # For LowFi we need 0.99976
    print("Passing=", passing_allclose_and_pcc)
    print("Output info=", output_info)
    passing_pcc, _ = comp_pcc(out_golden, out_result, pcc=0.9998) # For LowFi we need 0.99976
    assert passing_pcc
    #assert passing_allclose_and_pcc
