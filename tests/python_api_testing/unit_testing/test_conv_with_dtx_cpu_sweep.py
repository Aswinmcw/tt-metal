import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import blocked_mm, tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from python_api_testing.conv.pytorch_conv_tb import TestLevel, generate_conv_tb_with_pytorch_golden, generate_conv_tb

import torch

def run_conv_as_large_matmul(conv_op_test_params, pytorch_inputs_and_golden):
    print("Testing convolution with following parameters - ")
    conv_op_test_params.print("   ")
    ctp = conv_op_test_params.conv_params
    N = ctp.act_shape[0]
    C = ctp.act_shape[1]
    H = ctp.act_shape[2]
    W = ctp.act_shape[3]
    K = ctp.weight_shape[0]
    assert(ctp.weight_shape[1] == C)
    R = ctp.weight_shape[2]
    S = ctp.weight_shape[3]
    stride_h = ctp.stride_h;
    stride_w = ctp.stride_w;
    pad_h = ctp.pad_h;
    pad_w = ctp.pad_w;
    # check if params are valid
    assert (H - R + 2 * pad_h) >= 1 and (W - S + 2 * pad_w) >= 1
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1

    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    mm_output_shape = [1,1,_nearest_32(OH*OW),_nearest_32(K)]

    A_pyt = pytorch_inputs_and_golden[0]
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)

    # Prepare weights
    B_pyt = pytorch_inputs_and_golden[1]
    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    if(conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE):
        return True
    assert(conv_op_test_params.test_level == TestLevel.OP_FULL_COMPUTE)
    A_cl_data = A_cl.data()
    # Call DTX pass to transform A
    matrix_activation_h = (int) (_nearest_32(OH*OW) / 32)
    matrix_weight_w = (int) (_nearest_32(K) / 32)
    matrix_activation_w = (int) (_nearest_32(C*R*S)/32)
    (num_blocks,_,_,report_string) = ttl.tensor.compute_conv_op_block_info(matrix_activation_h, matrix_activation_w, matrix_weight_w)
    if report_string != "pass":
        print(report_string)
        assert False
    dim_order = [0,1,2]
    assert _nearest_32(C*R*S) % num_blocks == 0
    block_width = (int) (_nearest_32(C*R*S)/num_blocks)
    block_shape_yx = [_nearest_32(OH*OW), block_width]
    mm_input_shape = [num_blocks, _nearest_32(OH*OW), block_width]
    mm_weight_shape = [num_blocks, block_width, _nearest_32(K)]
    A_transformed_data = ttl.dtx.evaluate(A_cl_data, ttl.dtx.conv_transform([C,H,W], [R,S,stride_h,stride_w,pad_h,pad_w], (dim_order,block_shape_yx)))
    A_transformed_pytorch_tensor = torch.tensor(A_transformed_data).reshape(mm_input_shape)

    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    B_rm = B_tiled_.to(ttl.tensor.Layout.ROW_MAJOR)
    assert(B_rm.shape() == [1, 1, _nearest_32(C*R*S), _nearest_32(K)])
    B_data = B_rm.data()
    B_pytorch_tensor = torch.tensor(B_data).reshape(mm_weight_shape)


    # Run pytorch matmul
    print("matmul input shape - " + str(A_transformed_pytorch_tensor.shape))
    print("matmul weight shape - " + str(B_pytorch_tensor.shape))
    #out_pytorch = torch.matmul(A_transformed_pytorch_tensor, B_pytorch_tensor).reshape(mm_output_shape)
    out_pytorch = blocked_mm(A_transformed_pytorch_tensor, B_pytorch_tensor)
    assert(list(out_pytorch.shape) == mm_output_shape)
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), 0 : K]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Compare against pytorch golden result
    out_golden = pytorch_inputs_and_golden[2]
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    return passing_pcc

def test_sweep_conv():
    test_bench = generate_conv_tb()
    pytorch_conv_golden_tb = generate_conv_tb_with_pytorch_golden(test_bench)
    passing = True
    full_op_compute_passing_tests = []
    input_tensor_only_passing_tests = []
    failing_tests = []
    for conv_op_test_params, pytorch_inputs_and_golden in pytorch_conv_golden_tb.items():
        passing_ = run_conv_as_large_matmul(conv_op_test_params, pytorch_inputs_and_golden)
        if passing_:
            if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
                input_tensor_only_passing_tests.append(conv_op_test_params)
            else:
                full_op_compute_passing_tests.append(conv_op_test_params)
        else:
            failing_tests.append(conv_op_test_params)
            print("Failed test - ")
            conv_op_test_params.print("   ")
            assert(False)
        passing &= passing_
    print("Following tests that create only input tensors passed - ")
    for conv_op_test_params in input_tensor_only_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests that rull full op compute passed - ")
    for conv_op_test_params in full_op_compute_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests failed - ")
    for conv_op_test_params in failing_tests:
        conv_op_test_params.print("   ")
    print(str(len(input_tensor_only_passing_tests)) + " \"INPUT TENSORS CREATION\" tests PASSED.")
    print(str(len(full_op_compute_passing_tests)) + " \"FULL OP COMPUTE\" tests PASSED.")
    print(str(len(failing_tests)) + " \"FULL OP COMPUTE\" tests FAILED.")
    #assert passing
