import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import blocked_mm_with_conv_act, tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from python_api_testing.conv.pytorch_conv_tb import TestLevel, generate_conv_tb_with_pytorch_golden, generate_conv_tb

import torch

@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        (32, 32, 10, 10, 3, 3, 1, 1, 0, 0),
        #(64,64,14,14,3,3,1,1,1,1),
        #resnet 18 convs
        #(256, 128, 28, 28, 3, 3, 2, 2, 1, 1),
        #(256, 256, 14, 14, 3, 3, 1, 1, 1, 1,),
        #lenet conv
        #(16, 6, 14, 14, 5, 5, 1, 1, 0, 0),
    ),
)
def test_run_conv_as_large_matmul_cpu(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):
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

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    A_cl_data = A_cl.data()
    # Call DTX pass to transform A
    matrix_activation_h = (int) (_nearest_32(OH*OW) / 32)
    matrix_weight_w = (int) (_nearest_32(K) / 32)
    matrix_activation_w = (int) (_nearest_32(C*R*S)/32)
    (num_blocks_h,num_blocks_w,_,_,report_string) = ttl.tensor.compute_conv_op_block_info(matrix_activation_h, matrix_activation_w, matrix_weight_w)
    if report_string != "pass":
        print(report_string)
        assert False
    #if num_blocks != 2:
    #    print(str(num_blocks))
    #    assert False
    dim_order = [0,1,2]
    assert _nearest_32(OH*OW) % num_blocks_h == 0
    assert _nearest_32(C*R*S) % num_blocks_w == 0
    block_height = (int) (_nearest_32(OH*OW)/num_blocks_h)
    block_width = (int) (_nearest_32(C*R*S)/num_blocks_w)
    block_shape_yx = [block_height, block_width]
    mm_input_shape = [num_blocks_h*num_blocks_w, block_height, block_width]
    mm_weight_shape = [num_blocks_h*num_blocks_w, block_width, _nearest_32(K)]
    address_map = ttl.dtx.generate_address_map(ttl.dtx.conv_transform([C,H,W], [R,S,stride_h,stride_w,pad_h,pad_w], (dim_order,block_shape_yx)))

    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    B_rm = B_tiled_.to(ttl.tensor.Layout.ROW_MAJOR)
    assert(B_rm.shape() == [1, 1, _nearest_32(C*R*S), _nearest_32(K)])
    B_data = B_rm.data()
    B_pytorch_tensor = torch.tensor(B_data).reshape(mm_weight_shape)

    # Run pytorch matmul
    print("matmul weight shape - " + str(B_pytorch_tensor.shape))
    #out_pytorch = torch.matmul(A_transformed_pytorch_tensor, B_pytorch_tensor).reshape(mm_output_shape)
    # Run host side CPU function
    out_pytorch = blocked_mm_with_conv_act(A_cl_data, B_pytorch_tensor, mm_output_shape, address_map, num_blocks_h, num_blocks_w, _nearest_32(OH*OW), block_width)
    assert(list(out_pytorch.shape) == mm_output_shape)
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), 0 : K]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
