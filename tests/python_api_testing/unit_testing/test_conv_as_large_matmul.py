import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
from tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, _nearest_y, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.python_api_testing.conv.conv_unit_test_utils import create_conv_act_tensor, create_conv_weight_tensor
import torch

@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # # resnet18 convs
        #(64, 3, 224, 224, 7, 7, 2, 2, 3, 3),
        #(64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        # #K=128 C=64 H=56 W=56 R=3 S=3 U=2 V=2 PH=1 PW=1 dilation=1 groups=1
        # (128, 64, 56, 56, 3, 3, 2, 2, 1, 1),
        # #K=128 C=128 H=28 W=28 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (128, 64, 28, 28, 3, 3, 1, 1, 1, 1),
        # #K=128 C=64 H=56 W=56 R=1 S=1 U=2 V=2 PH=0 PW=0 dilation=1 groups=1
        # (128, 64, 56, 56, 1, 1, 2, 2, 0, 0),
        # #K=128 C=128 H=28 W=28 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (128, 128, 28, 28, 3, 3, 1, 1, 1, 1),
        # #K=256 C=128 H=28 W=28 R=3 S=3 U=2 V=2 PH=1 PW=1 dilation=1 groups=1
        # (256, 128, 28, 28, 3, 3, 2, 2, 1, 1),
        # #K=256 C=256 H=14 W=14 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # #K=256 C=128 H=28 W=28 R=1 S=1 U=2 V=2 PH=0 PW=0 dilation=1 groups=1
        # (256, 128, 28, 28, 1, 1, 2, 2, 0, 0),
        # #K=256 C=256 H=14 W=14 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # #K=512 C=256 H=14 W=14 R=3 S=3 U=2 V=2 PH=1 PW=1 dilation=1 groups=1
        # (512, 256, 14, 14, 3, 3, 2, 2, 1, 1),
        # #K=512 C=512 H=7 W=7 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),

        # channels = 3 padding
        (32, 3, 5, 5, 1, 1, 1, 1, 0, 0),
        # w/ conv padding
        (32, 32, 5, 5, 1, 1, 1, 1, 1, 1),
        # Hat = 1, Wat = 1, Wbt = 1
        (32, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 1, Wbt = 1
        (32, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 2, Wbt = 1
        (32, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 2, Wbt = 1
        (32, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 1, Wbt = 2
        (64, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 2, Wbt = 2
        (64, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 1, Wbt = 2
        (64, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 2, Wbt = 2
        (64, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 8, Wat = 8, Wbt = 8
        (8*32, 8*32, 16, 16, 1, 1, 1, 1, 0, 0),
    ),
)
def test_run_conv_as_large_matmul(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    #torch.set_printoptions(threshold=10000)
    torch.manual_seed(0)
    a_activation_shape = [1,C,H,W]
    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    b_weights_shape = [K,C,R,S]
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()

    # Parameters to define block dims
    act_block_h = 4
    act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    out_subblock_h = 4
    out_subblock_w = 2

    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1
    mm_output_shape = [1,1,_nearest_y(OH*OW, 32*act_block_h),_nearest_y(K, 32*weight_block_w)]

    # Prepare activations
    A_cl_host = create_conv_act_tensor(A_pyt, 1, C, H, W)
    A = A_cl_host.to(device)

    # Prepare weights
    B_tiled_host = create_conv_weight_tensor(B_pyt, K, C, R, S, weight_block_h, weight_block_w)
    B_tiled = B_tiled_host.to(device)
    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w))

    untilize_out = True
    # Run TT metal OP
    out = ttl.tensor.conv(A, B_tiled, [R,S,stride_h,stride_w,pad_h,pad_w], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w)
    out = out.to(host)
    assert(out.shape() == mm_output_shape)
    if not untilize_out:
        # untilize
        out = out.to(ttl.tensor.Layout.ROW_MAJOR)
    # Copy output to host and convert tt tensor to pytorch tensor
    out_pytorch_padded = torch.tensor(out.data()).reshape(mm_output_shape)
    # remove padding
    out_pytorch = out_pytorch_padded[:, :, 0 : (OH * OW), 0 : K]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Compare against golden
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
    ttl.device.CloseDevice(device)
