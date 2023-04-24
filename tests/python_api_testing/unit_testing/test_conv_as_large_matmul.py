import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
import torch

@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        (32, 64, 5, 5, 5, 5, 1, 1, 1, 1,),
        #(32, 1024, 8, 4, 1, 1),
        #(32, 32, 18, 18, 3, 3),
        # (32, 32, 10, 10, 1, 1),
        # (32, 32, 10, 10, 3, 3),
        #(64, 64, 32, 16, 1, 1),
        #(64, 64, 10, 10, 1, 1),
        #(32, 64, 10, 10, 3, 3),
    ),
)
def test_run_conv_as_large_matmul(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):
    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    # check if params are valid
    assert (H - R + 2 * pad_h) >= 1 and (W - S + 2 * pad_w) >= 1
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1
    mm_output_shape = [1,1,_nearest_32(OH*OW),K]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)
    A = A_cl.to(device, ttl.tensor.MemoryConfig(False, 0))

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    B_tiled = B_tiled_.to(device)

    # Run TT metal OP
    out = ttl.tensor.conv_as_large_bmm_single_core(A, B_tiled, [R,S,stride_h,stride_w,pad_h,pad_w])
    assert(out.shape() == mm_output_shape)
    # Copy output to host and convert tt tensor to pytorch tensor
    out_pytorch = torch.tensor(out.to(host).data()).reshape(mm_output_shape)
    ttl.device.CloseDevice(device)
    # remove padding
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), :]

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
