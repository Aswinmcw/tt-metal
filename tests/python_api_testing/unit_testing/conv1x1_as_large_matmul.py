from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from pymetal import ttlib as ttl
from pymetal.ttlib.utils import tilize_to_list, channels_last, convert_weights_2d_matrix, untilize
from python_api_testing.models.utility_functions import is_close
import torch

def run_1x1conv_test(K, C, H, W, untilize_out, use_single_bank_reader, matmul_blocked):
    #torch.manual_seed(0)
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,1,1]
    mm_output_shape = [1,1,H*W,K]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_cl = channels_last(A_pyt)
    A = ttl.tensor.Tensor(
        torch.flatten(A_cl).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
        ttl.tensor.MemoryConfig(False, 0) if use_single_bank_reader else ttl.tensor.MemoryConfig(True, -1)
        )

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_matrix = convert_weights_2d_matrix(B_pyt, b_weights_shape)
    assert(B_matrix.shape[0] == 1 and B_matrix.shape[1] == 1)
    assert(B_matrix.shape[2] == C and B_matrix.shape[3] == K)
    B_t = ttl.tensor.Tensor(
        tilize_to_list(B_matrix),
        B_matrix.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
        ttl.tensor.MemoryConfig(False, 0) if use_single_bank_reader else ttl.tensor.MemoryConfig(True, -1)
        )

    # Run TT metal OP
    if matmul_blocked:
        out = ttl.tensor.conv_as_large_bmm_single_core(A, B_t, untilize_out)
    else:
        out = ttl.tensor.conv_as_large_bmm_single_core_single_block(A, B_t, untilize_out, use_single_bank_reader)

    assert(out.shape() == mm_output_shape)
    out_pytorch = torch.tensor(out.to(host).data()).reshape(mm_output_shape)
    if not untilize_out:
        out_pytorch = untilize(out_pytorch)
    OH = H
    OW = W
    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt)
    assert(out_result.shape == out_golden.shape)
    maxmag = out_golden.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(out_result, out_golden, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())
    assert match.item()

if __name__ == "__main__":
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Conv activation shape
    # Conv as large matmul only works for certain values
    # hard code for now
    K = 128
    C = 128
    H = 16
    W = 16
    # Run simple conv single block + single DRAM bank
    #run_1x1conv_test(K,C,H,W,False,False,False)
    H = 32
    # Run conv using blocked matmul. Use multi bank reader.
    run_1x1conv_test(K,C,H,W,False,False,True)
    print("ALL PASSED!")
    ttl.device.CloseDevice(device)
