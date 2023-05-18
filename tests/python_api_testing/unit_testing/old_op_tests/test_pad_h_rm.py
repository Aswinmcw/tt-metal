import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
tt_lib.device.InitializeDevice(device)
host = tt_lib.device.GetHost()
tt_lib.device.StartDebugPrintServer(device)

if __name__ == "__main__":
    torch.manual_seed(123)
    N = 3
    C = 4
    H = 5
    HP = 64
    W = 32*5
    x = torch.randn((N,C,H,W))

    xt = tt_lib.tensor.Tensor(x.reshape(-1).tolist(), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.ROW_MAJOR, device)

    # test that reading back from row major is about the same (+/- BF16 conversion)
    xt_data = xt.to(host).data()
    tt_got_back_rm = torch.Tensor(xt_data).reshape((N,C,H,W))

    print("row_major read back max absdiff=")
    print_diff_argmax(tt_got_back_rm, x)

    # apply  h-padding
    xtp = tt_lib.tensor.pad_h_rm(xt, HP)
    assert(xtp.shape() == [N,C,HP,W])
    xtp_data = xtp.to(host).data()
    tt_got_back = torch.Tensor(xtp_data).reshape((N,C,HP,W))

    print("pad_h_rm() max absdiff=")
    padded_ref = torch.zeros(xtp.shape())
    padded_ref[:,:,0:H,:] = x
    print_diff_argmax(tt_got_back, padded_ref)

tt_lib.device.CloseDevice(device)
