from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")

import torch

from loguru import logger

import tt_lib
from utility_functions_new import comp_pcc, tt2torch_tensor
from mnist import *
import pytest


def test_mnist_convnet_inference(pcc):  # model location
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        torch_ConvNet, state_dict = load_torch("convnet_mnist.pt")
        test_dataset, test_loader = prep_data()
        first_input, label = next(iter(test_loader))

        tt_convnet = TtConvNet(device, host, state_dict)
        with torch.no_grad():
            img = first_input.to("cpu")
            # unsqueeze to go from [batch, 10] to [batch, 1, 1, 10]

            torch_output = torch_ConvNet(img).unsqueeze(1).unsqueeze(1)
            _, torch_predicted = torch.max(torch_output.data, -1)

            tt_image = tt_lib.tensor.Tensor(
                img.reshape(-1).tolist(),
                img.shape,
                tt_lib.tensor.DataType.BFLOAT16,
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            tt_output = tt_convnet(tt_image)
            tt_output = tt2torch_tensor(tt_output, host)

            pcc_passing, pcc_output = comp_pcc(torch_output, tt_output)
            logger.info(f"Output {pcc_output}")
            assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mnist_inference(pcc):
    test_mnist_convnet_inference(pcc)


if __name__ == "__main__":
    test_mnist_convnet_inference(0.99)
