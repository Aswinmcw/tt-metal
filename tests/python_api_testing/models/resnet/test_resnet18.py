from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from loguru import logger
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import pytest
from resnetBlock import ResNet, BasicBlock
from libs import tt_lib as ttl

from utility_functions import comp_allclose_and_pcc, comp_pcc
batch_size=1

@pytest.mark.parametrize("fold_batchnorm", [False, True], ids=['Batchnorm not folded', "Batchnorm folded"])
def test_run_resnet18_inference(fold_batchnorm, imagenet_sample_input):
    print("Start")
    image = imagenet_sample_input
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device
        print("Create device")
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        print("Initialize device")
        ttl.device.InitializeDevice(device)
        print("Get host")
        host = ttl.device.GetHost()
        print("HERE")
        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        torch_resnet.eval()

        state_dict = torch_resnet.state_dict()
        print("Resnet")
        tt_resnet18 = ResNet(BasicBlock, [2, 2, 2, 2],
                        device=device,
                        host=host,
                        state_dict=state_dict,
                        base_address="",
                        fold_batchnorm=fold_batchnorm)

        torch_output = torch_resnet(image).unsqueeze(1).unsqueeze(1)
        print("Starting resnet18 on GS + CPU")
        tt_output = tt_resnet18(image)

        print(comp_allclose_and_pcc(torch_output, tt_output))
        passing, info = comp_pcc(torch_output, tt_output)
        logger.info(info)
        assert passing
