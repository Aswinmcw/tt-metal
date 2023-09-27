# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torchvision import models
from loguru import logger
import pytest

import tt_lib
from tt_models.utility_functions import comp_pcc
from tt.vgg import vgg11
from vgg_utils import get_shape


_batch_size = 1


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_vgg11_inference(device, pcc, imagenet_sample_input):
    image = imagenet_sample_input

    batch_size = _batch_size
    with torch.no_grad():
        torch_vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        torch_vgg.eval()

        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg11(device, host, disable_conv_on_tt_device=True)

        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        tt_image = tt_lib.tensor.Tensor(
            image.reshape(-1).tolist(),
            get_shape(image.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        tt_output = tt_vgg(tt_image)

        tt_output = tt_output.cpu()
        tt_output = torch.Tensor(tt_output.to_torch()).reshape(tt_output.shape())

        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
