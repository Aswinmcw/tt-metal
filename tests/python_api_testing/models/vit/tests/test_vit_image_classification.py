from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import ViTForImageClassification as HF_ViTForImageClassication
from transformers import AutoImageProcessor as HF_AutoImageProcessor
from loguru import logger
import torch
from datasets import load_dataset

import tt_lib
from utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from tt.modeling_vit import TtViTForImageClassification


def test_vit_image_classification(pcc=0.95):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained(
            "google/vit-base-patch16-224"
        )
        image_processor = HF_AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        inputs = image_processor(image, return_tensors="pt")

        reference = HF_model
        state_dict = HF_model.state_dict()

        config = HF_model.config
        HF_output = reference(**inputs).logits

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        tt_inputs = torch_to_tt_tensor_rm(
            inputs["pixel_values"], device, put_on_device=False
        )
        tt_model = TtViTForImageClassification(
            config, base_address="", state_dict=state_dict, device=device
        )
        tt_model.vit.get_head_mask = reference.vit.get_head_mask
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output, host).squeeze(0)[:, 0, :]
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
