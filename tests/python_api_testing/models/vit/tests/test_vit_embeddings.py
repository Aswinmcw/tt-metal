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
from loguru import logger
import torch

from utility_functions_new import comp_allclose_and_pcc, comp_pcc
from tt.modeling_vit import ViTEmbeddings


def test_vit_embeddings(imagenet_sample_input, pcc=0.99):
    image = imagenet_sample_input

    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained(
            "google/vit-base-patch16-224"
        )

        state_dict = HF_model.state_dict()
        reference = HF_model.vit.embeddings

        config = HF_model.config
        HF_output = reference(image, None, None)

        tt_image = image
        tt_layer = ViTEmbeddings(
            config, base_address="vit.embeddings", state_dict=state_dict
        )

        tt_output = tt_layer(tt_image, None, None)
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
