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

import tt_lib
from utility_functions_new import comp_allclose_and_pcc, comp_pcc, torch_to_tt_tensor_rm, tt_to_torch_tensor
from tt.modeling_vit import TtViTModel
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
)

def test_vit_model(imagenet_sample_input, pcc=0.95):
    image = imagenet_sample_input
    head_mask = None
    output_attentions = None
    output_hidden_states = None
    interpolate_pos_encoding = None
    return_dict = None
    
    disable_compile_cache()
    
    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained("google/vit-base-patch16-224")

        state_dict = HF_model.state_dict()

        reference = HF_model.vit

        config = HF_model.config
        
        profiler.enable()
        profiler.start("\nExec time of reference model")
        HF_output = reference(image, head_mask, output_attentions, output_hidden_states, interpolate_pos_encoding, return_dict)[0]
        profiler.end("\nExec time of reference model")
        
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        tt_image = torch_to_tt_tensor_rm(image, device, put_on_device=False)
        tt_layer = TtViTModel(config, add_pooling_layer=False, base_address="vit", state_dict=state_dict, device=device)
        tt_layer.get_head_mask = reference.get_head_mask
        
        profiler.start("\nExecution time of tt_ViT first run")
        tt_output = tt_layer(tt_image, head_mask, output_attentions, output_hidden_states, interpolate_pos_encoding, return_dict)[0]
        profiler.end("\nExecution time of tt_ViT first run") 
        enable_compile_cache()
        
        PERF_CNT = 5 
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_ViT model")
            tt_output = tt_layer(tt_image, head_mask, output_attentions, output_hidden_states, interpolate_pos_encoding, return_dict)[0] 
            profiler.end("\nAverage execution time of tt_ViT model")
            
            
        tt_output = tt_to_torch_tensor(tt_output, host).squeeze(0)
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert(
            pcc_passing
        ), f"Model output does not meet PCC requirement {pcc}."
        
        profiler.print()
