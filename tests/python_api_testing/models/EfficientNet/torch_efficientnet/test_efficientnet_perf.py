from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import tt_lib
import torch
import pytest
from loguru import logger
from torchvision import models

from sweep_tests.comparison_funcs import comp_pcc
from efficientnet import efficientnet_v2_s
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
)

_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_efficient_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    
    disable_compile_cache()

    with torch.no_grad():
        torch_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        torch_model.eval()
        
        profiler.enable()
        profiler.start("\nExec time of reference model") 
        torch_output = torch_model(image)
        profiler.end("\nExec time of reference model")
        
        state_dict = torch_model.state_dict()

        tt_model = efficientnet_v2_s(state_dict)
        tt_model.eval()

        #logger.debug(tt_model)
        
        profiler.start("\nExecution time of tt_EffNetV2 first run")
        tt_output = tt_model(image)
        profiler.end("\nExecution time of tt_EffNetV2 first run") 
        enable_compile_cache()
        
        PERF_CNT = 5 
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_EffNetV2 model")
            tt_output = tt_model(image) 
            profiler.end("\nAverage execution time of tt_EffNetV2 model")
            
        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]

    profiler.print()

    logger.info(f"PASSED {passing[1]}")
