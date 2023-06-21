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
import timm
import pytest
from loguru import logger

from sweep_tests.comparison_funcs import comp_pcc
from inception import InceptionV4
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
)

_batch_size = 1


@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_inception_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    
    disable_compile_cache()

    with torch.no_grad():
        torch_model = timm.create_model('inception_v4', pretrained=True)
        torch_model.eval()

        profiler.enable()
        profiler.start("\nExec time of reference model")
        torch_output = torch_model(image)
        profiler.end("\nExec time of reference model")
        

        tt_model = InceptionV4(state_dict=torch_model.state_dict())
        tt_model.eval()

        profiler.start("\nExecution time of tt_InceptionV4 first run")
        tt_output = tt_model(image)
        profiler.end("\nExecution time of tt_InceptionV4 first run") 
        enable_compile_cache()
        
        PERF_CNT = 5 
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_InceptionV4 model")
            tt_output = tt_model(image) 
            profiler.end("\nAverage execution time of tt_InceptionV4 model")
            
        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]
        
        profiler.print()

    logger.info(f"PASSED {passing[1]}")
