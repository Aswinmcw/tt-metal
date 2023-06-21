from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from loguru import logger
import torch
import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)
from utility_functions_new import comp_pcc
from reference.ssd_head import ssdlite320_mobilenet_v3_large
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
)

@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssdlite320_mobilenet_v3_large_inference(pcc, imagenet_sample_input):
    image = imagenet_sample_input
    
    disable_compile_cache()
        
    with torch.no_grad():
        # Pretrained Torchvision model
        TV_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        TV_model.eval()

        # Pytorch CPU call
        PT_model = ssdlite320_mobilenet_v3_large()
        PT_model.eval()

        with torch.no_grad():
            profiler.enable()
            profiler.start("\nExec time of reference model")
            TV_output = TV_model(image)
            profiler.end("\nExec time of reference model")
            
            profiler.start("\nExecution time of tt_SSDlite_mobv3 first run")
            PT_output = PT_model(image)
            profiler.end("\nExecution time of tt_SSDlite_mobv3 first run") 
            
            enable_compile_cache()
        
            PERF_CNT = 5 
            for i in range(PERF_CNT):
                profiler.start("\nAverage execution time of tt_SSDlite_mobv3 model")
                PT_output = PT_model(image)                
                profiler.end("\nAverage execution time of tt_SSDlite_mobv3 model")

        passing_scores = comp_pcc(TV_output[0]["scores"], PT_output[0]["scores"], pcc)
        passing_labels = comp_pcc(TV_output[0]["labels"], PT_output[0]["labels"], pcc)
        passing_boxes = comp_pcc(TV_output[0]["boxes"], PT_output[0]["boxes"], pcc)

        assert passing_scores[0], passing_scores[1:]
        assert passing_labels[0], passing_labels[1:]
        assert passing_boxes[0], passing_boxes[1:]

        logger.info(f"ssd scores PASSED {passing_scores[1]}")
        logger.info(f"ssd labels PASSED {passing_labels[1]}")
        logger.info(f"ssd boxes  PASSED  {passing_boxes[1]}")
        
        profiler.print()
