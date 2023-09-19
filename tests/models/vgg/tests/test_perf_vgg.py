# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torchvision import models
import torch
from datasets import load_dataset
from loguru import logger
import pytest
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from models.utility_functions import prep_report, Profiler

from tests.models.vgg.tt.vgg import *

BATCH_SIZE = 1

def run_perf_vgg(imagenet_sample_input, expected_inference_time, expected_compile_time, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "16"

    image = imagenet_sample_input

    tt_image = tt_lib.tensor.Tensor(
        image.reshape(-1).tolist(),
        get_shape(image.shape),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    tt_vgg = vgg16(device, disable_conv_on_tt_device=True)

    torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_vgg.eval()

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_vgg(tt_image)
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_vgg(tt_image)
        tt_lib.device.Synchronize()
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_report(
        model_name="VGG",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    logger.info(f"vgg inference time: {second_iter_time}")
    logger.info(f"vgg compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (3.1,
         14,
        ),
    ),
)
def test_perf_bare_metal(use_program_cache, imagenet_sample_input, expected_inference_time, expected_compile_time, device):
    run_perf_vgg(imagenet_sample_input, expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (5.3,
         15,
        ),
    ),
)
def test_perf_virtual_machine(use_program_cache, imagenet_sample_input, expected_inference_time, expected_compile_time, device):
    run_perf_vgg(imagenet_sample_input, expected_inference_time, expected_compile_time, device)
