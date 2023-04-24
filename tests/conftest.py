import pytest
import torch
import random
import os
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield


@pytest.fixture(scope="function")
def function_level_defaults(reset_seeds):
    yield


@pytest.fixture(scope="session")
def is_dev_env():
    return os.environ.get("TT_METAL_ENV", "") == "dev"


@pytest.fixture(scope="session")
def model_location_generator():
    def model_location_generator_(rel_path):
        internal_weka_path = Path("/mnt/MLPerf")
        has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

        if has_internal_weka:
            return Path("/mnt/MLPerf") / rel_path
        else:
            return Path("/opt/tt-metal-models") / rel_path

    return model_location_generator_

@pytest.fixture
def imagenet_sample_input(model_location_generator):
    sample_path = model_location_generator("/mnt/MLPerf/tt_dnn-models/samples/ILSVRC2012_val_00048736.JPEG")
    im = Image.open(sample_path)
    im = im.resize((224, 224))
    im = transforms.ToTensor()(im).unsqueeze(0)
    return im
