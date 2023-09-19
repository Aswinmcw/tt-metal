# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models_wip.EfficientNet.demo.demo_utils import run_gs_demo
from models_wip.EfficientNet.tt.efficientnet_model import efficientnet_v2_l


def test_gs_demo_v2_l():
    run_gs_demo(efficientnet_v2_l)
