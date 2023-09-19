# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.models_wip.t5.tt.t5_for_conditional_generation import (
    t5_small_for_conditional_generation,
)
from models.models_wip.t5.demo.demo_utils import run_demo_t5


def test_demo_t5_small():
    run_demo_t5(t5_small_for_conditional_generation)
