# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import SourceFileLoader
from tests.ttnn.sweep_tests.sweep import (
    SWEEP_SOURCES_DIR,
    SWEEP_RESULTS_DIR,
    permutations,
    run_single_test,
    preprocess_parameter_value,
)
from loguru import logger
from dataclasses import dataclass, field
from types import ModuleType
import pandas as pd
import pytest
import os


@dataclass
class SweepTest:
    file_name: str
    results_filename: str
    sweep_module: ModuleType
    parameter_list: list
    sweep_test_index: int
    passed: int = field(default=0)
    failed: int = field(default=0)
    skipped: int = field(default=0)
    crashed: int = field(default=0)

    def __str__(self):
        return f"{os.path.basename(self.file_name)}-{self.sweep_test_index}"


sweep_tests = []
for file_name in sorted(SWEEP_SOURCES_DIR.glob("*.py")):
    logger.info(f"Running {file_name}")
    base_name = os.path.basename(file_name)
    base_name = os.path.splitext(base_name)[0]
    sweep_module = SourceFileLoader(f"sweep_module_{base_name}", str(file_name)).load_module()
    base_name = base_name + ".csv"
    results_filename = SWEEP_RESULTS_DIR / base_name
    for sweep_test_index, parameter_list in enumerate(permutations(sweep_module.parameters)):
        parameter_list = {key: preprocess_parameter_value(value) for key, value in parameter_list.items()}
        sweep_tests.append(SweepTest(file_name, results_filename, sweep_module, parameter_list, sweep_test_index))


@pytest.mark.parametrize("sweep_test", sweep_tests, ids=str)
def test_all_sweeps(device, sweep_test):
    status, message = run_single_test(
        sweep_test.file_name,
        sweep_test.sweep_test_index,
        device=device,
    )

    assert status not in {"failed", "crashed"}, f"{message} - {sweep_test.parameter_list}"
