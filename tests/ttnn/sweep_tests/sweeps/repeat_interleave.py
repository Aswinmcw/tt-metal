# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
import random
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

parameters = {
    "make_repeat_a_tensor": [False, True],
    "rank_of_tensor": [1, 2, 3, 4],
    "max_random_size_of_each_dim": [32],
    "dimension_to_repeat_on": [0, 1, 2, 3, 4, 5],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "dtype": [ttnn.bfloat16],
    "memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
}


def skip(rank_of_tensors, layout, **_) -> Tuple[bool, Optional[str]]:
    if rank_of_tensors < 2 and layout == ttnn.TILE_LAYOUT:
        return True, "Tile layout is only supported for tensors with rank >= 2"
    return False, None


def is_expected_to_fail(
    number_of_tensors, rank_of_tensors, dimension_to_concatenate_on, **_
) -> Tuple[bool, Optional[str]]:
    if number_of_tensors == 1:
        return True, "You must have at least two tensors to concat!"

    if dimension_to_concatenate_on >= rank_of_tensors:
        dimension_range = f"[{-rank_of_tensors}, {rank_of_tensors - 1}]"
        return (
            True,
            f"Dimension out of range (expected to be in range of {dimension_range}, but got {dimension_to_concatenate_on})",
        )

    return False, None


def run(
    dimensions_on_repeat_tensor,
    rank_of_tensors,
    max_random_size_of_each_dim,
    dimension_to_concatenate_on,
    layout,
    dtype,
    memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    random.seed(0)

    def get_size_of_dim(index):
        size_of_dim = random.randint(1, max_random_size_of_each_dim)
        if layout == ttnn.ROW_MAJOR_LAYOUT and index == rank_of_tensors - 1 and size_of_dim % 2 == 1:
            size_of_dim = (size_of_dim + 1) % max_random_size_of_each_dim
            if size_of_dim == 0:
                size_of_dim = 2
        return size_of_dim

    def calculate_input_shape():
        return [get_size_of_dim(index) for index in range(rank_of_tensors)]

    input_shape = calculate_input_shape()
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)

    if dimensions_on_repeat_tensor > 0:
        repeat = torch.tensor([1, 2])
    else:
        repeat = random.randint(1, max_random_size_of_each_dim)

    input_tensors = ttnn.from_torch(
        torch_input_tensor, device=device, layout=layout, dtype=dtype, memory_config=memory_config
    )
    output_tensor = ttnn.repeat_interleave(input_tensors, repeat, dim=dimension_to_concatenate_on)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_output_tensor = torch.repeat_interleave(torch_input_tensors, torch_repeat, dim=dimension_to_concatenate_on)
    return check_with_pcc(torch_output_tensor, output_tensor, 0.9999)
