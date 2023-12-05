# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import tt_lib as ttl


tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
    ttl.tensor.DataType.BFLOAT8_B: torch.float,
}
TILE_WIDTH = 32
TILE_HEIGHT = 32


def print_tiles(tiled_tensor, num_tiles_height, num_tiles_width):
    tile_torch_rows = torch.chunk(tiled_tensor, int(num_tiles_height), dim=2)
    row_idx = 0
    for row in tile_torch_rows:
        tiles = torch.chunk(row, int(num_tiles_width), dim=3)
        col_idx = 0
        for tile in tiles:
            tile_idx = row_idx * num_tiles_width + col_idx
            print("Trip Tile " + str(int(tile_idx)) + " with shape " + str(tile.shape))
            print(tile)
            col_idx = col_idx + 1
        row_idx = row_idx + 1


def compare_tiles(tiled_tensor_1, tiled_tensor_2, num_tiles_height, num_tiles_width):
    tile_torch_rows_1 = torch.chunk(tiled_tensor_1, int(num_tiles_height), dim=2)
    tile_torch_rows_2 = torch.chunk(tiled_tensor_2, int(num_tiles_height), dim=2)
    row_idx = 0
    for row_1 in tile_torch_rows_1:
        row_2 = tile_torch_rows_2[row_idx]
        tiles_1 = torch.chunk(row_1, int(num_tiles_width), dim=3)
        tiles_2 = torch.chunk(row_2, int(num_tiles_width), dim=3)
        col_idx = 0
        for tile_1 in tiles_1:
            tile_idx = row_idx * num_tiles_width + col_idx
            tile_2 = tiles_2[col_idx]
            passing = torch.allclose(tile_1, tile_2)
            if not passing:
                print("Tile " + str(tile_idx) + " not_matching ")
            col_idx = col_idx + 1
        row_idx = row_idx + 1


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        #        ttl.tensor.DataType.BFLOAT16,
        #        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, shard_scheme, shard_shape, compute_grid",
    [
        ([1, 4, 64, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (64, 64), (0, 3)),
        ([1, 1, 128, 128], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (64, 64), (0, 3)),
        ([1, 1, 2048, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (512, 64), None),
        ([1, 1, 2048, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 4096, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 8192, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 14336, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 256, 32], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (32, 32), None),
        ([1, 1, 128, 64], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (32, 64), None),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (2048, 32), None),
        ([1, 1, 2048, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (1024, 64), None),
        #        ([1, 1, 64, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (32, 64), None),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        #        ttl.tensor.ShardOrientation.COL_MAJOR
    ],
)
def test_tensor_conversion_between_torch_and_tt_tile(
    tt_dtype, device, tensor_shape, shard_scheme, shard_shape, compute_grid, shard_orientation
):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    if compute_grid == None:
        compute_grid = ttl.tensor.CoreCoord(
            device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
        )
    else:
        compute_grid = ttl.tensor.CoreCoord(compute_grid[0], compute_grid[1])

    shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    two_d_shape = (tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3])
    num_tiles_width = (two_d_shape[1]) / TILE_WIDTH
    num_tiles_height = (two_d_shape[0]) / TILE_HEIGHT
    num_shards = (two_d_shape[0] / shard_shape[0]) * (two_d_shape[1] / shard_shape[1])

    torch_tensor = None
    for row_idx in range(0, int(num_tiles_height)):
        tile_row = None
        for col_idx in range(0, int(num_tiles_width)):
            tile_idx = col_idx + num_tiles_width * row_idx
            tile = torch.full((1, 1, TILE_WIDTH, TILE_HEIGHT), tile_idx + 1, dtype=dtype)
            if tile_row == None:
                tile_row = tile
            else:
                tile_row = torch.cat((tile_row, tile), 3)
        if torch_tensor == None:
            torch_tensor = tile_row
        else:
            torch_tensor = torch.cat((torch_tensor, tile_row), 2)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype).to(ttl.tensor.Layout.TILE)
    mem_config = ttl.tensor.MemoryConfig(shard_scheme, ttl.tensor.BufferType.L1)
    tt_tensor = tt_tensor.to(device, mem_config, shard_spec)

    # tt_tensor_sharded = tt_tensor.cpu_sharded().to(ttl.tensor.Layout.ROW_MAJOR)
    # torch_tensor_after_round_trip_sharded = tt_tensor_sharded.to_torch()

    # for shard_id in range(0, int(num_shards)):
    #    print("shard " + str(int(shard_id)))
    #    shard_torch_tensor = tt_tensor.extract_shard(int(shard_id)).to_torch()
    #    print(shard_torch_tensor)

    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    torch_tensor_after_round_trip = tt_tensor.to_torch()
    # print_tiles(torch_tensor_after_round_trip, num_tiles_height, num_tiles_width)
    # compare_tiles(torch_tensor, torch_tensor_after_round_trip, num_tiles_height, num_tiles_width)

    # print("input tensor ")
    # print_tiles(torch_tensor, num_tiles_height, num_tiles_width)
    # print("torch_tensor_after_round_trip_sharded")
    # print_tiles(torch_tensor_after_round_trip_sharded, num_tiles_height, num_tiles_width)

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype, device, ttl.tensor.Layout.TILE, mem_config, shard_spec)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    torch_tensor_after_round_trip = tt_tensor.to_torch()

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        # ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, shard_scheme, shard_shape",
    [
        ([1, 1, 98, 256], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1, 256)),
        ([1, 1, 1, 1176], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1, 12)),
        #        ([1, 1, 64, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (32, 64), None),
    ],
)
def test_tensor_conversion_between_torch_and_tt_rm(tt_dtype, device, tensor_shape, shard_scheme, shard_shape):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    num_pages_width = tensor_shape[2] / shard_shape[0]
    num_pages_height = tensor_shape[3] / shard_shape[1]

    # num_pages_width = 98
    # num_pages_height = 1

    shard_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 7)),
            ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 8), ttl.tensor.CoreCoord(1, 8)),
        }
    )
    # shard_shape = (1, 256)

    # tensor_shape = [1, 1, 49000, 32]
    # tensor_shape = [1, 1, 98, 256]
    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    torch_tensor = None
    for row_idx in range(0, int(num_pages_width)):
        page_row = None
        for col_idx in range(0, int(num_pages_height)):
            page_idx = col_idx + num_pages_height * row_idx
            page = torch.full((1, 1, shard_shape[0], shard_shape[1]), page_idx, dtype=dtype)
            if page_row == None:
                page_row = page
            else:
                page_row = torch.cat((page_row, page), 3)
        if torch_tensor == None:
            torch_tensor = page_row
        else:
            torch_tensor = torch.cat((torch_tensor, page_row), 2)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)

    assert list(torch_tensor.size()) == tensor_shape

    mem_config = ttl.tensor.MemoryConfig(shard_scheme, ttl.tensor.BufferType.L1)
    tt_tensor = tt_tensor.to(device, mem_config, shard_spec)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing
