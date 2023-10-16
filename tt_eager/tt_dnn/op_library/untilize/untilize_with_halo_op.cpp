// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tensor/owned_buffer_functions.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

using range_t = std::array<int32_t, 2>;
const int32_t NEIGHBORHOOD_DIST = 2;    // => ncores to left and ncores to right

namespace untilize_with_halo_helpers {

struct PoolConfig {
    uint32_t in_w;
    uint32_t in_h;
    uint32_t out_w;
    uint32_t out_h;
    uint32_t stride_w;
    uint32_t stride_h;
    uint32_t pad_w;
    uint32_t pad_h;
    uint32_t window_w;
    uint32_t window_h;
    uint32_t dilation_w;
    uint32_t dilation_h;
};

range_t calculate_in_range(const range_t& out_range, const PoolConfig& pc) {
    // given out stick range, calculate corresponding window's center stick input coords
    range_t in_range;
    // start of the range
    {
        uint32_t out_w_i = out_range[0] % pc.out_w;
        uint32_t out_h_i = out_range[0] / pc.out_w;
        uint32_t in_w_i = out_w_i * pc.stride_w;
        uint32_t in_h_i = out_h_i * pc.stride_h;
        in_range[0] = in_h_i * pc.in_w + in_w_i;
    }
    // end of the range
    {
        uint32_t out_w_i = out_range[1] % pc.out_w;
        uint32_t out_h_i = out_range[1] / pc.out_w;
        // corresponding window's center stick input coords:
        uint32_t in_w_i = out_w_i * pc.stride_w;
        uint32_t in_h_i = out_h_i * pc.stride_h;
        in_range[1] = in_h_i * pc.in_w + in_w_i;
    }
    return in_range;
}

struct NewShardingConfig {
    int32_t first_partial_right_aligned_row_width;
    int32_t first_partial_image_num_rows;
    int32_t num_full_images;
    int32_t last_partial_image_num_rows;
    int32_t last_partial_left_aligned_row_width;
    int32_t skip_after_partial_right_aligned_row;
    int32_t skip_after_first_partial_image_row;
    int32_t skip_after_full_image;
};

inline NewShardingConfig get_shard_specs(int32_t start_stick, int32_t end_stick, const PoolConfig& pc, bool to_print = false) {
    int32_t nsticks_per_core = end_stick - start_stick;

    if (nsticks_per_core < 1) {
        return NewShardingConfig{
                .first_partial_right_aligned_row_width = 0,
                .first_partial_image_num_rows = 0,
                .num_full_images = 0,
                .last_partial_image_num_rows = 0,
                .last_partial_left_aligned_row_width = 0,
                .skip_after_partial_right_aligned_row = 0,
                .skip_after_first_partial_image_row = 0,
                .skip_after_full_image = 0
            };
    }
    if (nsticks_per_core < pc.in_w) {
        switch(nsticks_per_core) {
            case 1:
                // only one stick in the range. need to figure out if there are padding to be attached to it, and in which place
                int32_t in_w_i = start_stick % pc.in_w;
                int32_t in_h_i = (start_stick % (pc.in_w * pc.in_h)) % pc.in_w;
                if (in_w_i == pc.in_w - 1) {
                    // this is the last stick in the row
                    // there needs to be padding sticks attached
                    if (in_h_i % pc.in_h == pc.in_h - 1) {
                        // this is the last stick in the image
                        // there needs to be full padding rows
                        return NewShardingConfig{
                                .first_partial_right_aligned_row_width = 1,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 0,
                                .skip_after_partial_right_aligned_row = (int32_t) (pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0
                            };
                    } else {
                        // this is just the last stick in the row and not the image
                        // there are just width padding
                        return NewShardingConfig{
                                .first_partial_right_aligned_row_width = 1,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 0,
                                .skip_after_partial_right_aligned_row = 2 * (int32_t) pc.pad_w,
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0
                            };
                    }
                } else {
                    // this is just one stick without any padding
                    return NewShardingConfig{
                            .first_partial_right_aligned_row_width = 1,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = 0,
                            .skip_after_partial_right_aligned_row = 0,
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0
                        };
                }
            case 2:
                // two sticks in the range. figure out if there are any padding attached
                int32_t in_w_i = start_stick % pc.in_w;
                int32_t in_h_i = (start_stick % (pc.in_w * pc.in_h)) % pc.in_w;
                if (in_w_i == pc.in_w - 2) {
                    // these are two last sticks of the row
                    // there needs to be padding after
                    if (in_h_i == pc.in_h - 1) {
                        // these are the last sticks in the last row of the image
                        // insert full padding rows
                        return NewShardingConfig{
                                .first_partial_right_aligned_row_width = 2,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 0,
                                .skip_after_partial_right_aligned_row = (int32_t) (pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0
                            };
                    } else {
                        // just need width padding
                        return NewShardingConfig{
                                .first_partial_right_aligned_row_width = 2,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 0,
                                .skip_after_partial_right_aligned_row = 2 * (int32_t) pc.pad_w,
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0
                            };
                    }
                } else if (in_w_i == pc.in_w - 1) {
                    // these are one last stick of the row and one first stick of next row
                    // there needs to be padding in between
                    if (in_h_i == pc.in_h - 1) {
                        // these two sticks belong to different images
                        // insert full padding rows between them
                        return NewShardingConfig{
                                .first_partial_right_aligned_row_width = 1,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 1,
                                .skip_after_partial_right_aligned_row = (int32_t) (pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0
                            };
                    } else {
                        // just width padding between then
                        return NewShardingConfig{
                                .first_partial_right_aligned_row_width = 1,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 1,
                                .skip_after_partial_right_aligned_row = (int32_t) (2 * pc.pad_w),
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0
                            };
                    }
                } else {
                    // no padding needs to be attached
                    return NewShardingConfig{
                            .first_partial_right_aligned_row_width = 2,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = 0,
                            .skip_after_partial_right_aligned_row = 0,
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0
                        };
                }

            case 3:
            default:
                int32_t in_w_i = start_stick % pc.in_w;
                int32_t in_h_i = (start_stick % (pc.in_w * pc.in_h)) % pc.in_w;
                int32_t last_in_w_i = end_stick % pc.in_w;
                int32_t last_in_h_i = (end_stick % (pc.in_w * pc.in_h)) % pc.in_w;
                if (in_h_i == last_in_h_i) {
                    // all sticks belong to same row
                    if (last_in_w_i == pc.in_w - 1) {
                        // these sticks are the last in the row
                        // insert padding at end
                        // ...
                    } else {
                        // no padding is needed
                    }
                } else {
                    // sticks span two different rows. figure out where does the padding go.
                    // ...
                }
        }
    }

    if (to_print) log_debug("val: {}", 0);

    // First partial right-aligned row
    int32_t image_row_start_left_width = start_stick % pc.in_w;
    if (to_print) log_debug("image_row_start_left_width: {}", image_row_start_left_width);
    int32_t first_partial_right_aligned_row_width = image_row_start_left_width > 0 ? pc.in_w - image_row_start_left_width : 0;
    if (to_print) log_debug("first_partial_right_aligned_row_width: {}", first_partial_right_aligned_row_width);

    if (first_partial_right_aligned_row_width > nsticks_per_core) {
        return NewShardingConfig{
                .first_partial_right_aligned_row_width = nsticks_per_core,
                .first_partial_image_num_rows = 0,
                .num_full_images = 0,
                .last_partial_image_num_rows = 0,
                .last_partial_left_aligned_row_width = 0,
                .skip_after_partial_right_aligned_row = 0,
                .skip_after_first_partial_image_row = 0,
                .skip_after_full_image = 0
            };
    }

    // Last partial left-aligned row
    int32_t sticks_after_first_partial_row = nsticks_per_core - first_partial_right_aligned_row_width;
    if (to_print) log_debug("sticks_after_first_partial_row: {}", sticks_after_first_partial_row);
    int32_t last_partial_left_aligned_row_width = sticks_after_first_partial_row % pc.in_w;
    if (to_print) log_debug("last_partial_left_aligned_row_width: {}", last_partial_left_aligned_row_width);

    // Figure out how to allocate full image rows to first partial image, full images, or last partial image
    // This also affects skip after first_partial_right_aligned_row
    int32_t image_row_start_idx = start_stick / pc.in_w;
    int32_t image_row_start_idx_after_partial_right_aligned_row = (start_stick + first_partial_right_aligned_row_width) / pc.in_w;
    int32_t image_row_end_idx = end_stick / pc.in_w;
    int32_t image_start_idx = image_row_start_idx / pc.in_h;
    int32_t image_start_idx_after_partial_right_aligned_row = image_row_start_idx_after_partial_right_aligned_row / pc.in_h;
    int32_t image_start_height_after_partial_right_aligned_row = image_row_start_idx_after_partial_right_aligned_row % pc.in_h;
    int32_t image_end_idx = image_row_end_idx / pc.in_h;

    // Default case: We don't have a partial right aligned row; so we start with either full images, last partial image, or partial left aligned row
    int32_t skip_after_partial_right_aligned_row = 0;
    // Case: partial_right_aligned_row > 0 and completes an image
    if (first_partial_right_aligned_row_width > 0 && image_start_height_after_partial_right_aligned_row == 0) {
        skip_after_partial_right_aligned_row = pc.window_w - 1 + pc.pad_h * (pc.in_w + 2 * pc.pad_w);
    // Case: partial_right_aligned_row > 0 and doesn't complete an image
    } else if (first_partial_right_aligned_row_width > 0) {
        skip_after_partial_right_aligned_row = pc.window_w - 1;
    }

    int32_t first_partial_image_num_rows = 0;
    int32_t skip_after_first_partial_image_row = 0;
    // Only case where we have first_partial_image_rows: We have at at least 1 completed image and the starting image row is in the middle of an image
    if (image_end_idx - image_start_idx_after_partial_right_aligned_row > 0 && image_start_height_after_partial_right_aligned_row > 0) {
        first_partial_image_num_rows = pc.in_h - image_start_height_after_partial_right_aligned_row;
        skip_after_first_partial_image_row = pc.pad_h * (pc.in_w + 2 * pc.pad_w);
    }

    // Full images
    int32_t image_rows_after_first_partial_image = sticks_after_first_partial_row / pc.in_w - first_partial_image_num_rows;
    if (to_print) log_debug("image_rows_after_first_partial_image: {}", image_rows_after_first_partial_image);
    int32_t num_full_images = image_rows_after_first_partial_image / pc.in_h;
    int32_t skip_after_full_image = num_full_images > 0 ? pc.pad_h * (pc.in_w + 2 * pc.pad_w) : 0;

    // Last partial image rows
    int32_t last_partial_image_num_rows = image_rows_after_first_partial_image % pc.in_h;

    return NewShardingConfig{
               .first_partial_right_aligned_row_width = first_partial_right_aligned_row_width,
               .first_partial_image_num_rows = first_partial_image_num_rows,
               .num_full_images = num_full_images,
               .last_partial_image_num_rows = last_partial_image_num_rows,
               .last_partial_left_aligned_row_width = last_partial_left_aligned_row_width,
               .skip_after_partial_right_aligned_row = skip_after_partial_right_aligned_row,
               .skip_after_first_partial_image_row = skip_after_first_partial_image_row,
               .skip_after_full_image = skip_after_full_image
           };
}

// reader noc coords for left and right neighbors
std::map<CoreCoord, CoreCoord> left_neighbor_noc_xy, right_neighbor_noc_xy;
void init_neighbor_noc_xy_mapping(CoreCoord grid_size, uint32_t noc = 0) {
    TT_ASSERT(grid_size.x == 12 && grid_size.y == 9);
    if (noc == 0) {
        for (uint32_t y = 1; y <= grid_size.y; ++ y) {
            for (uint32_t x = 1; x <= grid_size.x; ++ x) {
                CoreCoord local_noc = {x, y > 5 ? y + 1 : y};
                CoreCoord left_noc, right_noc;
                // calculate left neighbor
                left_noc.x = local_noc.x;
                left_noc.y = local_noc.y;
                if (left_noc.x > 1) {
                    left_noc.x -= 1;
                } else {
                    left_noc.x = grid_size.x;
                    left_noc.y -= 1;
                }
                if (left_noc.y < 1) {
                    // there is no left neighbor
                } else {
                    if (left_noc.y == 6) {
                        // y = 6 is to be skipped
                        left_noc.y -= 1;
                    }
                    left_neighbor_noc_xy[local_noc] = {left_noc.x, left_noc.y};
                }
                // calculate right neighbor
                right_noc.x = local_noc.x;
                right_noc.y = local_noc.y;
                if (right_noc.x < grid_size.x) {
                    right_noc.x += 1;
                } else {
                    right_noc.y += 1;
                    right_noc.x = 1;
                }
                // NOTE: y = 6 is to be skipped. Hence go till y + 1
                if (right_noc.y > grid_size.y + 1) {
                    // there is no right neighbor
                } else {
                    if (right_noc.y == 6) {
                        // y = 6 is to be skipped
                        right_noc.y += 1;
                    }
                    right_neighbor_noc_xy[local_noc] = {right_noc.x, right_noc.y};
                }
            }
        }
    } else {
        // noc == 1
        TT_ASSERT(noc == 0, "noc = 1 for reader is not yet handled in sharded input case.");
    }
}

} // namespace untilize_with_halo_helpers

// The case of stride = 2
operation::ProgramWithCallbacks untilize_with_halo_multi_core_s2(const Tensor& input, Tensor& output, uint32_t pad_val) {
    Program program = Program();

    Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = input.shape();
    Shape output_shape = output.shape();

    DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    DataFormat cb_df = in_df;
    uint32_t tile_size = tt_metal::detail::TileSize(cb_df);

    uint32_t ntiles = input.volume() / TILE_HW;
    uint32_t ntiles_per_block = input_shape[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_shape[3] * input.element_size();

    // TODO: hard coded for testing only. need to pass these args in.
    // These are the input values before inserting and padding or halo
    uint32_t nbatch = input_shape[0];
    uint32_t in_h = std::sqrt(input_shape[2]);
    uint32_t in_w = in_h;
    TT_ASSERT(in_h * in_w == input_shape[2]);
    uint32_t in_c = input_shape[3];
    uint32_t pad_h = 1;
    uint32_t pad_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint32_t dilation_h = 1;
    uint32_t dilation_w = 1;

    // resuting output shape of subsequent pooling op
    uint32_t out_h = ((in_h + 2 * pad_h - (dilation_h * window_h - 1) - 1) / stride_h) + 1;
    uint32_t out_w = ((in_w + 2 * pad_w - (dilation_w * window_w - 1) - 1) / stride_w) + 1;

    untilize_with_halo_helpers::PoolConfig pc {
        .in_w = in_w,
        .in_h = in_h,
        .out_w = out_w,
        .out_h = out_h,
        .stride_w = stride_w,
        .stride_h = stride_h,
        .pad_w = pad_w,
        .pad_h = pad_h,
        .window_w = window_w,
        .window_h = window_h,
        .dilation_w = dilation_w,
        .dilation_h = dilation_h
    };


    {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
        log_debug(LogOp, "nbatch: {}", nbatch);
        log_debug(LogOp, "in_h: {}", in_h);
        log_debug(LogOp, "in_w: {}", in_w);
        log_debug(LogOp, "in_c: {}", in_c);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
        log_debug(LogOp, "window_h: {}", window_h);
        log_debug(LogOp, "window_w: {}", window_w);
        log_debug(LogOp, "stride_h: {}", stride_h);
        log_debug(LogOp, "stride_w: {}", stride_w);
    }

    auto grid_size = device->compute_with_storage_grid_size();

    untilize_with_halo_helpers::init_neighbor_noc_xy_mapping(grid_size);

    int32_t ncores_x = grid_size.x;     // distributing data to cores row-wise
    CoreRangeSet all_cores = input.shard_spec().value().shard_grid;
    int32_t ncores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        ncores += core_range.size();
    }
    CoreRangeSet core_range_cliff = CoreRangeSet({});
    uint32_t nblocks_per_core = input.shard_spec().value().shard_shape[0] / TILE_HEIGHT;
    uint32_t nblocks_per_core_cliff = 0;

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = nbatch * in_hw;
    uint32_t in_stick_nbytes = in_c * in_nbytes;
    uint32_t in_nsticks = in_nhw;
    uint32_t in_nsticks_per_batch = in_hw;
    uint32_t in_nsticks_per_core = in_nhw / ncores;

    {
        log_debug(LogOp, "shard_shape: {},{}", input.shard_spec().value().shard_shape[0], input.shard_spec().value().shard_shape[1]);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
        log_debug(LogOp, "in_hw: {}", in_hw);
        log_debug(LogOp, "in_nhw: {}", in_nhw);
        log_debug(LogOp, "in_stick_nbytes: {}", in_stick_nbytes);
        log_debug(LogOp, "in_nsticks_per_batch: {}", in_nsticks_per_batch);
        log_debug(LogOp, "in_nsticks_per_core: {}", in_nsticks_per_core);
    }

    uint32_t src_cb_id = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto src_cb_config = CircularBufferConfig(num_input_tiles * tile_size, {{src_cb_id, cb_df}})
                            .set_page_size(src_cb_id, tile_size)
                            .set_globally_allocated_address(input.buffer()->address());
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output of untilize from compute kernel goes into this CB
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block * nblocks_per_core;
    auto untilize_out_cb_config = CircularBufferConfig(num_output_tiles * tile_size, {{untilize_out_cb_id, cb_df}})
                                    .set_page_size(untilize_out_cb_id, tile_size);
    auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);

    // output after concatenating halo and padding goes into this CB, as input to next op.
    uint32_t out_cb_id = CB::c_out1;
    uint32_t out_cb_pagesize = out_nbytes * in_c;
    uint32_t local_npages = in_nsticks_per_core                                                 // data sticks
                                + (in_nsticks_per_core / in_w) * 2                              // left/right edge padding
                                + (in_nsticks_per_core / in_nsticks_per_batch) * (in_w + 2);    // padding rows
    // NOTE: this is always the same for all cores
    uint32_t halo_npages = (in_w + 1 + 2) * 2;  // left and right halo
    uint32_t out_cb_npages = local_npages + halo_npages;
    uint32_t out_nsticks_per_core = local_npages + halo_npages;
    {
        log_debug(LogOp, "out_cb_pagesize: {}", out_cb_pagesize);
        log_debug(LogOp, "local_npages: {}", local_npages);
        log_debug(LogOp, "halo_npages: {}", halo_npages);
        log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
    }
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, cb_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(output.buffer()->address());
    auto cb_out = CreateCircularBuffer(program, all_cores, out_cb_config);

    /** reader
     */

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src_cb_id
    };

    KernelID reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) untilize_out_cb_id,
        (std::uint32_t) out_cb_id
    };
    KernelID writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_sharded_with_halo_s2.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    /** compute
     */
    TT_ASSERT(core_range_cliff.ranges().size() == 0);
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    KernelID untilize_kernel_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_args});

    // const buffer with pad value
    uint32_t const_buffer_size = input_shape[3];
    auto const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(uint16_t(pad_val))));
    const Tensor const_tensor = Tensor(OwnedStorage{const_buffer},
                                       Shape({1, 1, 1, const_buffer_size}),
                                       DataType::BFLOAT16,
                                       Layout::ROW_MAJOR)
                                    .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                             .buffer_type = BufferType::L1});
    auto const_tensor_addr = const_tensor.buffer()->address();

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    // cliff core not yet supported
    TT_ASSERT(nblocks_per_core_cliff == 0);

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core // ntiles
    };

    TT_ASSERT(in_nhw % ncores == 0);

    // writer inserts halo and padding where needed
    vector<uint32_t> writer_rt_args = {
        ntiles_per_block * nblocks_per_core,  // in_nsticks,                     // 0
        in_nsticks_per_core,  // UNUSED
        0,  // partial_first_row_nsticks,
        pad_w,
        in_w,
        0,  // partial_top_image_nrows,        // 5
        pad_h,
        in_h,
        0,  // full_nimages,
        0,  // partial_bottom_image_nrows,
        0,  // partial_last_row_nsticks,       // 10
        0,  // halo_for_left_left_nsticks,
        0,  // halo_for_left_nsticks,
        0,  // halo_for_right_nsticks,
        0,  // halo_for_right_right_nsticks,
        0,  // local_in_stick_start,           // 15
        0,  // local_in_stick_end,
        in_nsticks_per_batch,
        in_nsticks_per_core,
        0,  // has_left,
        0,  // left_noc_x,                     // 20
        0,  // left_noc_y,
        0,  // has_right,
        0,  // right_noc_x,
        0,  // right_noc_y,
        0,  // has_left_left,                  // 25
        0,  // left_left_noc_x,
        0,  // left_left_noc_y,
        0,  // has_right_right,
        0,  // right_right_noc_x,
        0,  // right_right_noc_y,              // 30
        in_stick_nbytes,
        0,  // left_left_nsticks,
        0,  // left_nsticks,
        0,  // right_nsticks,
        0,  // right_right_nsticks,            // 35
        0,  // right_right_halo_offset,
        0,  // right_halo_offset,
        0,  // left_halo_offset,
        0,  // left_left_halo_offset,
        0,  // left_halo_pad_i_offset          // 40
        0,  // right_halo_pad_i_offset
        0,  // partial_first_row_skip
        0,  // partial_top_image_skip
        0,  // full_image_skip
        0,  // initial_pad_nsticks             // 45
        const_tensor_addr,
        // sharding config
        0,
        0,
        0,
        0,                                     // 50
        0,
        0,
        0,
        0,
    };

    uint32_t writer_noc = 0;

    TT_ASSERT(window_h == 3 && window_w == 3);

    int32_t halo_in_nsticks = (in_w + (window_w / 2)) * (window_h / 2);     // input sticks to the writer
    int32_t halo_out_nsticks = (in_w + 2 * pad_w) * pad_h + window_w / 2;   // output sticks from the writer

    log_debug(LogOp, "halo_in_nsticks: {}", halo_in_nsticks);
    log_debug(LogOp, "halo_out_nsticks: {}", halo_out_nsticks);

    /*
    // For each core, calculate how many sticks are coming from left left/left/right/right right neighbors:
    // These are used by the neighbor cores to set their runtime args for the data to be pushed to me.
    // Also calculate each of these segments start offset in my local l1: these need to take all my padding (left/right/top) into account
    map<int32_t, int32_t> my_left_halo, my_left_left_halo, my_right_halo, my_right_right_halo;
    map<int32_t, uint32_t> my_left_halo_offset, my_left_left_halo_offset, my_right_halo_offset, my_right_right_halo_offset;
    map<int32_t, uint32_t> my_left_halo_pad_i_offset, my_right_halo_pad_i_offset;
    int32_t in_stick_start = 0;
    for (int32_t i = 0; i < ncores_full; ++ i) {
        int32_t in_stick_batch_start = in_stick_start / in_nsticks_per_batch;
        int32_t in_stick_batch_end = (in_stick_start + in_nsticks_per_core) / in_nsticks_per_batch;

        // left side halo
        my_left_halo[i] = 0;
        my_left_left_halo[i] = 0;
        int32_t halo_stick_start = in_stick_start - halo_in_nsticks;
        int32_t stick_id = halo_stick_start < 0 ? 0 : halo_stick_start;
        while(stick_id < in_stick_start) {
            int32_t core = stick_id / in_nsticks_per_core;
            switch (core - i) {
                case - 2:
                    TT_ASSERT(false, "This case not correctly handled yet. Needs fixing");
                    ++ my_left_left_halo[i];
                    break;
                case - 1:
                    ++ my_left_halo[i];
                    break;
                default:
                    TT_ASSERT(false, "Encountered an unsupported case!!!!");
            }
            ++ stick_id;
        }
        my_left_left_halo_offset[i] = 0;    // always starts at the beginning
        if ((halo_stick_start / in_w) == ((halo_stick_start + my_left_left_halo[i]) / in_w)) {
            // left left and left halo start on the same row, there is no additional offset
            my_left_halo_offset[i] = my_left_left_halo_offset[i] + my_left_left_halo[i] * in_stick_nbytes;
        } else {
            // left left halo and left halo are in different rows, so there's 2 additional padding sticks (right/left edge)
            my_left_halo_offset[i] = my_left_left_halo_offset[i] + (my_left_left_halo[i] + 2) * in_stick_nbytes;
        }
        my_left_halo_pad_i_offset[i] = in_w - (in_stick_start % in_w);

        // right side halo
        my_right_halo[i] = 0;
        my_right_right_halo[i] = 0;
        int32_t halo_stick_end = in_stick_start + in_nsticks_per_core + halo_in_nsticks;
        stick_id = in_stick_start + in_nsticks_per_core;
        while(stick_id < halo_stick_end) {
            int32_t core = stick_id / in_nsticks_per_core;
            switch (core - i) {
                case 2:
                    TT_ASSERT(false, "This case not correctly handled yet. Needs fixing");
                    ++ my_right_right_halo[i];
                    break;
                case 1:
                    ++ my_right_halo[i];
                    break;
                default:
                    TT_ASSERT(false, "Something went really wrong!!");
            }
            ++ stick_id;
        }

        // uint32_t my_batch = in_stick_start / in_nsticks_per_batch;
        // uint32_t my_core = i;

        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;
        uint32_t initial_pad_nsticks = 0;
        if (in_stick_start % (in_h * in_w) == 0) {
            // This is start of image. Insert initial padding worth halo size
            initial_pad_nsticks = halo_out_nsticks;
        }

        // uint32_t local_nsticks = in_nsticks_per_core                                                // data sticks
        //                             + (in_nsticks_per_core / in_w) * 2                              // left/right edge padding
        //                             + (in_nsticks_per_core / in_nsticks_per_batch) * (in_w + 2);    // padding rows
        uint32_t local_nsticks = partial_first_row_nsticks + partial_first_row_skip
                                    + partial_top_image_nrows * (in_w + 2 * pad_w) + partial_top_image_skip
                                    + full_nimages * (in_w + 2 * pad_w) * in_h
                                    + full_image_skip
                                    + partial_bottom_image_nrows * (in_w + 2 * pad_w)
                                    + partial_last_row_nsticks;

        // NOTE: this is always the same for all cores
        uint32_t halo_nsticks = (in_w + window_w / 2 + 2 * pad_w);  // left or right halo
        uint32_t out_nsticks_per_core = local_nsticks + 2 * halo_nsticks;

        my_right_halo_offset[i] = my_left_halo_offset[i] + (((initial_pad_nsticks > 0) ? initial_pad_nsticks : halo_nsticks) + local_nsticks) * in_stick_nbytes;
        if ((in_stick_start + in_nsticks_per_core) / in_w == (in_stick_start + in_nsticks_per_core + my_right_halo[i]) / in_w) {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + my_right_halo[i] * in_stick_nbytes;
        } else {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + (my_right_halo[i] + 2) * in_stick_nbytes;
        }
        my_right_halo_pad_i_offset[i] = in_w - ((in_stick_start + in_nsticks_per_core) % in_w);

        if (0)
        {
            log_debug(LogOp, "==== Core {}", i);
            log_debug(LogOp, "local_nsticks: {}", local_nsticks);
            log_debug(LogOp, "halo_nsticks: {}", halo_nsticks);
            log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
            log_debug(LogOp, "in_stick_start {}", in_stick_start);
            log_debug(LogOp, "my_left_halo = {}", my_left_halo[i]);
            log_debug(LogOp, "my_left_halo_offset = {}", my_left_halo_offset[i]);
            log_debug(LogOp, "my_left_left_halo = {}", my_left_left_halo[i]);
            log_debug(LogOp, "my_left_left_halo_offset = {}", my_left_left_halo_offset[i]);
            log_debug(LogOp, "my_left_halo_pad_i_offset = {}", my_left_halo_pad_i_offset[i]);
            log_debug(LogOp, "my_right_halo = {}", my_right_halo[i]);
            log_debug(LogOp, "my_right_halo_offset = {}", my_right_halo_offset[i]);
            log_debug(LogOp, "my_right_right_halo = {}", my_right_right_halo[i]);
            log_debug(LogOp, "my_right_right_halo_offset = {}", my_right_right_halo_offset[i]);
            log_debug(LogOp, "my_right_halo_pad_i_offset = {}", my_right_halo_pad_i_offset[i]);
        }

        in_stick_start += in_nsticks_per_core;
    }

    in_stick_start = 0;
    uint32_t out_stick_start = 0;
    for (uint32_t i = 0; i < ncores_full; ++ i) {
        CoreCoord core = {i % ncores_x, i / ncores_x};  // logical
        CoreCoord noc_core = core;  // calculate physical
        if (writer_noc == 0) {
            noc_core.x += 1;
            noc_core.y += 1;
            if (noc_core.y > 5) {
                noc_core.y += 1;
            }
        } else {
            TT_ASSERT(false);
        }

        // reader rt args
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        // writer rt args
        writer_rt_args[15] = in_stick_start;    // unused
        writer_rt_args[16] = in_stick_start + in_nsticks_per_core;  // unused

        // int32_t partial_first_row_nsticks = (in_w - (in_stick_start % in_w)) % in_w;
        // int32_t batch = in_stick_start / in_hw;
        // int32_t partial_top_image_nrows = ((batch + 1) * in_h - (int32_t) ceil((float) in_stick_start / in_w)) % in_h;
        // int32_t full_nimages = ((int32_t) in_nsticks_per_core - (partial_first_row_nsticks + (partial_top_image_nrows * in_w))) / in_nsticks_per_batch;
        // int32_t rem_nsticks = (int32_t) in_nsticks_per_core - (partial_first_row_nsticks + partial_top_image_nrows * in_w + full_nimages * in_hw);
        // int32_t partial_bottom_image_nrows = rem_nsticks / in_w;
        // int32_t partial_last_row_nsticks = rem_nsticks % in_w;

        // ShardingConfig sc = get_specs_for_sharding_partition(out_stick_start, out_stick_start + out_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;
        uint32_t initial_pad_nsticks = 0;
        if (in_stick_start % (in_h * in_w) == 0) {
            // This is start of image. Insert initial padding worth halo size
            initial_pad_nsticks = halo_out_nsticks;
        }

        writer_rt_args[45] = initial_pad_nsticks;

        writer_rt_args[2] = partial_first_row_nsticks;
        writer_rt_args[5] = partial_top_image_nrows;
        writer_rt_args[8] = full_nimages;
        writer_rt_args[9] = partial_bottom_image_nrows;
        writer_rt_args[10] = partial_last_row_nsticks;

        writer_rt_args[42] = partial_first_row_skip;
        writer_rt_args[43] = partial_top_image_skip;
        writer_rt_args[44] = full_image_skip;

        if (untilize_with_halo_helpers::left_neighbor_noc_xy.count(noc_core) > 0) {
            CoreCoord left_noc = untilize_with_halo_helpers::left_neighbor_noc_xy.at(noc_core);
            writer_rt_args[19] = 1;
            writer_rt_args[20] = left_noc.x;
            writer_rt_args[21] = left_noc.y;
            writer_rt_args[33] = my_right_halo[i - 1];      // sticks to left neighbor core's right halo
            writer_rt_args[37] = my_right_halo_offset[i - 1];
            if (untilize_with_halo_helpers::left_neighbor_noc_xy.count(left_noc) > 0) {
                CoreCoord left_left_noc = untilize_with_halo_helpers::left_neighbor_noc_xy.at(left_noc);
                writer_rt_args[25] = 1;
                writer_rt_args[26] = left_left_noc.x;
                writer_rt_args[27] = left_left_noc.y;
                writer_rt_args[32] = my_right_right_halo[i - 2];    // sticks to left left neighbor core's right right halo
                writer_rt_args[36] = my_right_right_halo_offset[i - 2];
            } else {
                writer_rt_args[25] = 0;
            }
            writer_rt_args[40] = my_right_halo_pad_i_offset[i - 1];
        } else {
            writer_rt_args[19] = 0;
            writer_rt_args[25] = 0;
        }
        if (untilize_with_halo_helpers::right_neighbor_noc_xy.count(noc_core) > 0 && (i < ncores_full - 1)) {
            CoreCoord right_noc = untilize_with_halo_helpers::right_neighbor_noc_xy.at(noc_core);
            writer_rt_args[22] = 1;
            writer_rt_args[23] = right_noc.x;
            writer_rt_args[24] = right_noc.y;
            writer_rt_args[34] = my_left_halo[i + 1];       // sticks to right neighbor core's left halo
            writer_rt_args[38] = my_left_halo_offset[i + 1];
            if (untilize_with_halo_helpers::right_neighbor_noc_xy.count(noc_core) > 0 && (i < ncores_full - 2)) {
                CoreCoord right_right_noc = untilize_with_halo_helpers::right_neighbor_noc_xy.at(right_noc);
                writer_rt_args[28] = 1;
                writer_rt_args[29] = right_right_noc.x;
                writer_rt_args[30] = right_right_noc.y;
                writer_rt_args[35] = my_left_left_halo[i + 2];      // sticks to right right neighbor core's left left halo
                writer_rt_args[39] = my_left_left_halo_offset[i + 2];
            } else {
                writer_rt_args[28] = 0;
            }
            writer_rt_args[41] = my_left_halo_pad_i_offset[i + 1];
        } else {
            writer_rt_args[22] = 0;
            writer_rt_args[28] = 0;
        }

        if (0)
        {
            log_debug(LogOp, "++++ Core: {}", i);
            log_debug(LogOp, "out_stick_start: {}", out_stick_start);
            log_debug(LogOp, "halo::has_left: {}", writer_rt_args[19]);
            // log_debug(LogOp, "halo::has_left_left: {}", writer_rt_args[25]);
            log_debug(LogOp, "halo::has_right: {}", writer_rt_args[22]);
            // log_debug(LogOp, "halo::has_right_right: {}", writer_rt_args[28]);
            log_debug(LogOp, "local_in_stick_start: {}", writer_rt_args[15]);
            log_debug(LogOp, "partial_first_row_nsticks: {}", writer_rt_args[2]);
            log_debug(LogOp, "partial_top_image_nrows: {}", writer_rt_args[5]);
            log_debug(LogOp, "full_nimages: {}", writer_rt_args[8]);
            log_debug(LogOp, "partial_bottom_image_nrows: {}", writer_rt_args[9]);
            log_debug(LogOp, "partial_last_row_nsticks: {}", writer_rt_args[10]);
            log_debug(LogOp, "skip_after_partial_right_aligned_row: {}", sc.skip_after_partial_right_aligned_row);
            log_debug(LogOp, "skip_after_first_partial_image_row: {}", sc.skip_after_first_partial_image_row);
            log_debug(LogOp, "skip_after_full_image: {}", sc.skip_after_full_image);
            // log_debug(LogOp, "halo_for_left_left_nsticks: {}", writer_rt_args[11]);
            log_debug(LogOp, "halo_for_left_nsticks: {}", writer_rt_args[12]);
            log_debug(LogOp, "halo_for_right_nsticks: {}", writer_rt_args[13]);
            // log_debug(LogOp, "halo_for_right_right_nsticks: {}", writer_rt_args[14]);
            // log_debug(LogOp, "left_left_core_nsticks: {}", writer_rt_args[32]);
            log_debug(LogOp, "left_core_nsticks: {}", writer_rt_args[33]);
            log_debug(LogOp, "right_core_nsticks: {}", writer_rt_args[34]);
            // log_debug(LogOp, "right_right_core_nsticks: {}", writer_rt_args[35]);
            // log_debug(LogOp, "left_left_core_halo_offset: {}", writer_rt_args[36]);
            log_debug(LogOp, "left_core_halo_offset: {}", writer_rt_args[37]);
            log_debug(LogOp, "right_core_halo_offset: {}", writer_rt_args[38]);
            // log_debug(LogOp, "right_right_core_halo_offset: {}", writer_rt_args[39]);
            // log_debug(LogOp, "left_going_halo_pad_i_offset: {}", writer_rt_args[40]);
            // log_debug(LogOp, "right_going_halo_pad_i_offset: {}", writer_rt_args[41]);
        }

        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

        in_stick_start += in_nsticks_per_core;
        out_stick_start += out_nsticks_per_core;
    }

    */

    // For each core, calculate the desired resharding with halo and pad inserted in order to
    // to obtain resulting **output after the pooling/downsampling op to be equally distributed across all cores**.
    // The resulting shards with halo and pad could be different across cores.
    // The resharding is represented as a map:
    // map :: core -> [l_halo_start, l_halo_end) [local_start, local_end) [r_halo_start, r_halo_end)
    // (these are global input stick indices, [0, (in_nhw - 1)])
    // (and l_halo_end == local_start, local_end == r_halo_start)
    // calculate this core's [left halo, local owned, right halo]. These halo data could be "anywhere".
    std::map<uint32_t, std::array<range_t, 3>> my_shard;
    int32_t out_stick_start = 0;   // global "output" stick (after downsample/pool)
    int32_t pool_out_nsticks_per_core = nbatch * pc.out_h * pc.out_w / ncores;
    for (uint32_t core = 0; core < ncores; ++ core) {
        range_t out_range = {out_stick_start, out_stick_start + pool_out_nsticks_per_core};
        range_t in_range = untilize_with_halo_helpers::calculate_in_range(out_range, pc);  // this represents the "window" center input sticks
        int32_t l_halo_start = in_range[0] - halo_in_nsticks;
        l_halo_start = l_halo_start < 0 ? 0 : l_halo_start;
        int32_t r_halo_end = in_range[1] + halo_in_nsticks;
        r_halo_end = r_halo_end > in_nhw ? in_nhw : r_halo_end;
        my_shard[core] = {{
            { l_halo_start, in_range[0] }, // l_halo
            { in_range[0], in_range[1] },                   // local
            { in_range[1], r_halo_end }  // r_halo
        }};
        out_stick_start += pool_out_nsticks_per_core;
    }

    // Calculate data shuffle config for each core using its shard size:
    // Given equally distributed input across all cores,
    // for each core, identify the data to be scattered across its neighborhood:
    // construct the map (subset of alltoallv primitive)
    // map :: core -> [ll_start, ll_end] [l_start, l_end] [start, end] [r_start, r_end] [rr_start, rr_end]
    // where, ll_end == l_start, l_end == start, end == r_start, r_end == rr_start
    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> count_to_send;    // nsticks to push to each neighbor
    std::map<uint32_t, std::array<range_t, (NEIGHBORHOOD_DIST * 2 + 1)>> range_to_send;     // range to push to each neighbor
    uint32_t in_stick_start = 0;    // global input stick
    for (uint32_t core = 0; core < ncores; ++ core) {
        uint32_t in_stick_end = in_stick_start + in_nsticks_per_core;
        // log_debug("==== Core {}: in_sticks = [{},{})", core, in_stick_start, in_stick_end);
        // log_debug(" l_halo = [{},{})", my_shard[core][0][0], my_shard[core][0][1]);
        // log_debug(" local ({}) = [{},{})", my_shard[core][1][1] - my_shard[core][1][0], my_shard[core][1][0], my_shard[core][1][1]);
        // log_debug(" r_halo = [{},{})", my_shard[core][2][0], my_shard[core][2][1]);

        // calculate the shuffle config [excl. padding] <-- TODO: see if padding should be handled here or later
        //   - for each core in the neighborhood:
        //         calculate range to push to this core

        // first, cores on the left neighborhood (starting left->right):
        uint32_t count = 0;
        int32_t range_start = 0, range_end = 0;
        for (int32_t l_neighbor = NEIGHBORHOOD_DIST; l_neighbor > 0; -- l_neighbor) {
            int32_t l_core = core - l_neighbor;
            if (l_core >= 0) {  // this is a valid neighbor
                count = 0;
                // see if this neighbor needs anything from me
                // neighbor's right halo end > in_stick_start
                uint32_t stick = in_stick_start;
                range_start = stick;
                while (stick < my_shard[l_core][2][1]) {
                    ++ count;
                    ++ stick;
                }
                range_end = stick;
                count_to_send[core][NEIGHBORHOOD_DIST - l_neighbor] = count;
                range_to_send[core][NEIGHBORHOOD_DIST - l_neighbor] = { range_start, range_end };
            }
        }

        // check if there is data to keep local:
        // that is, if my full new shard range (incl. halos) intersects with my current shard
        count = 0;
        range_start = 0, range_end = 0;
        int32_t my_shard_start = my_shard[core][0][0];
        int32_t my_shard_end = my_shard[core][2][1];
        if (my_shard_end > in_stick_start && my_shard_start < in_stick_end) {
            uint32_t stick = in_stick_start < my_shard_start ? my_shard_start : in_stick_start;
            range_start = stick;
            while (stick < my_shard_end && stick < in_stick_end) {
                ++ count;
                ++ stick;
            }
            range_end = stick;
        }
        count_to_send[core][NEIGHBORHOOD_DIST] = count; // keep local
        range_to_send[core][NEIGHBORHOOD_DIST] = { range_start, range_end }; // keep local

        // cores on the right neighborhood (starting left->right)
        range_start = 0, range_end = 0;
        for (int32_t r_neighbor = 1; r_neighbor <= NEIGHBORHOOD_DIST; ++ r_neighbor) {
            int32_t r_core = core + r_neighbor;
            if (r_core < ncores) {  // this is a valid neighbor
                count = 0;
                // see if this neighbor needs anything from me
                // neighbor's left halo start < in_stick_end
                uint32_t stick = my_shard[r_core][0][0];
                stick = stick < in_stick_start ? in_stick_start : stick;
                range_start = stick;
                while (stick < in_stick_end) {
                    ++ count;
                    ++ stick;
                }
                range_end = stick;
                count_to_send[core][NEIGHBORHOOD_DIST + r_neighbor] = count;
                range_to_send[core][NEIGHBORHOOD_DIST + r_neighbor] = { range_start, range_end };
            }
        }

        in_stick_start += in_nsticks_per_core;
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> count_to_receive; // nsticks pushed from each neighbor
    for (uint32_t core = 0; core < ncores; ++ core) {
        // calculate the nsticks I receive from each of my neighbors
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t count = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                count = count_to_send[neighbor_core][NEIGHBORHOOD_DIST - neighbor];
            }
            count_to_receive[core][NEIGHBORHOOD_DIST + neighbor] = count;
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> updated_count_to_send;    // nsticks to push to each neighbor
    for (uint32_t core = 0; core < ncores; ++ core) {
        // for each core, calculate the offsets where data is to be pushed to
        // the push will be from locally constructed re-sharded data, so no additional need to insert padding when pushing.
        for (int32_t neighbor = -NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            // calculate the total nsticks incl. padding to be sent to this neighbor
            range_t to_send = range_to_send[core][NEIGHBORHOOD_DIST + neighbor];
            bool to_print = false;  // core == 1 && neighbor == -1;
            untilize_with_halo_helpers::NewShardingConfig sc = untilize_with_halo_helpers::get_shard_specs(to_send[0], to_send[1], pc, to_print);
            uint32_t count = 0;
            count += sc.first_partial_right_aligned_row_width;
            count += sc.skip_after_partial_right_aligned_row;
            count += sc.first_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);
            count += sc.num_full_images * (pc.in_h * (pc.in_w + 2 * pc.pad_w) + sc.skip_after_full_image);
            count += sc.last_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);
            count += sc.last_partial_left_aligned_row_width;
            updated_count_to_send[core][NEIGHBORHOOD_DIST + neighbor] = count;
            if (to_print) {
                log_debug(LogOp, "partial_first_row_nsticks: {}", sc.first_partial_right_aligned_row_width);
                log_debug(LogOp, "partial_top_image_nrows: {}", sc.first_partial_image_num_rows);
                log_debug(LogOp, "full_nimages: {}", sc.num_full_images);
                log_debug(LogOp, "partial_bottom_image_nrows: {}", sc.last_partial_image_num_rows);
                log_debug(LogOp, "partial_last_row_nsticks: {}", sc.last_partial_left_aligned_row_width);
                log_debug(LogOp, "skip_after_partial_right_aligned_row: {}", sc.skip_after_partial_right_aligned_row);
                log_debug(LogOp, "skip_after_first_partial_image_row: {}", sc.skip_after_first_partial_image_row);
                log_debug(LogOp, "skip_after_full_image: {}", sc.skip_after_full_image);
            }

            //
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> updated_count_to_receive; // nsticks pushed from each neighbor
    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> receive_at_offset_nsticks;  // data is to be received at these offsets
    for (uint32_t core = 0; core < ncores; ++ core) {
        uint32_t cumulative_count = 0;
        // calculate the nsticks I receive from each of my neighbors
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t count = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                count = updated_count_to_send[neighbor_core][NEIGHBORHOOD_DIST - neighbor];
            }
            updated_count_to_receive[core][NEIGHBORHOOD_DIST + neighbor] = count;
            receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = cumulative_count;
            cumulative_count += count;
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> send_to_offset_nsticks;  // data is to be received at these offsets
    for (uint32_t core = 0; core < ncores; ++ core) {
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t offset = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                offset = receive_at_offset_nsticks[neighbor_core][NEIGHBORHOOD_DIST - neighbor];
            }
            send_to_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = offset;

            // receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST] <-- offset for start of locally owned data
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> send_from_offset_nsticks; // after adding pad etc, the offset to start send data from
    in_stick_start = 0;
    for (uint32_t core = 0; core < ncores; ++ core) {
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t offset = 0;
            log_debug("HMMMMMMM {} {}", core, neighbor);
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                bool to_print = core == 13 && neighbor == 2;
                if (to_print) log_debug("RANGE: {} {}", in_stick_start, range_to_send[core][NEIGHBORHOOD_DIST + neighbor][0]);
                untilize_with_halo_helpers::NewShardingConfig sc = untilize_with_halo_helpers::get_shard_specs(in_stick_start, range_to_send[core][NEIGHBORHOOD_DIST + neighbor][0], pc, to_print);
                offset += sc.first_partial_right_aligned_row_width;
                offset += sc.skip_after_partial_right_aligned_row;
                offset += sc.first_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);
                offset += sc.num_full_images * (pc.in_h * (pc.in_w + 2 * pc.pad_w) + sc.skip_after_full_image);
                offset += sc.last_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);
                offset += sc.last_partial_left_aligned_row_width;
            }
            send_from_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = offset;
        }
        in_stick_start += in_nsticks_per_core;
    }
    // // for each core, for each neighbor, calculate the offset where the data is to go
    // std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> send_to_offset_nsticks;
    // for (uint32_t core = 0; core < ncores; ++ core) {
    //     for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
    //         int32_t neighbor_core = core + neighbor;
    //         send_to_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = sum(updated_count_to_receive[neighbor_core][...]);
    //     }
    // }

    // Calculate all information needed to copy locally owned data to output shard.
    in_stick_start = 0;
    for (uint32_t core = 0; core < ncores; ++ core) {
        // for each core, identify its sections with padding for the data it locally owns
        // that is, compute the sharding config
        uint32_t in_stick_end = in_stick_start + in_nsticks_per_core;

        // sum of all nsticks i will receive from all my left neighbors, incl. padding
        // this represents the offset where my locally owned in data will go.

        // logically insert padding into locally owned data and generate the sharding config
        // information to calculate for a core:
        //  + partial_first_row_nsticks (no pad)
        //  + skip_after_partial_first_row (pad)
        //  + partial_first_image_nrows (no pad) + padding per row (pad)
        //  + skip_after_partial_first_image (pad)
        //  + full_nimages (no pad) + padding per row (pad) + padding per image (pad)
        //  + skip_after_full_images (pad)
        //  + partial_last_image_nrows (no pad) + padding per row (pad)
        //  + partial_last_row_nsticks (no pad)
        untilize_with_halo_helpers::NewShardingConfig sc = untilize_with_halo_helpers::get_shard_specs(in_stick_start, in_stick_end, pc);
        int32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        int32_t skip_after_partial_first_row = sc.skip_after_partial_right_aligned_row;
        int32_t partial_first_image_nrows = sc.first_partial_image_num_rows;
        int32_t skip_after_partial_first_image = sc.skip_after_first_partial_image_row;
        int32_t full_nimages = sc.num_full_images;
        int32_t skip_after_full_images = sc.skip_after_full_image;
        int32_t partial_last_image_nrows = sc.last_partial_image_num_rows;
        int32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;

        // args for handling local data movement:
        //     intial_pad_nsticks
        //     receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST]; // local_offset_nsticks
        //     partial_first_row_nsticks
        //     partial_first_row_skip
        //     partial_top_image_nrows, partial_top_image_skip_per_row
        //     partial_top_image_skip
        //     full_nimages, full_image_skip_per_row
        //     full_image_skip
        //     partial_bottom_image_nrows, partial_bottom_image_skip_per_row
        //     partial_last_row_nsticks

        // args for handling remote data shuffle:
        //     NEIGHBORHOOD_DIST
        //     ll_send_count
        //     ll_send_stick_start, ll_send_stick_end
        //     ll_send_at_offset
        //     l_send_count
        //     l_send_stick_start, l_send_stick_end
        //     l_send_at_offset
        //     r_send_count
        //     r_send_stick_start, r_send_stick_end
        //     r_send_at_offset
        //     rr_send_count
        //     rr_send_stick_start, rr_send_stick_end
        //     rr_send_at_offset

        in_stick_start += in_nsticks_per_core;
    }

    // Handle padding. TODO

    // With all the above info, calculate the actual offsets on each neighbor of each core where to send the padded data.

    in_stick_start = 0;
    for (uint32_t core = 0; core < ncores; ++ core) {
        uint32_t in_stick_end = in_stick_start + in_nsticks_per_core;
        log_debug("==== Core {}:", core);
        log_debug(" in shard = [{},{})", in_stick_start, in_stick_end);
        log_debug(" re shard = [{},{}) [{},{}) [{},{})", my_shard[core][0][0], my_shard[core][0][1],
                                                         my_shard[core][1][0], my_shard[core][1][1],
                                                         my_shard[core][2][0], my_shard[core][2][1]);
        for (uint32_t neighbor = 0; neighbor < 2 * NEIGHBORHOOD_DIST + 1; ++ neighbor) {
            log_debug(" + N {} :: recv: (count = {}, offset = {})\t send: (count = {}, from = {}, to = {}) : [{},{})", neighbor,
                                                                  updated_count_to_receive[core][neighbor],
                                                                  receive_at_offset_nsticks[core][neighbor],
                                                                  updated_count_to_send[core][neighbor],
                                                                  send_from_offset_nsticks[core][neighbor],
                                                                  send_to_offset_nsticks[core][neighbor],
                                                                  range_to_send[core][neighbor][0], range_to_send[core][neighbor][1]);
        }
        in_stick_start += in_nsticks_per_core;
    }

    auto override_runtime_args_callback = [
        reader_kernel_id=reader_kernel_id,
        writer_kernel_id=writer_kernel_id,
        ncores=ncores,
        ncores_x=ncores_x
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core = {i % ncores_x, i / ncores_x};
            // in and out are sharded
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks untilize_with_halo_multi_core(const Tensor& a, Tensor& output, uint32_t pad_val) {
    Program program = Program();

    Device *device = a.device();
    Buffer *src_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = a.shape();
    Shape output_shape = output.shape();

    DataFormat in_df = datatype_to_dataformat_converter(a.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    DataFormat cb_df = in_df;
    uint32_t tile_size = tt_metal::detail::TileSize(cb_df);

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = input_shape[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_shape[3] * a.element_size();

    {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
    }

    // TODO: hard coded for testing only. need to pass these args in.
    uint32_t nbatch = input_shape[0];
    uint32_t in_h = std::sqrt(input_shape[2]);
    uint32_t in_w = in_h;
    TT_ASSERT(in_h * in_w == input_shape[2]);
    uint32_t in_c = input_shape[3];
    uint32_t pad_h = 1;
    uint32_t pad_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;

    {
        log_debug(LogOp, "nbatch: {}", nbatch);
        log_debug(LogOp, "in_h: {}", in_h);
        log_debug(LogOp, "in_w: {}", in_w);
        log_debug(LogOp, "in_c: {}", in_c);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
        log_debug(LogOp, "window_h: {}", window_h);
        log_debug(LogOp, "window_w: {}", window_w);
    }

    auto grid_size = device->compute_with_storage_grid_size();

    untilize_with_halo_helpers::init_neighbor_noc_xy_mapping(grid_size);

    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    CoreRangeSet all_cores = a.shard_spec().value().shard_grid;
    int32_t ncores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        ncores += core_range.size();
    }
    CoreRangeSet core_range_cliff = CoreRangeSet({});
    uint32_t nblocks_per_core = a.shard_spec().value().shard_shape[0] / TILE_HEIGHT;
    uint32_t nblocks_per_core_cliff = 0;

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = nbatch * in_hw;
    uint32_t in_stick_nbytes = in_c * in_nbytes;
    uint32_t in_nsticks = in_nhw;
    uint32_t in_nsticks_per_batch = in_hw;
    uint32_t in_nsticks_per_core = in_nhw / ncores;

    {
        log_debug(LogOp, "shard_shape: {},{}", a.shard_spec().value().shard_shape[0], a.shard_spec().value().shard_shape[1]);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "ncores_y: {}", ncores_y);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
        log_debug(LogOp, "in_hw: {}", in_hw);
        log_debug(LogOp, "in_nhw: {}", in_nhw);
        log_debug(LogOp, "in_stick_nbytes: {}", in_stick_nbytes);
        log_debug(LogOp, "in_nsticks_per_batch: {}", in_nsticks_per_batch);
        log_debug(LogOp, "in_nsticks_per_core: {}", in_nsticks_per_core);
    }

    uint32_t src_cb_id = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto src_cb_config = CircularBufferConfig(num_input_tiles * tile_size, {{src_cb_id, cb_df}})
                            .set_page_size(src_cb_id, tile_size)
                            .set_globally_allocated_address(a.buffer()->address());
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output of untilize from compute kernel goes into this CB
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block * nblocks_per_core;
    auto untilize_out_cb_config = CircularBufferConfig(num_output_tiles * tile_size, {{untilize_out_cb_id, cb_df}})
                                    .set_page_size(untilize_out_cb_id, tile_size);
    auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);

    // output after concatenating halo and padding goes into this CB, as input to next op.
    uint32_t out_cb_id = CB::c_out1;
    uint32_t out_cb_pagesize = out_nbytes * in_c;
    uint32_t local_npages = in_nsticks_per_core                                                 // data sticks
                                + (in_nsticks_per_core / in_w) * 2                              // left/right edge padding
                                + (in_nsticks_per_core / in_nsticks_per_batch) * (in_w + 2);    // padding rows
    // NOTE: this is always the same for all cores
    uint32_t halo_npages = (in_w + 1 + 2) * 2;  // left and right halo
    uint32_t out_cb_npages = local_npages + halo_npages;
    uint32_t out_nsticks_per_core = local_npages + halo_npages;
    {
        log_debug(LogOp, "out_cb_pagesize: {}", out_cb_pagesize);
        log_debug(LogOp, "local_npages: {}", local_npages);
        log_debug(LogOp, "halo_npages: {}", halo_npages);
        log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
    }
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, cb_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(output.buffer()->address());
    auto cb_out = CreateCircularBuffer(program, all_cores, out_cb_config);

    /** reader
     */

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src_cb_id
    };

    KernelID reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) untilize_out_cb_id,
        (std::uint32_t) out_cb_id
    };
    KernelID writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_sharded_with_halo.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    /** compute
     */
    TT_ASSERT(core_range_cliff.ranges().size() == 0);
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    KernelID untilize_kernel_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_args});

    // const buffer with pad value (-INF)
    uint32_t const_buffer_size = input_shape[3];
    auto const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(uint16_t(pad_val))));
    const Tensor const_tensor = Tensor(OwnedStorage{const_buffer},
                                       Shape({1, 1, 1, const_buffer_size}),
                                       DataType::BFLOAT16,
                                       Layout::ROW_MAJOR)
                                    .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                             .buffer_type = BufferType::L1});
    auto const_tensor_addr = const_tensor.buffer()->address();

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    // cliff core not yet supported
    TT_ASSERT(nblocks_per_core_cliff == 0);

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core // ntiles
    };

    TT_ASSERT(in_nhw % ncores == 0);

    vector<uint32_t> writer_rt_args = {
        ntiles_per_block * nblocks_per_core,  // in_nsticks,                     // 0
        in_nsticks_per_core,  // UNUSED
        0,  // partial_first_row_nsticks,
        pad_w,
        in_w,
        0,  // partial_top_image_nrows,        // 5
        pad_h,
        in_h,
        0,  // full_nimages,
        0,  // partial_bottom_image_nrows,
        0,  // partial_last_row_nsticks,       // 10
        0,  // halo_for_left_left_nsticks,
        0,  // halo_for_left_nsticks,
        0,  // halo_for_right_nsticks,
        0,  // halo_for_right_right_nsticks,
        0,  // local_in_stick_start,           // 15
        0,  // local_in_stick_end,
        in_nsticks_per_batch,
        in_nsticks_per_core,
        0,  // has_left,
        0,  // left_noc_x,                     // 20
        0,  // left_noc_y,
        0,  // has_right,
        0,  // right_noc_x,
        0,  // right_noc_y,
        0,  // has_left_left,                  // 25
        0,  // left_left_noc_x,
        0,  // left_left_noc_y,
        0,  // has_right_right,
        0,  // right_right_noc_x,
        0,  // right_right_noc_y,              // 30
        in_stick_nbytes,
        0,  // left_left_nsticks,
        0,  // left_nsticks,
        0,  // right_nsticks,
        0,  // right_right_nsticks,            // 35
        0,  // right_right_halo_offset,
        0,  // right_halo_offset,
        0,  // left_halo_offset,
        0,  // left_left_halo_offset,
        0,  // left_halo_pad_i_offset          // 40
        0,  // right_halo_pad_i_offset
        0,  // partial_first_row_skip
        0,  // partial_top_image_skip
        0,  // full_image_skip
        0,  // initial_pad_nsticks             // 45
        const_tensor_addr,
    };

    uint32_t writer_noc = 0;

    TT_ASSERT(window_h == 3 && window_w == 3);
    // int32_t halo_in_nsticks = (in_w + (window_w / 2)) * (window_h / 2);
    int32_t halo_in_nsticks = (in_w + 1) * 1;
    int32_t halo_out_nsticks = (in_w + 2 * pad_w) * pad_h + window_w / 2;

    log_debug(LogOp, "halo_in_nsticks: {}", halo_in_nsticks);
    log_debug(LogOp, "halo_out_nsticks: {}", halo_out_nsticks);

    // NOTE: Irrespective of batch boundary, always ass the left/right halo to the output shards.
    // IE: output shards ALWAYS have halo region on left and right as long as they are not the start/end of the input
    // TODO: Ensure this produces the correct output after padding is inserted.

    // For each core, calculate how many sticks are coming from left left/left/right/right right neighbors:
    // These are used by the neighbor cores to set their runtime args for the data to be pushed to me.
    // Also calculate each of these segments start offset in my local l1: these need to take all my padding (left/right/top) into account
    map<int32_t, int32_t> my_left_halo, my_left_left_halo, my_right_halo, my_right_right_halo;
    map<int32_t, uint32_t> my_left_halo_offset, my_left_left_halo_offset, my_right_halo_offset, my_right_right_halo_offset;
    map<int32_t, uint32_t> my_left_halo_pad_i_offset, my_right_halo_pad_i_offset;
    int32_t in_stick_start = 0;
    for (int32_t i = 0; i < ncores_full; ++ i) {
        int32_t in_stick_batch_start = in_stick_start / in_nsticks_per_batch;
        int32_t in_stick_batch_end = (in_stick_start + in_nsticks_per_core) / in_nsticks_per_batch;

        // left side halo
        my_left_halo[i] = 0;
        my_left_left_halo[i] = 0;
        int32_t halo_stick_start = in_stick_start - halo_in_nsticks;
        int32_t stick_id = halo_stick_start < 0 ? 0 : halo_stick_start;
        while(stick_id < in_stick_start) {
            int32_t core = stick_id / in_nsticks_per_core;
            switch (core - i) {
                case - 2:
                    TT_ASSERT(false, "This case not correctly handled yet. Needs fixing");
                    ++ my_left_left_halo[i];
                    break;
                case - 1:
                    ++ my_left_halo[i];
                    break;
                default:
                    TT_ASSERT(false, "Encountered an unsupported case!!!!");
            }
            ++ stick_id;
        }
        my_left_left_halo_offset[i] = 0;    // always starts at the beginning
        if ((halo_stick_start / in_w) == ((halo_stick_start + my_left_left_halo[i]) / in_w)) {
            // left left and left halo start on the same row, there is no additional offset
            my_left_halo_offset[i] = my_left_left_halo_offset[i] + my_left_left_halo[i] * in_stick_nbytes;
        } else {
            // left left halo and left halo are in different rows, so there's 2 additional padding sticks (right/left edge)
            my_left_halo_offset[i] = my_left_left_halo_offset[i] + (my_left_left_halo[i] + 2) * in_stick_nbytes;
        }
        my_left_halo_pad_i_offset[i] = in_w - (in_stick_start % in_w);

        // right side halo
        my_right_halo[i] = 0;
        my_right_right_halo[i] = 0;
        int32_t halo_stick_end = in_stick_start + in_nsticks_per_core + halo_in_nsticks;
        stick_id = in_stick_start + in_nsticks_per_core;
        while(stick_id < halo_stick_end) {
            int32_t core = stick_id / in_nsticks_per_core;
            switch (core - i) {
                case 2:
                    TT_ASSERT(false, "This case not correctly handled yet. Needs fixing");
                    ++ my_right_right_halo[i];
                    break;
                case 1:
                    ++ my_right_halo[i];
                    break;
                default:
                    TT_ASSERT(false, "Something went really wrong!!");
            }
            ++ stick_id;
        }

        // uint32_t my_batch = in_stick_start / in_nsticks_per_batch;
        // uint32_t my_core = i;

        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;
        uint32_t initial_pad_nsticks = 0;
        if (in_stick_start % (in_h * in_w) == 0) {
            // This is start of image. Insert initial padding worth halo size
            initial_pad_nsticks = halo_out_nsticks;
        }

        // uint32_t local_nsticks = in_nsticks_per_core                                                // data sticks
        //                             + (in_nsticks_per_core / in_w) * 2                              // left/right edge padding
        //                             + (in_nsticks_per_core / in_nsticks_per_batch) * (in_w + 2);    // padding rows
        uint32_t local_nsticks = partial_first_row_nsticks + partial_first_row_skip
                                    + partial_top_image_nrows * (in_w + 2 * pad_w) + partial_top_image_skip
                                    + full_nimages * (in_w + 2 * pad_w) * in_h
                                    + full_image_skip
                                    + partial_bottom_image_nrows * (in_w + 2 * pad_w)
                                    + partial_last_row_nsticks;

        // NOTE: this is always the same for all cores
        uint32_t halo_nsticks = (in_w + window_w / 2 + 2 * pad_w);  // left or right halo
        uint32_t out_nsticks_per_core = local_nsticks + 2 * halo_nsticks;

        my_right_halo_offset[i] = my_left_halo_offset[i] + (((initial_pad_nsticks > 0) ? initial_pad_nsticks : halo_nsticks) + local_nsticks) * in_stick_nbytes;
                                    // (in_nsticks_per_core /* local sticks */ + (in_nsticks_per_core / in_w) * 2 /* 2 padding sticks per row */) * in_stick_nbytes;
        if ((in_stick_start + in_nsticks_per_core) / in_w == (in_stick_start + in_nsticks_per_core + my_right_halo[i]) / in_w) {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + my_right_halo[i] * in_stick_nbytes;
        } else {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + (my_right_halo[i] + 2) * in_stick_nbytes;
        }
        my_right_halo_pad_i_offset[i] = in_w - ((in_stick_start + in_nsticks_per_core) % in_w);

        if (0)
        {
            log_debug(LogOp, "==== Core {}", i);
            log_debug(LogOp, "local_nsticks: {}", local_nsticks);
            log_debug(LogOp, "halo_nsticks: {}", halo_nsticks);
            log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
            log_debug(LogOp, "in_stick_start {}", in_stick_start);
            log_debug(LogOp, "my_left_halo = {}", my_left_halo[i]);
            log_debug(LogOp, "my_left_halo_offset = {}", my_left_halo_offset[i]);
            log_debug(LogOp, "my_left_left_halo = {}", my_left_left_halo[i]);
            log_debug(LogOp, "my_left_left_halo_offset = {}", my_left_left_halo_offset[i]);
            log_debug(LogOp, "my_left_halo_pad_i_offset = {}", my_left_halo_pad_i_offset[i]);
            log_debug(LogOp, "my_right_halo = {}", my_right_halo[i]);
            log_debug(LogOp, "my_right_halo_offset = {}", my_right_halo_offset[i]);
            log_debug(LogOp, "my_right_right_halo = {}", my_right_right_halo[i]);
            log_debug(LogOp, "my_right_right_halo_offset = {}", my_right_right_halo_offset[i]);
            log_debug(LogOp, "my_right_halo_pad_i_offset = {}", my_right_halo_pad_i_offset[i]);
        }

        in_stick_start += in_nsticks_per_core;
    }

    in_stick_start = 0;
    uint32_t out_stick_start = 0;
    for (uint32_t i = 0; i < ncores_full; ++ i) {
        CoreCoord core = {i % ncores_x, i / ncores_x};  // logical
        CoreCoord noc_core = core;  // calculate physical
        if (writer_noc == 0) {
            noc_core.x += 1;
            noc_core.y += 1;
            if (noc_core.y > 5) {
                noc_core.y += 1;
            }
        } else {
            TT_ASSERT(false);
        }

        // reader rt args
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        // writer rt args
        writer_rt_args[15] = in_stick_start;    // unused
        writer_rt_args[16] = in_stick_start + in_nsticks_per_core;  // unused

        // int32_t partial_first_row_nsticks = (in_w - (in_stick_start % in_w)) % in_w;
        // int32_t batch = in_stick_start / in_hw;
        // int32_t partial_top_image_nrows = ((batch + 1) * in_h - (int32_t) ceil((float) in_stick_start / in_w)) % in_h;
        // int32_t full_nimages = ((int32_t) in_nsticks_per_core - (partial_first_row_nsticks + (partial_top_image_nrows * in_w))) / in_nsticks_per_batch;
        // int32_t rem_nsticks = (int32_t) in_nsticks_per_core - (partial_first_row_nsticks + partial_top_image_nrows * in_w + full_nimages * in_hw);
        // int32_t partial_bottom_image_nrows = rem_nsticks / in_w;
        // int32_t partial_last_row_nsticks = rem_nsticks % in_w;

        // ShardingConfig sc = get_specs_for_sharding_partition(out_stick_start, out_stick_start + out_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;
        uint32_t initial_pad_nsticks = 0;
        if (in_stick_start % (in_h * in_w) == 0) {
            // This is start of image. Insert initial padding worth halo size
            initial_pad_nsticks = halo_out_nsticks;
        }

        writer_rt_args[45] = initial_pad_nsticks;

        writer_rt_args[2] = partial_first_row_nsticks;
        writer_rt_args[5] = partial_top_image_nrows;
        writer_rt_args[8] = full_nimages;
        writer_rt_args[9] = partial_bottom_image_nrows;
        writer_rt_args[10] = partial_last_row_nsticks;

        writer_rt_args[42] = partial_first_row_skip;
        writer_rt_args[43] = partial_top_image_skip;
        writer_rt_args[44] = full_image_skip;

        if (untilize_with_halo_helpers::left_neighbor_noc_xy.count(noc_core) > 0) {
            CoreCoord left_noc = untilize_with_halo_helpers::left_neighbor_noc_xy.at(noc_core);
            writer_rt_args[19] = 1;
            writer_rt_args[20] = left_noc.x;
            writer_rt_args[21] = left_noc.y;
            writer_rt_args[33] = my_right_halo[i - 1];      // sticks to left neighbor core's right halo
            writer_rt_args[37] = my_right_halo_offset[i - 1];
            if (untilize_with_halo_helpers::left_neighbor_noc_xy.count(left_noc) > 0) {
                CoreCoord left_left_noc = untilize_with_halo_helpers::left_neighbor_noc_xy.at(left_noc);
                writer_rt_args[25] = 1;
                writer_rt_args[26] = left_left_noc.x;
                writer_rt_args[27] = left_left_noc.y;
                writer_rt_args[32] = my_right_right_halo[i - 2];    // sticks to left left neighbor core's right right halo
                writer_rt_args[36] = my_right_right_halo_offset[i - 2];
            } else {
                writer_rt_args[25] = 0;
            }
            writer_rt_args[40] = my_right_halo_pad_i_offset[i - 1];
        } else {
            writer_rt_args[19] = 0;
            writer_rt_args[25] = 0;
        }
        if (untilize_with_halo_helpers::right_neighbor_noc_xy.count(noc_core) > 0 && (i < ncores_full - 1)) {
            CoreCoord right_noc = untilize_with_halo_helpers::right_neighbor_noc_xy.at(noc_core);
            writer_rt_args[22] = 1;
            writer_rt_args[23] = right_noc.x;
            writer_rt_args[24] = right_noc.y;
            writer_rt_args[34] = my_left_halo[i + 1];       // sticks to right neighbor core's left halo
            writer_rt_args[38] = my_left_halo_offset[i + 1];
            if (untilize_with_halo_helpers::right_neighbor_noc_xy.count(noc_core) > 0 && (i < ncores_full - 2)) {
                CoreCoord right_right_noc = untilize_with_halo_helpers::right_neighbor_noc_xy.at(right_noc);
                writer_rt_args[28] = 1;
                writer_rt_args[29] = right_right_noc.x;
                writer_rt_args[30] = right_right_noc.y;
                writer_rt_args[35] = my_left_left_halo[i + 2];      // sticks to right right neighbor core's left left halo
                writer_rt_args[39] = my_left_left_halo_offset[i + 2];
            } else {
                writer_rt_args[28] = 0;
            }
            writer_rt_args[41] = my_left_halo_pad_i_offset[i + 1];
        } else {
            writer_rt_args[22] = 0;
            writer_rt_args[28] = 0;
        }

        if (0)
        {
            log_debug(LogOp, "++++ Core: {}", i);
            log_debug(LogOp, "out_stick_start: {}", out_stick_start);
            log_debug(LogOp, "halo::has_left: {}", writer_rt_args[19]);
            // log_debug(LogOp, "halo::has_left_left: {}", writer_rt_args[25]);
            log_debug(LogOp, "halo::has_right: {}", writer_rt_args[22]);
            // log_debug(LogOp, "halo::has_right_right: {}", writer_rt_args[28]);
            log_debug(LogOp, "local_in_stick_start: {}", writer_rt_args[15]);
            log_debug(LogOp, "partial_first_row_nsticks: {}", writer_rt_args[2]);
            log_debug(LogOp, "partial_top_image_nrows: {}", writer_rt_args[5]);
            log_debug(LogOp, "full_nimages: {}", writer_rt_args[8]);
            log_debug(LogOp, "partial_bottom_image_nrows: {}", writer_rt_args[9]);
            log_debug(LogOp, "partial_last_row_nsticks: {}", writer_rt_args[10]);
            log_debug(LogOp, "skip_after_partial_right_aligned_row: {}", sc.skip_after_partial_right_aligned_row);
            log_debug(LogOp, "skip_after_first_partial_image_row: {}", sc.skip_after_first_partial_image_row);
            log_debug(LogOp, "skip_after_full_image: {}", sc.skip_after_full_image);
            // log_debug(LogOp, "halo_for_left_left_nsticks: {}", writer_rt_args[11]);
            log_debug(LogOp, "halo_for_left_nsticks: {}", writer_rt_args[12]);
            log_debug(LogOp, "halo_for_right_nsticks: {}", writer_rt_args[13]);
            // log_debug(LogOp, "halo_for_right_right_nsticks: {}", writer_rt_args[14]);
            // log_debug(LogOp, "left_left_core_nsticks: {}", writer_rt_args[32]);
            log_debug(LogOp, "left_core_nsticks: {}", writer_rt_args[33]);
            log_debug(LogOp, "right_core_nsticks: {}", writer_rt_args[34]);
            // log_debug(LogOp, "right_right_core_nsticks: {}", writer_rt_args[35]);
            // log_debug(LogOp, "left_left_core_halo_offset: {}", writer_rt_args[36]);
            log_debug(LogOp, "left_core_halo_offset: {}", writer_rt_args[37]);
            log_debug(LogOp, "right_core_halo_offset: {}", writer_rt_args[38]);
            // log_debug(LogOp, "right_right_core_halo_offset: {}", writer_rt_args[39]);
            // log_debug(LogOp, "left_going_halo_pad_i_offset: {}", writer_rt_args[40]);
            // log_debug(LogOp, "right_going_halo_pad_i_offset: {}", writer_rt_args[41]);
        }

        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

        in_stick_start += in_nsticks_per_core;
        out_stick_start += out_nsticks_per_core;
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        // TODO: support cliff core
        TT_ASSERT(false);
    }

    auto override_runtime_args_callback = [
        reader_kernel_id=reader_kernel_id,
        writer_kernel_id=writer_kernel_id,
        ncores=ncores,
        ncores_x=ncores_x
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core = {i % ncores_x, i / ncores_x};
            // in and out are sharded
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void UntilizeWithHalo::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Input tensor is not TILE for untilize");
    TT_ASSERT(input_tensor_a.memory_config().is_sharded());
    TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only works for sharded input");
    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);

    // Only stride 1 and 2 mode are supported
    TT_ASSERT(stride_ == 1 || stride_ == 2, "Only stride 1 and stride 2 modes are supported.");
}

std::vector<Shape> UntilizeWithHalo::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.shape();
    Shape output_shape = input_shape;
    // pad_h, pad_w
    // calculate the sizes (num sticks) for each of the 7 sections (5 local, 2 halo)
    // output num sticks:
    // local sections:
    // 1. (partial first row width + pad_w)
    // 2. (out_w + pad_w * 2) * (num full rows partial top image)
    // 3. (out_w + pad_w * 2) * (pad_h + out_h) * num full images
    // 4. (out_w + pad_w * 2) * (pad_h + num full rows partial bottom image)
    // 5. (partial last row width + pad_w)
    // halo sections on local core:
    // A. left halo: out_w + pad_w + 1
    // B. right halo: out_w + pad_w + 1
    // corresponding halo sections on neighbors
    // Aa. left left halo:
    // Ab. left halo:
    // Ba. right halo:
    // Bb. right right halo:

    // calculate total number of sticks including all padding, excluding halo
    uint32_t out_cb_pagesize = 2 * input_shape[3];
    uint32_t nbatch = input_shape[0];
    uint32_t in_hw = input_shape[2];
    uint32_t in_h = std::sqrt(in_hw);
    uint32_t in_w = in_h;
    uint32_t in_nhw = nbatch * in_hw;
    uint32_t total_data_nsticks = in_nhw                        // input data sticks
                                    + (in_h * nbatch) * 2       // 2 padding sticks per row
                                    + nbatch * (in_w + 2)       // one padding row on top (incl. edge padding sticks) per batch.
                                    + (in_w + 2);               // one padding row at the end
    uint32_t halo_nsticks = in_w + 1 + 2;                       // size of halo = in_w + 1, and 2 padding for left and right edge
    // get ncores from shard shape and input shape
    // number of halos is at most 2 * ncores (largest case)
    auto shard_shape = input.shard_spec().value().shard_shape;
    uint32_t ncores = in_nhw / shard_shape[0];
    uint32_t total_halo_nsticks = (2 * ncores - 2) * halo_nsticks;
    uint32_t total_nsticks = total_halo_nsticks + total_data_nsticks;
    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[3] remains same
    // output_shape[2] changes
    output_shape[2] = total_nsticks / nbatch;
    output_shape[2] = ceil(output_shape[2] / ncores) * ncores;

    log_debug(LogOp, "output_shape: {} {} {} {}", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(LogOp, "derived ncores: {}", ncores);

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHalo::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    uint32_t ncores = input_tensor.shape()[0] * input_tensor.shape()[2] / shard_spec.shard_shape[0];
    shard_spec.shard_shape[0] = output_shape[2] / ncores;
    // log_debug(LogOp, "derived ncores: {}", ncores);
    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::ROW_MAJOR, input_tensor.device(), output_mem_config_, shard_spec)};
}

operation::ProgramWithCallbacks UntilizeWithHalo::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (stride_) {
        case 1:
            return { untilize_with_halo_multi_core(input_tensor_a, output_tensor, pad_val_) };
        case 2:
            return { untilize_with_halo_multi_core_s2(input_tensor_a, output_tensor, pad_val_) };
        default:
            TT_ASSERT(false, "Unsupported stride value");
    };
    return {};
}

tt::stl::reflection::Attributes UntilizeWithHalo::attributes() const {
    return {
        {"pad_val", pad_val_},
        {"output_mem_config", output_mem_config_},
    };
}

Tensor untilize_with_halo(const Tensor &input_tensor_a, const uint32_t pad_val, const uint32_t stride, const MemoryConfig& mem_config) {
    return operation::run_without_autoformat(UntilizeWithHalo{pad_val, stride, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
