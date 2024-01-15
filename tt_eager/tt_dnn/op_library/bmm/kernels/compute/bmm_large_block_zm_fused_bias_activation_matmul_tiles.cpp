// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

namespace NAMESPACE {

FORCE_INLINE void reload_from_cb_to_dst(uint32_t in1_cb_id, uint32_t mm_partials_cb_id, uint32_t out_subblock_num_tiles) {
    // Reconfigure input
    copy_tile_to_dst_init_short();
    unpack_reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        copy_tile(mm_partials_cb_id, i, i);
    }
    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_init_short();
    unpack_reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
}

void MAIN {

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0); // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4); // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5); //out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6); // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);  // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8); // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9); // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10); // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(11); // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(12); // number of tiles in out_block

    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id = tt::CB::c_in1;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;
    constexpr uint32_t mm_partials_cb_id = tt::CB::c_intermed0;

    constexpr uint32_t mm_out_cb_id = out_cb_id;

    constexpr bool spill = num_blocks > 1;

    UNPACK(( DPRINT << "in0_block_w = " << in0_block_w << " in0_num_subblocks = " << in0_num_subblocks << ENDL() ));
    UNPACK(( DPRINT << "in0_block_num_tiles = " << in0_block_num_tiles << " in0_subblock_num_tiles = " << in0_subblock_num_tiles << ENDL() ));
    UNPACK(( DPRINT << "in1_num_subblocks = " << in1_num_subblocks << " in1_block_num_tiles = " << in1_block_num_tiles << ENDL() ));
    UNPACK(( DPRINT << "in1_per_core_w = " << in1_per_core_w << " num_blocks = " << num_blocks << ENDL() ));
    UNPACK(( DPRINT << "out_subblock_h = " << out_subblock_h << " out_subblock_w = " << out_subblock_w << ENDL() ));
    UNPACK(( DPRINT << "out_subblock_num_tiles = " << out_subblock_num_tiles << " batch = " << batch << ENDL() ));
    UNPACK(( DPRINT << "in0_cb_id = " << in0_cb_id << " in1_cb_id = " << in1_cb_id << ENDL() ));
    UNPACK(( DPRINT << "out_cb_id = " << out_cb_id << " mm_partials_cb_id = " << mm_partials_cb_id << ENDL() ));
    UNPACK(( DPRINT << "mm_out_cb_id = " << mm_out_cb_id << " spill = " << (int)spill << ENDL() ));

    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id);
    for (uint32_t b = 0; b < batch; b++){
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        MATH(( DPRINT << "START BATCH ->  " << b << " out of  " << batch << ENDL() ));

        for(uint32_t block = 0; block < num_blocks; block++)
        {
            MATH(( DPRINT << "START BLOCK ->  " << block << " out of  " << num_blocks << ENDL() ));

            bool last_out = block == (num_blocks-1);

            cb_wait_front(in0_cb_id, in0_block_num_tiles);
            cb_wait_front(in1_cb_id, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                MATH(( DPRINT << "START IN0 SUBBLOCK ->  " << in0_subblock << " out of  " << in0_num_subblocks << ENDL() ));
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {

                    // just acquire
                    tile_regs_acquire();

                    // Compute output sub-block from in0_subblock x in1_subblock
                    // int dst_index = 0;
                    // int in0_index_h_offset = 0;
                    // PACK(( DPRINT << "START INNER SUBBLOCK MATMUL -> in0_subblock = " << in0_subblock << " in1_subblock = " << in1_subblock << ENDL() ));
                    // for (uint32_t h = 0; h < out_subblock_h; h++) {
                    //     for (uint32_t w = 0; w < out_subblock_w; w++) {
                    //         int in1_index_inner_dim_offset = 0;
                    //         for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                    //             int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                    //             int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                    //             matmul_tiles(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, false /* transpose */);
                    //             in1_index_inner_dim_offset += in1_per_core_w;
                    //         }
                    //         dst_index++;
                    //     }
                    //     in0_index_h_offset += in0_block_w;
                    // }
                    // PACK(( DPRINT << "END INNER SUBBLOCK MATMUL -> in0_subblock = " << in0_subblock << " in1_subblock = " << in1_subblock << ENDL() ));

                    // Compute output sub-block
                    uint32_t dst_index = 0; // start at 0, each call to matmul_block internally increments dst_index

                    // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w

                    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                        uint32_t in0_index = in0_subblock*out_subblock_h*in0_block_w + inner_dim_idx;
                        uint32_t in1_index = in1_subblock*out_subblock_w*in0_block_w + inner_dim_idx*out_subblock_w; // offset into in1 block

                        // MATH(( DPRINT << "START -> inner_dim_idx = " << inner_dim_idx << " in0_index = " << in0_index << " in1_index = " << in1_index << ENDL() ));
                        // MATH(( DPRINT << "in0_index = " << in0_index << " in1_index = " << in1_index << ENDL() ));
                        // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                        // accumulation is done by iterating matmul_block across inner dim
                        // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                        matmul_block(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, false, out_subblock_w, out_subblock_h, in0_block_w);
                        // in0_index ++;  // stride right by 1
                        // // in1_index += out_subblock_w; // to stride down by 1 need to stride by out_subblock_w
                        // in1_index += in1_per_core_w; // to stride down by 1 need to stride by in_per_core_w (should be called in1_block_w)
                        // MATH(( DPRINT << "END -> inner_dim_idx = " << inner_dim_idx << " in0_index = " << in0_index << " in1_index = " << in1_index << ENDL() ));
                    }


                    if (last_out) {
                        tile_regs_commit();
                        // Pack out to output buffer
                        cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, mm_out_cb_id);
                        }
                        tile_regs_release();
                        cb_push_back(mm_out_cb_id, out_subblock_num_tiles);
                    } else {
                        tile_regs_commit();
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, mm_partials_cb_id);
                        }
                        tile_regs_release();
                        cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
                MATH(( DPRINT << "END IN0 SUBBLOCK ->  " << in0_subblock << " out of  " << in0_num_subblocks << ENDL() ));
            }

            if (spill) enable_reload = true;

            cb_pop_front(in0_cb_id, in0_block_num_tiles);
            cb_pop_front(in1_cb_id, in1_block_num_tiles);
            MATH(( DPRINT << "END BLOCK ->  " << block << " out of  " << num_blocks << ENDL() ));

        }
        MATH(( DPRINT << "END BATCH ->  " << b << " out of  " << batch << ENDL() ));
    }
}
}
