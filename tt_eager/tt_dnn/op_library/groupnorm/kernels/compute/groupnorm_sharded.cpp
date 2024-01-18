// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/matmul.h"

#include "debug/dprint.h"


inline void tilize_in(
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    uint32_t block_h,
    uint32_t block_w
) {
    tilize_init_short(in_cb_id, block_w);
    for (uint32_t h = 0; h < block_h; ++h) {
        cb_reserve_back(out_cb_id, block_w);
        tilize_block(in_cb_id, block_w, out_cb_id);
        cb_push_back(out_cb_id, block_w);
        cb_pop_front(in_cb_id, block_w);
    }
    tilize_uninit();
}

inline void untilize_out(
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    uint32_t block_h,
    uint32_t block_w
) {
    untilize_init_short(in_cb_id);
    for (uint32_t h = 0; h < block_h; ++h) {
        cb_wait_front(in_cb_id, block_w);
        cb_reserve_back(out_cb_id, block_w);
        untilize_block(in_cb_id, block_w, out_cb_id);
        cb_pop_front(in_cb_id, block_w);
        cb_push_back(out_cb_id, block_w);
    }
    untilize_uninit(in_cb_id);
}




// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t is_mcast_sender                = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma                       = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta                        = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group      = get_compile_time_arg_val(3);

    constexpr uint32_t batch                          = get_compile_time_arg_val(4);
    constexpr uint32_t group                          = get_compile_time_arg_val(5);

    constexpr uint32_t num_batch_group                = get_compile_time_arg_val(6);

    constexpr uint32_t block_h                        = get_compile_time_arg_val(7);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(8);
    constexpr uint32_t block_hw                       = get_compile_time_arg_val(9);

    constexpr uint32_t subblock_w                     = get_compile_time_arg_val(10);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(11);

    constexpr uint32_t tilize_in0                      = get_compile_time_arg_val(12);

    constexpr uint32_t per_core_M                       = get_compile_time_arg_val(13);
    constexpr uint32_t per_core_N                       = get_compile_time_arg_val(14);


    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_eps = tt::CB::c_in3;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_x = tt::CB::c_intermed0; // x minus mean
    constexpr uint32_t cb_xmm = tt::CB::c_intermed1; // x minus mean
    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    // constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    // constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_xmm2 = cb_x; // xmm^2
    // constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3; // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = cb_xmm; // stream gamma/beta
    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t cb_ex_global = num_cores_per_mcast_group == 1 ? cb_ex_partial : tt::CB::dataflow7;
    constexpr uint32_t cb_ex2 = num_cores_per_mcast_group == 1 ? cb_ex_partial2 : tt::CB::dataflow4;
    constexpr uint32_t cb_ex2pe = num_cores_per_mcast_group == 1 ? cb_ex_partial : tt::CB::c_intermed3;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index_bg_offset = 0;

    constexpr int cb_in = tilize_in0 ? cb_x : cb_in0;
    constexpr int cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr int cb_outgamma = do_beta ? cb_fusion : cb_out;

    binary_op_init_common(cb_in0, cb_in0, cb_xmm);

    // UNPACK (( DPRINT << "block_h " << block_h << ENDL() ));
    // UNPACK (( DPRINT << "block_w " << block_w << ENDL() ));
    // UNPACK (( DPRINT << "num_batch_group " << num_batch_group << ENDL() ));

    index_bg_offset = 0;
    for (uint32_t b = 0; b < num_batch_group; ++b) {

        if constexpr (tilize_in0) {
            tilize_in(cb_in0, cb_in, block_h, block_w);
            cb_wait_front(cb_in, block_hw);
        }

        // Partial-E[x] for each core,
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_wait_front(cb_scaler, 1);
        cb_reserve_back(cb_ex_partial, 1);
        index_h_offset = 0;
        tile_regs_acquire();
        for (uint32_t h = 0; h < block_h; ++h) {
            for (uint32_t w = 0; w < block_w; ++w) {
                uint32_t index = index_h_offset + w;
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_in, cb_scaler, index, scaler0, dst0);
            }
            index_h_offset += block_w;
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial);
        tile_regs_release();
        reduce_revert_delta();
        cb_push_back(cb_ex_partial, 1);

        // cb_wait_front(cb_ex_partial, 1);
        // UNPACK (( DPRINT << "cb_ex_partial " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_ex_partial, 0, SliceRange::h0_w0_32()) << ENDL() ));
        // UNPACK (( DPRINT << "cb_scaler " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_scaler, 0, SliceRange::h0_w0_32()) << ENDL() ));

        if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
            UNPACK (( DPRINT << "mcast !!! " << ENDL() ));
            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            cb_reserve_back(cb_ex, 1);
            tile_regs_acquire();
            cb_wait_front(cb_scaler_global, 1);
            for (uint32_t w = 0; w < num_cores_per_mcast_group; w++) {
                cb_wait_front(cb_ex_external, 1);
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex);
            tile_regs_release();
            reduce_revert_delta();
            cb_push_back(cb_ex, 1);
        }

        // x - E[x]
        index_h_offset = 0;
        sub_tiles_bcast_scalar_init_short();
        cb_reserve_back(cb_xmm, block_hw);
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t index = w + index_subblock_w_offset;
                    sub_tiles_bcast_scalar(cb_in, cb_ex_global, index, 0, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_xmm);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            cb_pop_front(cb_in, block_w);
        }
        cb_pop_front(cb_ex_global, 1);
        cb_push_back(cb_xmm, block_hw);
        cb_wait_front(cb_xmm, block_hw);

        // UNPACK (( DPRINT << "cb_xmm " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_xmm, 0, SliceRange::h0_w0_32()) << ENDL() ));

        // (x - E[x])^2,
        mul_tiles_init();
        index_h_offset = 0;
        cb_reserve_back(cb_xmm2, block_hw);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t index = w + index_subblock_w_offset + index_h_offset;
                    mul_tiles(cb_xmm, cb_xmm, index, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_xmm2);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_push_back(cb_xmm2, block_hw);
        cb_wait_front(cb_xmm2, block_hw);

        // UNPACK (( DPRINT << "cb_xmm2 " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_xmm2, 0, SliceRange::h0_w0_32()) << ENDL() ));

        // Partial-Var(x)
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_wait_front(cb_scaler, 1);
        cb_reserve_back(cb_ex_partial2, 1);
        index_h_offset = 0;
        tile_regs_acquire();
        for (uint32_t h = 0; h < block_h; ++h) {
            for (uint32_t w = 0; w < block_w; ++w) {
                uint32_t index = index_h_offset + w;
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xmm2, cb_scaler, index, scaler0, dst0);
            }
            index_h_offset += block_w;
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        reduce_revert_delta();
        cb_push_back(cb_ex_partial2, 1);
        cb_pop_front(cb_xmm2, block_hw);

        // cb_wait_front(cb_ex_partial2, 1);
        // UNPACK (( DPRINT << "cb_ex_partial2 " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_ex_partial2, 0, SliceRange::h0_w0_32()) << ENDL() ));

        // global reduce,
        if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            cb_reserve_back(cb_ex2, 1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_cores_per_mcast_group; w++) {
                cb_wait_front(cb_ex_external2, 1);
                reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ex_external2, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external2, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2);
            tile_regs_release();
            reduce_revert_delta();
            cb_push_back(cb_ex2, 1);
        }

        // 1/[sqrt(Var + eps)],
        cb_wait_front(cb_ex2, 1);

        // UNPACK (( DPRINT << "cb_ex2 " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_ex2, 0, SliceRange::h0_w0_32()) << ENDL() ));

        cb_reserve_back(cb_ex2pe, 1);
        tile_regs_acquire();
        add_tiles_init();
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
        tile_regs_wait();
        // sqrt(Var + eps)
        sqrt_tile_init();
        sqrt_tile(dst0);
        tile_regs_wait();
        // 1/[sqrt(Var + eps)]
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        cb_push_back(cb_ex2pe, 1);
        tile_regs_release();

        // UNPACK (( DPRINT << "cb_eps " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_eps, 0, SliceRange::h0_w0_32()) << ENDL() ));

        // UNPACK (( DPRINT << "cb_ex2pe " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_ex2pe, 0, SliceRange::h0_w0_32()) << ENDL() ));

        // (x - Ex) * 1/[sqrt(Var + eps)]
        mul_tiles_bcast_scalar_init_short();
        index_h_offset = 0;
        cb_reserve_back(cb_im, block_hw);
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t index = w + index_subblock_w_offset + index_h_offset;
                    mul_tiles_bcast_scalar(cb_xmm, cb_ex_global, index, 0, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_im);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_push_back(cb_im, block_hw);
        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_xmm, block_hw);
        cb_wait_front(cb_im, block_hw);

        // UNPACK (( DPRINT << "cb_im " << ENDL() ));
        // UNPACK (( DPRINT << TSLICE(cb_im, 0, SliceRange::h0_w0_32()) << ENDL() ));

    }

    // UNPACK (( DPRINT << "UNPACK Done " << ENDL() ));
    // MATH (( DPRINT << "MATH Done " << ENDL() ));
    // PACK (( DPRINT << "PACK Done " << ENDL() ));


}
}
