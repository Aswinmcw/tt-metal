// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

inline bool need_to_do_mask_h(uint32_t tile_idx, uint32_t ht, uint32_t wt) { return (((tile_idx / wt) + 1) % ht) == 0; }

namespace NAMESPACE {
void MAIN {
    const auto num_tiles = get_arg_val<uint32_t>(0);
    const auto p = get_arg_val<uint32_t>(1);
    const bool p_is_negative = get_arg_val<uint32_t>(2) == 1;
    const auto origin_h = get_arg_val<uint32_t>(3);
    const auto origin_w = get_arg_val<uint32_t>(4);

    constexpr auto cb_x = tt::CB::c_in0;         // input(==x)
    constexpr auto cb_one = tt::CB::c_in1;       // one
    constexpr auto cb_decimal = tt::CB::c_in2;   // decimal
    constexpr auto cb_mask_h_w = tt::CB::c_in3;  // mask_h_w

    constexpr auto cb_y = tt::CB::c_out0;  // output(==y)

    constexpr auto cb_xabs = tt::CB::c_intermed0;      // |x|
    constexpr auto cb_xpow = tt::CB::c_intermed1;      // |x|^p
    constexpr auto cb_xpowadd = tt::CB::c_intermed2;   // Add[|x|^p * exp(log(|x|) * decimal)]
    constexpr auto cb_logx = tt::CB::c_intermed3;      // log(|x|)
    constexpr auto cb_exp_lxmd = tt::CB::c_intermed4;  // exp(log(|x|) * decimal)
    constexpr auto cb_last = tt::CB::c_intermed5;      // |x|^p * exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    const auto ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto wt = (origin_w + TILE_W - 1) / TILE_W;

    binary_op_init_common(cb_logx, cb_decimal);

    cb_wait_front(cb_decimal, onetile);  // comes from the reader
    cb_wait_front(cb_one, onetile);      // comes from the reader

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, 2);  // comes from the reader
    }

    // Compute cb_xpowadd
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Comput cb_xabs and mask(optional)
        // |x|
        ACQ();
        cb_wait_front(cb_x, onetile);  // comes from the reader
        cb_reserve_back(cb_xabs, onetile);

        copy_tile_init();
        copy_tile(cb_x, 0, dst0);

        if (do_mask_h && need_to_do_mask_h(tile_idx, ht, wt)) {
            copy_tile_init();
            copy_tile(cb_mask_h_w, 0, dst1);

            mask_tile_init();
            mask_tile(dst0, dst1);
        }

        if (do_mask_w && ((tile_idx + 1) % wt) == 0) {
            copy_tile_init();
            copy_tile(cb_mask_h_w, 1, dst1);

            mask_tile_init();
            mask_tile(dst0, dst1);
        }

        abs_tile_init();
        abs_tile(dst0);

        pack_tile(dst0, cb_xabs);

        cb_pop_front(cb_x, onetile);
        cb_push_back(cb_xabs, onetile);
        REL();

        // Compute cb_logx
        // log(|x|)
        ACQ();
        cb_wait_front(cb_xabs, onetile);
        cb_reserve_back(cb_logx, onetile);

        copy_tile_init();
        copy_tile(cb_xabs, 0, dst0);

        log_tile_init();
        log_tile(dst0);

        pack_tile(dst0, cb_logx);

        cb_push_back(cb_logx, onetile);
        REL();
        // We don't pop cb_xabs here.

        // Compute cb_exp_lxmd
        // exp(log(|x|) * decimal)
        ACQ();
        cb_wait_front(cb_logx, onetile);
        cb_reserve_back(cb_exp_lxmd, onetile);

        mul_tiles_init();
        mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

        exp_tile_init();
        exp_tile(dst0);

        pack_tile(dst0, cb_exp_lxmd);

        cb_pop_front(cb_logx, onetile);
        cb_push_back(cb_exp_lxmd, onetile);
        REL();

        // Compute cb_xpow
        // |x|^p
        ACQ();
        cb_reserve_back(cb_xpow, onetile);

        copy_tile_init();
        copy_tile(cb_xabs, 0, dst0);

        power_tile_init();
        power_tile(dst0, p);

        if (p_is_negative) {
            recip_tile_init();
            recip_tile(dst0);
        }

        pack_tile(dst0, cb_xpow);

        cb_pop_front(cb_xabs, onetile);
        cb_push_back(cb_xpow, onetile);
        REL();

        // Compute cb_last
        // |x|^p * exp(log(|x|) * decimal)
        ACQ();
        cb_wait_front(cb_xpow, onetile);
        cb_wait_front(cb_exp_lxmd, onetile);
        cb_reserve_back(cb_last, onetile);

        mul_tiles_init();
        mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

        pack_tile(dst0, cb_last);

        cb_pop_front(cb_xpow, onetile);
        cb_pop_front(cb_exp_lxmd, onetile);
        cb_push_back(cb_last, onetile);
        REL();

        if (tile_idx == 0) {
            ACQ();
            cb_wait_front(cb_last, onetile);
            cb_reserve_back(cb_xpowadd, onetile);

            copy_tile_init();
            copy_tile(cb_last, 0, dst0);

            pack_tile(dst0, cb_xpowadd);

            cb_pop_front(cb_last, onetile);
            cb_push_back(cb_xpowadd, onetile);
            REL();
        } else {
            ACQ();
            cb_wait_front(cb_last, onetile);
            cb_wait_front(cb_xpowadd, onetile);
            cb_reserve_back(cb_xpowadd, onetile);

            add_tiles_init();
            add_tiles(cb_last, cb_xpowadd, 0, 0, dst0);

            pack_tile(dst0, cb_xpowadd);

            cb_pop_front(cb_last, onetile);
            cb_pop_front(cb_xpowadd, onetile);
            cb_push_back(cb_xpowadd, onetile);
            REL();
        }
    }
    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_one, onetile);
    if (do_mask_h || do_mask_w) {
        cb_pop_front(cb_mask_h_w, 2);
    }

    // Compute cb_y
    ACQ();
    cb_wait_front(cb_xpowadd, onetile);
    cb_reserve_back(cb_y, onetile);

    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
    reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xpowadd, cb_one, 0, 0, dst0);
    reduce_revert_delta();

    pack_tile(dst0, cb_y);

    cb_pop_front(cb_xpowadd, onetile);
    cb_push_back(cb_y, onetile);
    REL();
}  // void MAIN
}  // namespace NAMESPACE
