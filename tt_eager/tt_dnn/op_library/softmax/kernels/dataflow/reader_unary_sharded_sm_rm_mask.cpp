// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "dataflow_api.h"

FORCE_INLINE void generate_bcast_scaler() {
    constexpr auto cb_bcast_scaler = tt::CB::c_in1;
    uint32_t scaler = get_arg_val<uint32_t>(0);
    cb_reserve_back(cb_bcast_scaler, 1);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_bcast_scaler));
    uint32_t idx = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        uint32_t curr_idx = idx;
        for (uint32_t j = 0; j < 8; ++j) {
            ptr[curr_idx] = scaler;
            curr_idx++;
        }
        idx += 128;
    }
    cb_push_back(cb_bcast_scaler, 1);
}

// HW-bcast scale for fused scale-attn-softmax
FORCE_INLINE void generate_inv_sqrt_hw_bcast_tile() {
    constexpr auto cb_fused_scale = tt::CB::c_in2;
    uint32_t u = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_fused_scale, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_fused_scale));
    ptr[0] = u>>16;
    cb_push_back(cb_fused_scale, 1);
}

void kernel_main() {

    #if FUSED_SCALE_MASK
    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr bool is_dram_mask = get_compile_time_arg_val(1) == 1;
    const uint32_t mask_addr  = get_arg_val<uint32_t>(2);
    const uint32_t mask_start_tile_id  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_attn = tt::CB::c_in3;
    uint32_t mask_tile_bytes = get_tile_size(cb_attn);

    #define stick_size_is_pow2 get_compile_time_arg_val(2) == 1
    #if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(3);
    #else
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    #endif
    #if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<is_dram_mask> addr_mask = {
        .bank_base_address = mask_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size
    };
    #else
    const InterleavedAddrGen<is_dram_mask> addr_mask = {
        .bank_base_address = mask_addr,
        .page_size = page_size
    };
    #endif

    generate_inv_sqrt_hw_bcast_tile();

    cb_reserve_back(cb_attn, block_wt);
    uint32_t l1_write_addr = get_write_ptr(cb_attn);
    for (uint32_t w = 0; w<block_wt; w++) {
        uint64_t mask_noc_addr = get_noc_addr(mask_start_tile_id + w, addr_mask);
        noc_async_read(mask_noc_addr, l1_write_addr, 32);
        mask_noc_addr += 32;
        noc_async_read(mask_noc_addr, l1_write_addr + 512, 32);
        l1_write_addr += mask_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_attn, block_wt);
    #endif

    generate_bcast_scaler();
}
