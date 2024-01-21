// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr            = get_arg_val<uint32_t>(i++);
    uint32_t src1_addr            = get_arg_val<uint32_t>(i++);
    uint32_t Mt                   = get_arg_val<uint32_t>(i++);
    uint32_t Kt                   = get_arg_val<uint32_t>(i++);
    uint32_t Nt                   = get_arg_val<uint32_t>(i++);
    uint32_t MtKt                 = get_arg_val<uint32_t>(i++); // if 0
    uint32_t in1_KtNt_skip        = get_arg_val<uint32_t>(i++); // 0 if in0 and in1 Kt are the same
    uint32_t in1_KtNt_mul_32      = get_arg_val<uint32_t>(i++);
    uint32_t blocks               = get_arg_val<uint32_t>(i++);
    uint32_t in0_start_id         = get_arg_val<uint32_t>(i++);
    uint32_t in1_start_id         = get_arg_val<uint32_t>(i++);

    uint32_t act_mcast_dest_noc_start_x                  = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_start_y                  = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_end_x                    = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_end_y                    = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_num_dests                         = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_num_cores                         = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_semaphore_addr             = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_receiver_semaphore_addr           = get_arg_val<uint32_t>(i++);

    uint32_t act_mcast_sender_size_bytes                 = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_id                         = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_noc_x                      = get_arg_val<uint32_t>(i++);
    volatile tt_l1_ptr uint32_t *act_mcast_sender_noc_y  = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(i));

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    #define transpose_hw_bool get_compile_time_arg_val(2) == 1

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_intermed0 = 24;
    constexpr uint32_t cb_id_intermed1 = 25;
    constexpr uint32_t cb_id_intermed2 = 26;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = in0_tile_bytes,
        .data_format = in0_data_format
    };

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = in1_tile_bytes,
        .data_format = in1_data_format
    };

    uint32_t in0_batch = in0_start_id;
    uint32_t in1_batch;
    uint32_t in0_Mt;
    uint32_t in1_Nt;
    uint32_t in0_tensor_id;
    uint32_t in1_tensor_id;

    uint32_t cb_intermed1_addr_initial = get_read_ptr(cb_id_intermed1);
    uint32_t cb_intermed2_addr_initial = get_write_ptr(cb_id_intermed2);
    uint32_t cb_intermed1_addr;
    uint32_t cb_intermed2_addr;
    constexpr uint32_t bfloat16_row_bytes = 64;
    constexpr uint32_t num_rows_in_one_tile = 32;

    for (uint32_t b = 0; b < blocks; b++) {
        in0_Mt = in0_batch;
        in1_batch = in1_start_id;

    for (uint32_t m = 0; m < Mt; m++) {
        in1_Nt = in1_batch;

    for (uint32_t n = 0; n < Nt; n++) {
        cb_intermed1_addr = cb_intermed1_addr_initial;
        cb_intermed2_addr = cb_intermed2_addr_initial;
        in0_tensor_id = in0_Mt;
        in1_tensor_id = in1_Nt;

        cb_reserve_back(cb_id_in0, Kt);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Read A's tile at (mt, kt)
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(in0_tensor_id, s0, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);

            in0_tensor_id++; // A is MK
        }

        cb_reserve_back(cb_id_intermed2, 1);
        for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Read B's tile at (kt, nt)
                cb_reserve_back(cb_id_in1, onetile);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(in1_tensor_id, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);

                #if (transpose_hw_bool)
                in1_tensor_id++; // Kt is in B[3], so it is contiguous in memory
                #else
                in1_tensor_id += Nt; // Kt is in B[2], so stride is Nt
                #endif
            } // Kt loop

            // Read 32 untilized tiles and select correct rows to reconstruct single correct tile
            cb_wait_front(cb_id_intermed1, 1);
            noc_async_read(get_noc_addr(cb_intermed1_addr), cb_intermed2_addr, bfloat16_row_bytes);
            noc_async_read_barrier();
            cb_pop_front(cb_id_intermed1, 1);
            cb_intermed1_addr += bfloat16_row_bytes;
            cb_intermed2_addr += bfloat16_row_bytes;

            in1_tensor_id += in1_KtNt_skip; // different depending on transpose_hw
        } // 32 tiles loop
        cb_push_back(cb_id_intermed2, 1);

        // Next tile in Nt
        #if (transpose_hw_bool)
        in1_Nt += Kt; // next tile in Nt is in B[2], so stride is Kt
        #else
        in1_Nt++;
        #endif
    } // Nt loop

    in0_Mt += Kt;
    // here, KtNt is the stride of the full B tensor (ie. max cache length is incorporated in one of Kt or Nt depending on transpose_hw)
    in1_batch += in1_KtNt_mul_32; // different depending on transpose_hw
    } // Mt loop

    in0_batch += MtKt;
    } // B loop
}
