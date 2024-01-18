// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"
// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_addr = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr   = get_compile_time_arg_val(1);
    constexpr uint32_t num_mcast_cores                = get_compile_time_arg_val(2);
    constexpr uint32_t num_group_batch                = get_compile_time_arg_val(3);

    const bool has_mcast_first_group                    = get_arg_val<uint32_t>(0);
    const bool has_mcast_last_group                     = get_arg_val<uint32_t>(1);

    const uint32_t mcast_dest_noc_start_x               = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_start_y               = get_arg_val<uint32_t>(3);
    const uint32_t mcast_dest_noc_end_x                 = get_arg_val<uint32_t>(4);
    const uint32_t mcast_dest_noc_end_y                 = get_arg_val<uint32_t>(5);

    uint32_t mcast_first_group_dest_noc_start_x;
    uint32_t mcast_first_group_dest_noc_start_y;
    uint32_t mcast_first_group_dest_noc_end_x;
    uint32_t mcast_first_group_dest_noc_end_y;
    uint32_t mcast_last_group_dest_noc_start_x;
    uint32_t mcast_last_group_dest_noc_start_y;
    uint32_t mcast_last_group_dest_noc_end_x;
    uint32_t mcast_last_group_dest_noc_end_y;
    volatile tt_l1_ptr uint32_t * noc_coord;

    uint64_t reduce_sender_first_group_semaphore_noc_addr;
    uint64_t multicast_first_group_data_noc;
    uint64_t reduce_sender_last_group_semaphore_noc_addr;
    uint64_t multicast_last_group_data_noc;

    if (has_mcast_first_group and has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x               = get_arg_val<uint32_t>(6);
        mcast_first_group_dest_noc_start_y               = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_end_x                 = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_y                 = get_arg_val<uint32_t>(9);

        mcast_last_group_dest_noc_start_x               = get_arg_val<uint32_t>(10);
        mcast_last_group_dest_noc_start_y               = get_arg_val<uint32_t>(11);
        mcast_last_group_dest_noc_end_x                 = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_end_y                 = get_arg_val<uint32_t>(13);

        noc_coord        = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(14));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x               = get_arg_val<uint32_t>(6);
        mcast_first_group_dest_noc_start_y               = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_end_x                 = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_y                 = get_arg_val<uint32_t>(9);

        noc_coord        = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(10));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x               = get_arg_val<uint32_t>(6);
        mcast_last_group_dest_noc_start_y               = get_arg_val<uint32_t>(7);
        mcast_last_group_dest_noc_end_x                 = get_arg_val<uint32_t>(8);
        mcast_last_group_dest_noc_end_y                 = get_arg_val<uint32_t>(9);

        noc_coord        = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(10));

    } else {
        noc_coord        = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(6));
    }

    const uint64_t reduce_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        reduce_sender_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        0);

    if (has_mcast_first_group) {
        reduce_sender_first_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_first_group_data_noc = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            0);
    }
    if (has_mcast_last_group) {
        reduce_sender_last_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_last_group_data_noc = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            0);
    }

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    *reduce_sender_semaphore_addr_ptr = VALID;
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0;
    constexpr uint32_t cb_ex = tt::CB::dataflow1;
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3;
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4;
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    const DataFormat data_format = get_dataformat(cb_ex_partial);

    DPRINT << "Reader Sender !! num_cores_per_mcast_group " << num_cores_per_mcast_group << ENDL();

    if constexpr(num_mcast_cores > 1) {
        for (int bg=0; bg < num_group_batch; ++bg) {
            // wait for local data ready
            cb_wait_front(cb_ex_partial, 1);
            // wait for all other cores data ready
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_mcast_cores-1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);

            // read data from other cores
            uint32_t l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
            uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external);
            for(uint32_t i = 0; i < num_mcast_cores; ++i) {
                cb_reserve_back(cb_ex_external, 1);
                uint64_t noc_addr_ex_par = get_noc_addr(noc_coord[i*2], noc_coord[i*2+1], l1_read_addr_ex_par);
                noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
                noc_async_read_barrier();
                cb_push_back(cb_ex_external, 1);
            }

            // wait for global reduce done
            uint32_t l1_write_addr_ex = get_write_ptr(cb_ex);
            uint32_t l1_write_addr_ex_global = get_write_ptr(cb_ex_global);
            cb_wait_front(cb_ex, 1);

            // mcast to other cores
            uint32_t l1_read_addr_ex_global = get_read_ptr(cb_ex_global);
            noc_async_write_multicast(l1_read_addr_ex_global, multicast_data_noc | l1_read_addr_ex_global, 1, num_mcast_cores-1, true);
            noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_mcast_cores-1, false);
            noc_async_write_barrier();
        }
    }
}
