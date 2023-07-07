#include <stdint.h>
#include "dataflow_kernel_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    kernel_profiler::mark_time(34);
    // in0 tensor args
    uint32_t in0_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id           = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w                = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h                = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride       = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w                        = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h                        = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles                = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w                        = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h                        = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles                = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks                         = get_arg_val<uint32_t>(16);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests                = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x             = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y             = get_arg_val<uint32_t>(23);
    uint32_t in0_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(24);
    uint32_t in0_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(25);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests                = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y             = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(33);
    uint32_t in1_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(34);

    // batch args
    uint32_t MtKt                               = get_arg_val<uint32_t>(35); // if 0
    uint32_t KtNt                               = get_arg_val<uint32_t>(36);
    uint32_t batch                              = get_arg_val<uint32_t>(37);
    uint32_t bcast_B                            = get_arg_val<uint32_t>(38);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    volatile uint32_t* in0_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in0_mcast_receiver_semaphore_addr);
    volatile uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in1_mcast_receiver_semaphore_addr);

    bool one_time_noc_wait_0 = true;
    bool one_time_noc_wait_1 = true;
    bool one_time_cb_push = true;

    for (uint32_t b = 0; b < batch; b++) {
        for(uint32_t block = 0; block < num_blocks; block++) {
            // Operand 0
            dataflow::cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            dataflow_internal::noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            uint64_t in0_mcast_sender_semaphore_noc_addr = dataflow::get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);
            dataflow_internal::noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            dataflow_internal::noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
            kernel_profiler::mark_time_once(35, &one_time_noc_wait_0);

            dataflow::cb_push_back(cb_id_in0, in0_block_num_tiles);

            // Operand 1
            dataflow::cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            // Set in1 semaphore value to INVALID
            dataflow_internal::noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

            uint64_t in1_mcast_sender_semaphore_noc_addr = dataflow::get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);
            dataflow_internal::noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
            dataflow_internal::noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);
            kernel_profiler::mark_time_once(36, &one_time_noc_wait_1);

            dataflow::cb_push_back(cb_id_in1, in1_block_num_tiles);
            kernel_profiler::mark_time_once(37, &one_time_cb_push);
        }
    }
}
