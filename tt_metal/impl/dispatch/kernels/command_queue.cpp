// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue.hpp"

#include "debug_print.h"

void kernel_main() {
    Buffer buffer;
    // Read command from host command queue... l1 read addr since
    // pulling in the actual command into l1
    static constexpr u32 command_start_addr = UNRESERVED_BASE; // Space between UNRESERVED_BASE -> data_start is for commands
    static constexpr u32 data_start_addr = DEVICE_COMMAND_DATA_ADDR;

    // These are totally temporary until PK checks in his changes for
    // separating kernels from firmware
    noc_prepare_deassert_reset_flag(DEASSERT_RESET_SRC_L1_ADDR);
    noc_prepare_assert_reset_flag(ASSERT_RESET_SRC_L1_ADDR);

    // Write my own NOC address to local L1 so that when I dispatch kernels,
    // they will know how to let me know they have finished
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(DISPATCH_MESSAGE_REMOTE_SENDER_ADDR) = get_noc_addr(DISPATCH_MESSAGE_ADDR);

    while (true) {
        volatile tt_l1_ptr u32* command_ptr = reinterpret_cast<volatile tt_l1_ptr u32*>(command_start_addr);

        cq_wait_front();
        // Hardcoded for time being, need to clean this up
        u64 src_noc_addr = get_noc_addr(PCIE_NOC_X, PCIE_NOC_Y, cq_read_interface.fifo_rd_ptr << 4);
        noc_async_read(src_noc_addr, u32(command_start_addr), NUM_16B_WORDS_IN_DEVICE_COMMAND << 4);
        noc_async_read_barrier();

        // Control data
        u32 wrap = command_ptr[0];

        if (wrap) {
            // Basically popfront without the extra conditional
            cq_read_interface.fifo_rd_ptr = 6; // Head to beginning of command queue
            notify_host_of_cq_read_toggle();
            notify_host_of_cq_read_pointer();
            continue;
        }

        u32 finish = command_ptr[1];              // Whether to notify the host that we have finished
        u32 num_workers = command_ptr[2];         // If num_workers > 0, it means we are launching a program
        u32 num_multicast_messages = command_ptr[3];
        u32 data_size_in_bytes = command_ptr[4];  // The amount of trailing data after the device command rounded to the
                                                  // nearest multiple of 32
        u32 num_buffer_reads = command_ptr[5];    // How many ReadBuffer commands we are running
        u32 num_buffer_writes = command_ptr[6];   // How many WriteBuffer commands we are running
        u32 num_program_srcs = command_ptr[7];

        command_ptr = reinterpret_cast<volatile u32*>(command_start_addr + (CONTROL_SECTION_NUM_ENTRIES + NUM_DISPATCH_CORES) * sizeof(u32));
        read_buffers(num_buffer_reads, command_ptr, buffer);
        write_buffers(num_buffer_writes, command_ptr, buffer);

        command_ptr = reinterpret_cast<volatile u32*>(command_start_addr + (CONTROL_SECTION_NUM_ENTRIES + NUM_DISPATCH_CORES + NUM_DATA_MOVEMENT_INSTRUCTIONS * NUM_ENTRIES_PER_BUFFER_RELAY) * sizeof(u32));
        write_program(num_program_srcs, command_ptr, buffer);

        command_ptr = reinterpret_cast<volatile u32*>(command_start_addr + (CONTROL_SECTION_NUM_ENTRIES) * sizeof(u32));
        launch_program(num_workers, num_multicast_messages, command_ptr);

        finish_program(finish);

        // This tells the dispatch core how to update its read pointer
        cq_pop_front(data_size_in_bytes + DeviceCommand::size_in_bytes());
    }
}
