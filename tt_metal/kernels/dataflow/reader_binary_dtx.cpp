#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"
void kernel_main() {
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Arguments for in0
    uint32_t in0_addr_base  = get_arg_val<uint32_t>(1);
    uint32_t in0_noc_x = get_arg_val<uint32_t>(2);
    uint32_t in0_noc_y = get_arg_val<uint32_t>(3);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(4);
    uint32_t address_map_l1_addr = get_arg_val<uint32_t>(5);
    uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(6);

    // Arguments for in1
    uint32_t src1_addr  = get_arg_val<uint32_t>(7);
    uint32_t in1_block_w = get_arg_val<uint32_t>(8);
    uint32_t in1_block_h = get_arg_val<uint32_t>(9);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(12);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(13);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(14);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    volatile std::uint32_t* address_map = (volatile uint32_t*)(address_map_l1_addr);
    DPRINT << "A=" << num_blocks << ENDL();

    // Put zeroes in the zero buffer
    constexpr uint32_t num_elements_in_zeros_buffer = l1_mem::address_map::ZEROS_SIZE / sizeof(uint32_t);
    volatile uint32_t* zero_base_ptr = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }
    uint64_t zeros_base_noc_addr = get_noc_addr(l1_mem::address_map::ZEROS_BASE);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const InterleavedPow2AddrGen s1 = {
        .bank_base_address = src1_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .log_base_2_of_bank_unit_size = tile_size_pow2_exponent
    };
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
    uint32_t index = 0;
    DPRINT << "B=" << num_blocks << ENDL();
    for (uint32_t b = 0; b < num_blocks; b += 1) {
        // Read weights
        //DPRINT << "T" << ENDL();
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        //DPRINT << "R" << ENDL();
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for(uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for(uint32_t w = 0; w < in1_block_w; w++) {
                uint64_t in1_tile_noc_addr = get_noc_addr(in1_tensor_tile_id, s1);
                noc_async_read(in1_tile_noc_addr, l1_write_addr_in1, single_tile_size_bytes);
                l1_write_addr_in1 += single_tile_size_bytes;
                in1_tensor_tile_id += in1_tensor_stride_w;
            }
            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
        }
        noc_async_read_barrier();
        // //DPRINT << "O" << ENDL();
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

        // Read from DRAM into L1 using DTX address map and push one block at a time to CB
        //DPRINT << "P" << ENDL();

        //DPRINT << "Q" << ENDL();
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t bytes_read = 0;
        uint32_t num_reads_current_block = address_map[index];
        index += 1;
        for(uint32_t i = 0; i < num_reads_current_block; i+=1) {
            // There are 4 entries in the address map per read
            uint32_t src_address = address_map[index];
            uint32_t dst_address = address_map[index+1];
            uint32_t read_size = address_map[index+2];
            uint32_t pad = address_map[index+3];
            // if (index == 0) {
            // DPRINT << "S=" << src_address << ENDL();
            // DPRINT << "D=" << dst_address << ENDL();
            // DPRINT << "R=" << read_size << ENDL();
            // DPRINT << "pad=" << pad << ENDL();
            // }
            if(pad == 1) {
                // source address is set to max. This refers to padding location.
                // read zeroes from zero buffer
                uint32_t dst_addr = l1_write_addr_in0 + dst_address;
                if (dst_addr < 200 * 1024) {
                    DPRINT << "PROBLEM" << ENDL();
                }
                uint32_t pad_size = read_size;
                if (pad_size <= l1_mem::address_map::ZEROS_SIZE) {
                    noc_async_read(zeros_base_noc_addr, dst_addr, pad_size);
                }
                else {
                    // padding size is bigger than the zero buffer size
                    // read from zero buffer multiple times
                    uint32_t zeros_to_read = pad_size;
                    uint32_t zeros_read_size = l1_mem::address_map::ZEROS_SIZE;
                    while(zeros_to_read != 0) {
                        noc_async_read(zeros_base_noc_addr, dst_addr, zeros_read_size);
                        zeros_to_read -= zeros_read_size;
                        if (zeros_to_read < zeros_read_size) {
                            zeros_read_size = zeros_to_read;
                        }
                    }
                }
            }
            else {
                uint32_t src_addr = in0_addr_base + src_address;
                uint64_t src_noc_addr = get_noc_addr(in0_noc_x, in0_noc_y, src_addr);
                uint32_t dst_addr = l1_write_addr_in0 + dst_address;
                if (dst_addr < 200 * 1024) {
                    DPRINT << "PROBLEM" << ENDL();
                }
                noc_async_read(src_noc_addr, dst_addr, read_size);
            }
            bytes_read += read_size;
            index += 4;
        }
        if(bytes_read != in0_block_size_bytes) {
            DPRINT << "PROBLEM" << ENDL();
        }
        noc_async_read_barrier();
        //DPRINT << "S" << ENDL();
        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
