#include <cstdint>

constexpr int MAX_CHANNELS = 8;

inline std::uint64_t get_src_dram_addr_from_txn_index(
    std::uint32_t dram_channels_noc_x[MAX_CHANNELS],
    std::uint32_t dram_channels_noc_y[MAX_CHANNELS],
    std::uint32_t transaction_index,
    std::uint32_t src_addr,
    std::uint32_t num_bits_of_num_channels) {
    std::uint32_t channel_remainder_mask = ~(~0x0 << num_bits_of_num_channels);
    std::uint32_t dram_channel_id = (transaction_index) & channel_remainder_mask;
    return get_noc_addr(dram_channels_noc_x[dram_channel_id], dram_channels_noc_y[dram_channel_id], src_addr);
}

void kernel_main() {
    std::uint32_t   num_bits_of_num_channels = get_arg_val<uint32_t>(0);
    std::uint32_t   num_dram_channels = 1 << num_bits_of_num_channels;

    std::uint32_t   orig_buffer_src_addr     = get_arg_val<uint32_t>(1);

    // Get dram channel coordinates from host, as we don't know them on FW side
    std::uint32_t dram_channels_noc_x[MAX_CHANNELS];
    std::uint32_t dram_channels_noc_y[MAX_CHANNELS];
    std::uint32_t dram_channels_noc_x_raw_start = L1_ARG_BASE + 8;
    std::uint32_t dram_channels_noc_y_raw_start = L1_ARG_BASE + 12;
    for (std::uint32_t dram_channel_id = 0; dram_channel_id < MAX_CHANNELS; dram_channel_id++) {
        dram_channels_noc_x[dram_channel_id] = *(volatile uint32_t*)(dram_channels_noc_x_raw_start + dram_channel_id * 2 * 4);
        dram_channels_noc_y[dram_channel_id] = *(volatile uint32_t*)(dram_channels_noc_y_raw_start + dram_channel_id * 2 * 4);
    }

    std::uint32_t remaining_args_offset = 8 + num_dram_channels * 2 * 4;

    std::uint32_t l1_arg_base_index        = remaining_args_offset >> 2;
    std::uint32_t orig_buffer_dst_addr     = get_arg_val<uint32_t>(l1_arg_base_index);

    std::uint32_t transaction_size         = get_arg_val<uint32_t>(l1_arg_base_index + 1);
    std::uint32_t starting_txn_index       = get_arg_val<uint32_t>(l1_arg_base_index + 2);
    std::uint32_t num_transactions         = get_arg_val<uint32_t>(l1_arg_base_index + 3);
    std::uint32_t num_repetitions          = get_arg_val<uint32_t>(l1_arg_base_index + 4);

    std::uint32_t buffer_src_addr  = orig_buffer_src_addr;
    std::uint32_t buffer_dst_addr  = orig_buffer_dst_addr;
    for (std::uint32_t i=0; i<num_repetitions; i++) {
        std::uint32_t transaction_size_const = transaction_size;

        noc_fast_read_set_len(transaction_size_const);
        for (std::uint32_t j=starting_txn_index; j<(starting_txn_index + num_transactions); j++) {
            // banking destinations as well
            std::uint32_t num_rows_into_dram = j >> num_bits_of_num_channels;
            std::uint32_t buffer_src_addr = num_rows_into_dram * transaction_size_const + orig_buffer_src_addr;
            std::uint64_t buffer_src_noc_addr = get_src_dram_addr_from_txn_index(dram_channels_noc_x, dram_channels_noc_y, j, buffer_src_addr, num_bits_of_num_channels);
            noc_fast_read_set_src_xy(buffer_src_noc_addr);

            noc_fast_read(buffer_src_addr, buffer_dst_addr);
            buffer_dst_addr += transaction_size_const;
        }
        noc_fast_read_inc_num_issued(num_transactions);
        noc_async_read_barrier();

        // reset these to the original value for each repetition
        buffer_dst_addr  = orig_buffer_dst_addr;
    }
}
