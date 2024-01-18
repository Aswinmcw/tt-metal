// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// XXXX fix for WH
#if defined(FROM_PCIE)
#define NOC_ADDR_X 0
#define NOC_ADDR_Y 4
#elif defined(FROM_DRAM)
#define NOC_ADDR_X 1
#define NOC_ADDR_Y 6
#elif defined(FROM_L1)
#define NOC_ADDR_X 0
#define NOC_ADDR_Y 0
#endif

void kernel_main() {
    cb_reserve_back(0, PAGE_COUNT);
    uint32_t cb_addr = get_write_ptr(0);
    uint64_t noc_addr = get_noc_addr(NOC_ADDR_X, NOC_ADDR_Y, 0);
    for (int i = 0; i < ITERATIONS; i++) {
        uint32_t read_ptr = cb_addr;
        for (int j = 0; j < PAGE_COUNT; j++) {
            noc_async_read(noc_addr, read_ptr, PAGE_SIZE);
#if defined(LATENCY)
            noc_async_read_barrier();
#endif
            read_ptr += PAGE_SIZE;
        }
    }
#if !defined(LATENCY)
    noc_async_read_barrier();
#endif
}
