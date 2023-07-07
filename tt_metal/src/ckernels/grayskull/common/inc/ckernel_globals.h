#pragma once

#include <cstdint>
#include "ckernel_structs.h"
#include "tensix_functions.h"
#include "hostdevcommon/common_runtime_address_map.h"

extern uint32_t cfg_state_id;
extern uint32_t unp_cfg_context;
extern uint32_t gl_alu_format_spec_reg;

extern volatile uint32_t l1_buffer[16];

//extern const int32_t unpack_src_format[24];
//extern const int32_t unpack_dst_format[24];
//extern const int32_t pack_src_format[16];
//extern const int32_t pack_dst_format[16];

extern uint32_t pack_sync_tile_dst_ptr;
extern uint32_t math_sync_tile_dst_index;

extern CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];
extern CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];

extern uint32_t __ldm_bss_start[];
extern uint32_t __ldm_bss_end[];
extern uint32_t __ldm_data_start[];
extern uint32_t __ldm_data_end[];
extern void (* __init_array_start[])();
extern void (* __init_array_end[])();
extern uint32_t __firmware_start[];

extern void kernel_init();
extern void kernel_launch();

inline void l1_to_local_mem_copy(uint32_t *local_mem_addr, uint32_t *l1_addr, int32_t len) {
    // Cover L1 load latency of 6 cycles for the bulk of the copy
    int32_t n = 0;
    while (n < len - 5) {
        uint32_t v0 = l1_addr[n + 0];
        uint32_t v1 = l1_addr[n + 1];
        uint32_t v2 = l1_addr[n + 2];
        uint32_t v3 = l1_addr[n + 3];
        uint32_t v4 = l1_addr[n + 4];
        uint32_t v5 = l1_addr[n + 5];
        local_mem_addr[n + 0] = v0;
        local_mem_addr[n + 1] = v1;
        local_mem_addr[n + 2] = v2;
        local_mem_addr[n + 3] = v3;
        local_mem_addr[n + 4] = v4;
        local_mem_addr[n + 5] = v5;
        n += 6;
    }
    // Could optimize this further (eg, loop of 2 or 4), probably not worth it
    while (n < len) {
        local_mem_addr[n] = l1_addr[n];
        n++;
    }
}

inline void firmware_kernel_common_init(void *init_local_l1_base) {

    // Handle stuff typically done in crt0 in asm.  Easier to do in C
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    uint32_t offset = (uint32_t)__ldm_data_start - MEM_LOCAL_BASE;
    l1_to_local_mem_copy((uint32_t *)__ldm_data_start, (uint32_t *)((uint8_t *)init_local_l1_base + offset), num_words);

    for (void (** fptr)() = __init_array_start; fptr < __init_array_end; fptr++) {
        (**fptr)();
    }
}
