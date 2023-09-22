// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "fw_debug.h"
#include "ckernel_globals.h"
#include "risc_common.h"
#include <tensix.h>
#include "run_sync.h"

#include "tools/profiler/kernel_profiler.hpp"

#include "debug_status.h"
#include "debug_print.h"

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

namespace ckernel
{

enum class ttRiscCores : std::uint32_t { Unpack = 0, Math = 1, Pack = 2, Brisc = 3, Nrisc = 4};

volatile tt_reg_ptr uint * const reg_base = reinterpret_cast<volatile uint *>(0xFFB10000);
volatile tt_reg_ptr uint * const pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
volatile tt_reg_ptr uint * const regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile tt_reg_ptr uint * const instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
volatile tt_l1_ptr uint *mailbox_base[4] = {
    reinterpret_cast<volatile tt_l1_ptr uint *>(TENSIX_MAILBOX0_BASE), reinterpret_cast<volatile tt_l1_ptr uint *>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile tt_l1_ptr uint *>(TENSIX_MAILBOX2_BASE), reinterpret_cast<volatile tt_l1_ptr uint *>(TENSIX_MAILBOX3_BASE)
};
volatile tt_l1_ptr uint *dbg_event_scratch = nullptr;
tt_reg_ptr uint *regmem = reinterpret_cast<tt_reg_ptr uint *>(REGFILE_BASE);

uint32_t cfg_state_id __attribute__((used)) = 0;  // Flip between 0 and 1 to keep state between kernel calls
uint32_t dest_offset_id __attribute__((used)) = 0; // Flip between 0 and 1 to keep dest pointer between kernel calls

uint32_t dbg_event_index = 0;
uint32_t dbg_event_end = 0;
uint32_t op_info_offset __attribute__((used)) = 0;

const uint8_t thread_id = COMPILE_FOR_TRISC;

volatile uint local_mem_barrier __attribute__((used));

#define GET_TRISC_RUN_EVAL(x, t) (x)->trisc##t
#define GET_TRISC_RUN(x, t) GET_TRISC_RUN_EVAL(x, t)
volatile tt_l1_ptr uint8_t * const trisc_run = &GET_TRISC_RUN((tt_l1_ptr run_sync_message_t *)(MEM_SLAVE_RUN_MAILBOX_ADDRESS), COMPILE_FOR_TRISC);
} // namespace ckernel

volatile tt_l1_ptr uint32_t l1_buffer[16] __attribute__ ((section ("l1_data"))) __attribute__ ((aligned (16))) __attribute__((used));

using namespace ckernel;

int main(int argc, char *argv[])
{
    DEBUG_STATUS('I');

    uint tt_l1_ptr *local_l1_start_addr = (uint tt_l1_ptr *)PREPROCESSOR_EXPAND(MEM_TRISC, COMPILE_FOR_TRISC, _INIT_LOCAL_L1_BASE);
    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, local_l1_start_addr, num_words);

    FWEVENT("Launching production env kernels");

    // Initialize GPRs to all 0s
    for (int i = 0; i < 64; i++)
        regfile[i] = 0;

    // Init L1 buffer with 1.0f (used for reduce max)
    union {
        float f;
        uint32_t u;
    } f2u = {.f = 1.0f};

    // Save a little code space.  GCC fails to remove the loop variable so loop with a ptr
#pragma GCC unroll 0
    for (uint i = 0; i < 16; i++) l1_buffer[i] = f2u.u;  // Load const into L1 buffer

    reset_cfg_state_id();

    // Cleanup profiler buffer incase we never get the go message
    kernel_profiler::init_profiler();
    while (1) {


        DEBUG_STATUS('W');
        while (*trisc_run != RUN_SYNC_MESSAGE_GO);

        kernel_profiler::init_profiler();
        kernel_profiler::mark_time(CC_MAIN_START);

        DEBUG_STATUS('R');
        kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
        kernel_init();
        kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
        DEBUG_STATUS('D');

        // Signal completion
        tensix_sync();
        *trisc_run = RUN_SYNC_MESSAGE_DONE;

        kernel_profiler::mark_time(CC_MAIN_END);
    }
}
