// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

// TODO(agrebenisan): Go over the following and figure out which ones you actually need
#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
// #include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
#include "noc_addr_ranges_gen.h"

#include <kernel.cpp>

uint8_t noc_index = NOC_INDEX;

void kernel_launch() {

    firmware_kernel_common_init((void tt_l1_ptr *)MEM_ERISC_INIT_LOCAL_L1_BASE);
    noc_local_state_init(noc_index);
    // kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
    kernel_main();
    // kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
}
