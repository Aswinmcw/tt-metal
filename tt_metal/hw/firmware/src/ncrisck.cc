// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
#include "tensix_functions.h"

#include "kernel.cpp"

uint8_t noc_index = NOC_INDEX;

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];

void kernel_launch() {

    firmware_kernel_common_init((void tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE);

    noc_local_state_init(noc_index);

    kernel_profiler::mark_kernel_start();
    kernel_main();
    kernel_profiler::mark_kernel_end();
}
