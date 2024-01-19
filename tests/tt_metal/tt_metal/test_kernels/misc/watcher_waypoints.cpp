// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ckernel.h"
#include "debug/status.h"

/*
 * A test for the watcher waypointing feature.
*/
#if defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_NCRISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif
    uint32_t wait_cycles = 1200000000; // ~1sec

    // Post a new waypoint with a delay after (to let the watcher poll it)
    DEBUG_STATUS('A', 'A', 'A', 'A');
    ckernel::wait(wait_cycles);
    DEBUG_STATUS('B', 'B', 'B', 'B');
    ckernel::wait(wait_cycles);
    DEBUG_STATUS('C', 'C', 'C', 'C');
    ckernel::wait(wait_cycles);
#if defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_NCRISC)
}
#else
}
}
#endif
