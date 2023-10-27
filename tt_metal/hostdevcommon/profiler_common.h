/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

inline __attribute__((always_inline)) uint32_t get_flat_id(uint32_t coreX, uint32_t coreY)
{
    constexpr uint32_t DRAM_ROW = 6;
    constexpr uint32_t MULTIPLIER = 12;
    uint32_t coreFlatID = 0;

    if (coreY > DRAM_ROW){
        coreFlatID = ((coreY - 2) * MULTIPLIER + (coreX - 1));
    }
    else{
        coreFlatID = ((coreY - 1) * MULTIPLIER + (coreX - 1));
    }
    return coreFlatID;
}

namespace kernel_profiler{
/**
 * L1 buffer structure for profiler markers
 * _____________________________________________________________________________________________________
 *|                  |                        |              |             |             |              |

 *|__________________|________________________|______________|_____________|_____________|______________|
 *
 * */

enum BufferIndex {FW_START, FW_START_L, KERNEL_START, KERNEL_START_L, KERNEL_END, KERNEL_END_L, FW_END, FW_END_L, CUSTOM_MARKERS};

enum ControlBuffer
{
    HOST_BUFFER_END_INDEX_BR,
    HOST_BUFFER_END_INDEX_NC,
    HOST_BUFFER_END_INDEX_T0,
    HOST_BUFFER_END_INDEX_T1,
    HOST_BUFFER_END_INDEX_T2,
    DEVICE_BUFFER_END_INDEX_BR,
    DEVICE_BUFFER_END_INDEX_NC,
    DEVICE_BUFFER_END_INDEX_T0,
    DEVICE_BUFFER_END_INDEX_T1,
    DEVICE_BUFFER_END_INDEX_T2,
    FW_RESET_H,
    FW_RESET_L,
    CONTROL_BUFFER_SIZE
};



}
