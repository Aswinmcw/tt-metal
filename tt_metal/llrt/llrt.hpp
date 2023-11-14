/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <random>
#include <algorithm>
#include <functional>
#include <tuple>
#include <iostream>
#include <filesystem>

#include "llrt/tt_cluster.hpp"
#include "tensix.h"
#include "tt_metal/third_party/umd/device/device_api.h"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"
#include "llrt_common/tiles.hpp"
#include "llrt/tt_memory.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
// XXXX TODO(PGK): fix include paths so device can export interfaces
#include "tt_metal/src/firmware/riscv/common/dev_msgs.h"

namespace tt {

// llrt = lower-level runtime
namespace llrt {

using RamSrcAddr = unsigned int;
using RamDstAddr = unsigned int;
using SrcL1Core = CoreCoord;
using SrcL1Cores = std::vector<SrcL1Core>;
using DstL1Core = CoreCoord;
using DstL1Cores = std::vector<DstL1Core>;
using SrcChannelId = int;
using DstChannelId = int;
using DramBufferSize = unsigned int;
using DramSrcAddr = unsigned int;
using DramDstAddr = unsigned int;
using L1Addr = std::uint32_t;
using SrcAddr = std::uint32_t;
using DestAddr = std::uint32_t;
using LoadFirmwareFlag = bool;
using CountOffset = unsigned int;
using NCHW = std::array<std::uint32_t, 4>;
using RSUV = std::array<std::uint32_t, 4>;
using BYTES_PER_DATUM = std::uint32_t;
using TRANSACTION_SIZE = std::uint32_t;
using NUM_TRANSACTIONS = std::uint32_t;
using NUM_REPETITIONS = std::uint32_t;

using WorkerCore = tt_cxy_pair;
using WorkerCores = std::vector<WorkerCore>;
using CircularBufferConfigVec = std::vector<uint32_t>;

// made these free functions -- they're copy/paste of the member functions
// TODO: clean-up epoch_loader / epoch_binary -- a bunch of functions there should not be member functions
ll_api::memory get_risc_binary(string path, chip_id_t chip_id, bool fw_build);
uint16_t get_binary_code_size16(const ll_api::memory& mem, int riscv_id);

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
inline void write_hex_vec_to_core(chip_id_t chip, const CoreCoord &core, std::vector<uint32_t> hex_vec, uint64_t addr, bool small_access = false) { tt::Cluster::instance().write_dram_vec(hex_vec, tt_cxy_pair(chip, core), addr, small_access); }

inline std::vector<std::uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord &core, uint64_t addr, uint32_t size) {
    vector<std::uint32_t> read_hex_vec;
    tt::Cluster::instance().read_dram_vec(read_hex_vec, tt_cxy_pair(chip, core), addr, size);
    return read_hex_vec;
}

void write_launch_msg_to_core(chip_id_t chip, CoreCoord core, launch_msg_t *msg);

void print_worker_cores(chip_id_t chip_id = 0);

bool is_worker_core(CoreCoord &core, chip_id_t chip_id = 0);

CircularBufferConfigVec create_circular_buffer_config_vector();

void set_config_for_circular_buffer(
    CircularBufferConfigVec &circular_buffer_config_vec,
    uint32_t circular_buffer_index,
    uint32_t addr_in_bytes,
    uint32_t size_in_bytes,
    uint32_t num_pages);

void write_circular_buffer_config_vector_to_core(
    chip_id_t chip, const CoreCoord &core, CircularBufferConfigVec circular_buffer_config_vec);

void write_graph_interpreter_op_info_to_core(
    chip_id_t chip, const CoreCoord &core, op_info_t op_info, int op_idx);


void program_brisc_startup_addr(chip_id_t chip_id, const CoreCoord &core);

bool test_load_write_read_risc_binary(
    ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int riscv_id);

bool test_load_write_read_trisc_binary(
    ll_api::memory &mem, chip_id_t chip_id, const CoreCoord &core, int triscv_id);

// subchannel hard-coded to 0 for now
CoreCoord get_core_for_dram_channel(int dram_channel_id, chip_id_t chip_id = 0);

namespace internal_ {

void wait_until_cores_done(chip_id_t device_id,
                           int run_state,
                           std::unordered_set<CoreCoord>& not_done_phys_cores);

}  // namespace internal_

inline uint64_t relocate_dev_addr(uint64_t addr, uint64_t local_init_addr) {

    uint64_t relo_addr;
    if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
        // Move addresses in the local memory range to l1 (copied by kernel)
        relo_addr = (addr & ~MEM_LOCAL_BASE) + local_init_addr;
    } else if ((addr & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
        // Move addresses in the trisc memory range to l1 (copied by kernel)
        relo_addr = (addr & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
    } else {
        relo_addr = addr;
    }
    return relo_addr;
}

}  // namespace llrt

}  // namespace tt
