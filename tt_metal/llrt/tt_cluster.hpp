/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include "common/base.hpp"
#include "common/metal_soc_descriptor.h"
#include "common/test_common.hpp"
#include "common/tt_backend_api_types.hpp"
#include "host_mem_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "third_party/umd/device/device_api.h"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor.h"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

// XXXX
// TODO: including wormhole in grayskull build is dangerous
// Include noc/noc_parameters.h here so that including it from wormhole
// doesn't pull in the wrong file!
#include "dev_mem_map.h"
#include "noc/noc_parameters.h"
#include "tt_metal/third_party/umd/src/firmware/riscv/wormhole/eth_interface.h"
// XXXX TODO(PGK): fix include paths so device can export interfaces
#include "tt_metal/src/firmware/riscv/common/dev_msgs.h"

static constexpr std::uint32_t SW_VERSION = 0x00020000;

using tt_target_dram = std::tuple<int, int, int>;
using tt::DEVICE;
using tt::TargetDevice;

namespace tt {

class Cluster {
   public:
    Cluster &operator=(const Cluster &) = delete;
    Cluster &operator=(Cluster &&other) noexcept = delete;
    Cluster(const Cluster &) = delete;
    Cluster(Cluster &&other) noexcept = delete;

    static const Cluster &instance();

    size_t number_of_devices() const { return this->cluster_desc_->get_number_of_chips(); }
    size_t number_of_pci_devices() const { return this->cluster_desc_->get_chips_with_mmio().size(); }

    ARCH arch() const { return this->arch_; }

    const metal_SocDescriptor &get_soc_desc(chip_id_t chip) const { return this->sdesc_per_chip_.at(chip); }
    uint32_t get_harvested_rows(chip_id_t chip) const;

    //! device driver and misc apis
    void clean_system_resources() const;

    void verify_eth_fw() const;
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) const;

    void assert_risc_reset(const chip_id_t &chip) const;
    void deassert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) const;
    void deassert_risc_reset(const chip_id_t &target_device_id, bool start_stagger = false) const;

    void write_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access = false) const;
    void read_dram_vec(
        vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, uint32_t size, bool small_access = false) const;

    // Accepts physical noc coordinates
    void write_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, bool small_access = false) const;
    void write_dram_vec(
        const uint32_t *mem_ptr, uint32_t len, tt_cxy_pair dram_core, uint64_t addr, bool small_access = false) const;
    void read_dram_vec(
        vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, uint32_t size, bool small_access = false) const;
    void read_dram_vec(
        uint32_t *mem_ptr, tt_cxy_pair dram_core, uint64_t addr, uint32_t size, bool small_access = false) const;

    void write_reg(const std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const;
    void read_reg(std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const;

    void write_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, chip_id_t src_device_id) const;
    void write_sysmem_vec(const uint32_t* vec, uint32_t size, uint64_t addr, chip_id_t src_device_id) const;
    void read_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, uint32_t size, chip_id_t src_device_id) const;

    int get_device_aiclk(const chip_id_t &chip_id) const;

    // will write a value for each core+hart's debug buffer, indicating that by default
    // any prints will be ignored unless specifically enabled for that core+hart
    // (using tt_start_debug_print_server)
    void reset_debug_print_server_buffers() const;

    void dram_barrier(chip_id_t chip_id) const;
    void l1_barrier(chip_id_t chip_id) const;

    uint32_t get_num_host_channels(chip_id_t device_id) const;
    uint32_t get_host_channel_size(chip_id_t device_id, uint32_t channel) const;
    // Returns address in host space
    void *host_dma_address(uint64_t offset, chip_id_t src_device_id, uint16_t channel) const;

    // Ethernet cluster api
    // Returns set of connected chip ids
    std::unordered_set<chip_id_t> get_ethernet_connected_chip_ids(chip_id_t chip_id) const;

    // Returns set of logical active ethernet coordinates on chip
    std::unordered_set<CoreCoord> get_active_ethernet_cores(chip_id_t chip_id) const;

    // Returns set of logical inactive ethernet coordinates on chip
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores(chip_id_t chip_id) const;

    // Returns connected ethernet core on the other chip
    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(std::tuple<chip_id_t, CoreCoord> eth_core) const;
    std::unique_ptr<tt_device> device_;
   private:
    Cluster();
    ~Cluster();

    void open_device(
        const std::string &sdesc_path = "", const std::string &ndesc_path = "", const bool &skip_driver_allocs = false);
    void start_device(const tt_device_params &device_params);
    void close_device();

    tt_cxy_pair convert_physical_cxy_to_virtual(const tt_cxy_pair &physical_cxy) const;
    void configure_static_tlbs(const std::uint32_t &chip);

    ARCH arch_;
    TargetDevice target_type_;


    // Need to hold reference to cluster descriptor to detect total number of devices available in cluster
    // UMD static APIs `detect_available_device_ids` and `detect_number_of_chips` only returns number of MMIO mapped
    // devices
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc_;
    std::unordered_map<chip_id_t, metal_SocDescriptor> sdesc_per_chip_;

    std::set<chip_id_t> target_device_ids_;

    tt_device_dram_address_params dram_address_params = {
        DRAM_BARRIER_BASE
    };

    tt_device_l1_address_params l1_address_params = {
        (uint32_t)MEM_NCRISC_INIT_IRAM_L1_BASE,
        (uint32_t)MEM_BRISC_FIRMWARE_BASE,
        (uint32_t)MEM_TRISC0_SIZE,
        (uint32_t)MEM_TRISC1_SIZE,
        (uint32_t)MEM_TRISC2_SIZE,
        (uint32_t)MEM_TRISC0_BASE,
        (uint32_t)GET_MAILBOX_ADDRESS_HOST(l1_barrier),
        (uint32_t)eth_l1_mem::address_map::ERISC_BARRIER_BASE
    };

    tt_driver_host_address_params host_address_params = {
        host_mem::address_map::ETH_ROUTING_BLOCK_SIZE, host_mem::address_map::ETH_ROUTING_BUFFERS_START};

    tt_driver_eth_interface_params eth_interface_params = {
        NOC_ADDR_LOCAL_BITS,
        NOC_ADDR_NODE_ID_BITS,
        ETH_RACK_COORD_WIDTH,
        CMD_BUF_SIZE_MASK,
        MAX_BLOCK_SIZE,
        REQUEST_CMD_QUEUE_BASE,
        RESPONSE_CMD_QUEUE_BASE,
        CMD_COUNTERS_SIZE_BYTES,
        REMOTE_UPDATE_PTR_SIZE_BYTES,
        CMD_DATA_BLOCK,
        CMD_WR_REQ,
        CMD_WR_ACK,
        CMD_RD_REQ,
        CMD_RD_DATA,
        CMD_BUF_SIZE,
        CMD_DATA_BLOCK_DRAM,
        ETH_ROUTING_DATA_BUFFER_ADDR,
        REQUEST_ROUTING_CMD_QUEUE_BASE,
        RESPONSE_ROUTING_CMD_QUEUE_BASE,
        CMD_BUF_PTR_MASK};
};

}  // namespace tt

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram);
