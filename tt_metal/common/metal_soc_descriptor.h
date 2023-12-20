// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "core_coord.h"
#include "third_party/umd/device/tt_soc_descriptor.h"

//! tt_SocDescriptor contains information regarding the SOC configuration targetted.
/*!
    Should only contain relevant configuration for SOC
*/
struct metal_SocDescriptor : public tt_SocDescriptor {
   public:
    std::vector<CoreCoord> preferred_worker_dram_core;  // per channel preferred worker endpoint
    std::vector<CoreCoord> preferred_eth_dram_core;     // per channel preferred eth endpoint
    std::vector<size_t> dram_address_offsets;           // starting address offset
    CoreCoord compute_with_storage_grid_size;
    std::vector<RelativeCoreCoord> compute_with_storage_cores;  // saved as CoreType::WORKER
    std::vector<RelativeCoreCoord> storage_cores;               // saved as CoreType::WORKER
    std::vector<RelativeCoreCoord> producer_cores;
    std::vector<RelativeCoreCoord> consumer_cores;
    std::vector<CoreCoord> logical_ethernet_cores;
    int l1_bank_size;
    uint32_t dram_core_size;

    // in tt_SocDescriptor worker_log_to_routing_x and worker_log_to_routing_y map logical coordinates to NOC virtual
    // coordinates UMD accepts NOC virtual coordinates but Metal needs NOC physical coordinates to ensure a harvested
    // core is not targetted
    std::unordered_map<tt_xy_pair, CoreDescriptor> physical_cores;
    std::vector<tt_xy_pair> physical_workers;
    std::vector<tt_xy_pair> physical_harvested_workers;
    std::vector<tt_xy_pair> physical_ethernet_cores;

    std::unordered_map<int, int> worker_log_to_physical_routing_x;
    std::unordered_map<int, int> worker_log_to_physical_routing_y;
    // Physical to virtual maps are only applicable for x and y of tensix workers
    std::unordered_map<int, int> physical_routing_to_virtual_routing_x;
    std::unordered_map<int, int> physical_routing_to_virtual_routing_y;

    std::map<CoreCoord, int> logical_eth_core_to_chan_map;
    std::map<int, CoreCoord> chan_to_logical_eth_core_map;

    metal_SocDescriptor(const tt_SocDescriptor& other, uint32_t harvesting_mask);
    metal_SocDescriptor() = default;

    CoreCoord get_preferred_worker_core_for_dram_channel(int dram_chan) const;
    CoreCoord get_preferred_eth_core_for_dram_channel(int dram_chan) const;
    size_t get_address_offset(int dram_chan) const;

    bool is_harvested_core(const CoreCoord& core) const;
    const std::vector<CoreCoord>& get_pcie_cores() const;
    const std::vector<CoreCoord> get_dram_cores() const;
    const std::vector<CoreCoord>& get_logical_ethernet_cores() const;
    const std::vector<CoreCoord>& get_physical_ethernet_cores() const;

    tt_cxy_pair convert_to_umd_coordinates(const tt_cxy_pair& physical_cxy) const;

   private:
    void generate_physical_descriptors_from_virtual(uint32_t harvesting_mask);
    void load_dram_metadata_from_device_descriptor();
    void load_dispatch_and_banking_config(uint32_t harvesting_mask);
    void generate_logical_eth_coords_mapping();
};
