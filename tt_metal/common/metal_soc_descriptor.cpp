// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_soc_descriptor.h"

#include <fstream>
#include <iostream>
#include <string>

#include "common/assert.hpp"
#include "dev_mem_map.h"
#include "tt_metal/third_party/umd/device/tt_device.h"
#include "yaml-cpp/yaml.h"

CoreCoord metal_SocDescriptor::get_preferred_worker_core_for_dram_channel(int dram_chan) const {
    TT_ASSERT(
        dram_chan < this->preferred_worker_dram_core.size(),
        "dram_chan={} must be within range of preferred_worker_dram_core.size={}",
        dram_chan,
        this->preferred_worker_dram_core.size());
    return this->preferred_worker_dram_core.at(dram_chan);
};

CoreCoord metal_SocDescriptor::get_preferred_eth_core_for_dram_channel(int dram_chan) const {
    TT_ASSERT(
        dram_chan < this->preferred_eth_dram_core.size(),
        "dram_chan={} must be within range of preferred_eth_dram_core.size={}",
        dram_chan,
        this->preferred_eth_dram_core.size());
    return this->preferred_eth_dram_core.at(dram_chan);
};

size_t metal_SocDescriptor::get_address_offset(int dram_chan) const {
    TT_ASSERT(
        dram_chan < this->dram_address_offsets.size(),
        "dram_chan={} must be within range of dram_address_offsets.size={}",
        dram_chan,
        this->dram_address_offsets.size());
    return this->dram_address_offsets.at(dram_chan);
}

bool metal_SocDescriptor::is_harvested_core(const CoreCoord& core) const {
    for (const auto& core_it : this->physical_harvested_workers) {
        if (core_it == core) {
            return true;
        }
    }
    return false;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_pcie_cores() const { return this->pcie_cores; }

const std::vector<CoreCoord> metal_SocDescriptor::get_dram_cores() const {
    std::vector<CoreCoord> cores;

    // This is inefficient, but is currently not used in a perf path
    for (const auto& channel_it : this->dram_cores) {
        for (const auto& core_it : channel_it) {
            cores.push_back(core_it);
        }
    }

    return cores;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_physical_ethernet_cores() const {
    return this->physical_ethernet_cores;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_logical_ethernet_cores() const {
    return this->logical_ethernet_cores;
}

void metal_SocDescriptor::load_dram_metadata_from_device_descriptor() {
    YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
    int num_dram_channels = this->get_num_dram_channels();
    this->preferred_eth_dram_core.clear();
    for (const auto& core_node : device_descriptor_yaml["dram_preferred_eth_endpoint"]) {
        if (core_node.IsScalar()) {
            this->preferred_eth_dram_core.push_back(format_node(core_node.as<std::string>()));
        } else {
            TT_THROW("Only NOC coords supported for dram_preferred_eth_endpoint cores");
        }
    }
    if (this->preferred_eth_dram_core.size() != num_dram_channels) {
        TT_THROW(
            "Expected to specify preferred DRAM endpoint for ethernet core for {} channels but yaml specifies {} "
            "channels",
            num_dram_channels,
            this->preferred_eth_dram_core.size());
    }

    this->preferred_worker_dram_core.clear();
    for (const auto& core_node : device_descriptor_yaml["dram_preferred_worker_endpoint"]) {
        if (core_node.IsScalar()) {
            this->preferred_worker_dram_core.push_back(format_node(core_node.as<std::string>()));
        } else {
            TT_THROW("Only NOC coords supported for dram_preferred_worker_endpoint");
        }
    }
    if (this->preferred_worker_dram_core.size() != num_dram_channels) {
        TT_THROW(
            "Expected to specify preferred DRAM endpoint for worker core for {} channels but yaml specifies {} "
            "channels",
            num_dram_channels,
            this->preferred_worker_dram_core.size());
    }

    this->dram_address_offsets = device_descriptor_yaml["dram_address_offsets"].as<std::vector<size_t>>();
    if (this->dram_address_offsets.size() != num_dram_channels) {
        TT_THROW(
            "Expected DRAM offsets for {} channels but yaml specified {} channels",
            num_dram_channels,
            this->dram_address_offsets.size());
    }

    int dram_banks_per_prev_core = 0;
    int dram_banks_per_core = 0;
    for (int dram_channel = 0; dram_channel < this->dram_address_offsets.size(); dram_channel++) {
        if (this->dram_address_offsets.at(dram_channel) == 0) {
            if (dram_banks_per_prev_core > 0 and dram_banks_per_prev_core != dram_banks_per_core) {
                TT_THROW("Expected {} DRAM banks per DRAM core", dram_banks_per_prev_core);
            } else if (dram_banks_per_core != 0) {
                dram_banks_per_prev_core = dram_banks_per_core;
            }
            dram_banks_per_core = 1;
        } else {
            dram_banks_per_core++;
        }
    }

    this->dram_core_size = dram_banks_per_core * this->dram_bank_size;
}

// UMD expects virtual NOC coordinates for worker cores
tt_cxy_pair metal_SocDescriptor::convert_to_umd_coordinates(const tt_cxy_pair& physical_cxy) const {
    CoreCoord physical_coord({.x = physical_cxy.x, .y = physical_cxy.y});
    const CoreDescriptor& core_desc = this->physical_cores.at(physical_coord);
    CoreCoord virtual_coord = physical_coord;
    if (core_desc.type == CoreType::WORKER or core_desc.type == CoreType::HARVESTED) {
        virtual_coord.x = static_cast<size_t>(this->physical_routing_to_virtual_routing_x.at(physical_cxy.x));
        virtual_coord.y = static_cast<size_t>(this->physical_routing_to_virtual_routing_y.at(physical_cxy.y));
    }
    return tt_cxy_pair(physical_cxy.chip, virtual_coord);
}

void metal_SocDescriptor::generate_physical_descriptors_from_virtual(uint32_t harvesting_mask) {
    // No need to remap virtual descriptors to physical because Grayskull does not have translation tables enabled,
    // meaning UMD removes the physical harvested rows rather than using virtual coordinates
    if (harvesting_mask == 0 or (this->arch == tt::ARCH::GRAYSKULL)) {
        this->worker_log_to_physical_routing_x = this->worker_log_to_routing_x;
        this->worker_log_to_physical_routing_y = this->worker_log_to_routing_y;
        this->physical_cores = this->cores;
        this->physical_workers = this->workers;
        this->physical_harvested_workers = this->harvested_workers;

        for (const auto& [virtual_noc_core, core_desc] : this->cores) {
            if (core_desc.type == CoreType::WORKER or core_desc.type == CoreType::HARVESTED) {
                this->physical_routing_to_virtual_routing_x.insert({virtual_noc_core.x, virtual_noc_core.x});
                this->physical_routing_to_virtual_routing_y.insert({virtual_noc_core.y, virtual_noc_core.y});
            }
        }

        return;
    }

    std::set<int> row_coordinates_to_remove;
    int row_coordinate = 0;
    int tmp = harvesting_mask;
    while (tmp) {
        if (tmp & 1) {
            row_coordinates_to_remove.insert(row_coordinate);
        }
        tmp = tmp >> 1;
        row_coordinate++;
    }

    std::set<int> virtual_harvested_rows;
    for (const CoreCoord& virtual_harvested_core : this->harvested_workers) {
        virtual_harvested_rows.insert(virtual_harvested_core.y);
    }

    if (row_coordinates_to_remove.size() != virtual_harvested_rows.size()) {
        TT_THROW(
            "Expected number of harvested rows removed by UMD ({}) to match number of harvested rows set in harvesting "
            "mask ({})",
            virtual_harvested_rows.size(),
            row_coordinates_to_remove.size());
    }

    // Columns are not harvested so virtual x == physical x
    std::set<int> virtual_y_coords;
    for (const auto& [virtual_noc_core, core_desc] : this->cores) {
        if (core_desc.type == CoreType::WORKER or core_desc.type == CoreType::HARVESTED) {
            virtual_y_coords.insert(virtual_noc_core.y);
            this->physical_routing_to_virtual_routing_x.insert({virtual_noc_core.x, virtual_noc_core.x});
        }
    }
    this->worker_log_to_physical_routing_x = this->worker_log_to_routing_x;

    std::unordered_map<int, int> virtual_routing_to_physical_routing_y;
    auto virtual_y_coord_it = virtual_y_coords.begin();
    // worker grid size does not include harvested rows
    for (int logical_y_coord = 0; logical_y_coord < this->worker_grid_size.y; logical_y_coord++) {
        while (row_coordinates_to_remove.find(*virtual_y_coord_it) != row_coordinates_to_remove.end()) {
            virtual_y_coord_it++;
        }
        int physical_y_coord = *virtual_y_coord_it;
        virtual_y_coord_it++;
        this->worker_log_to_physical_routing_y.insert({logical_y_coord, physical_y_coord});
        int virtual_y_coord = this->worker_log_to_routing_y.at(logical_y_coord);
        this->physical_routing_to_virtual_routing_y.insert({physical_y_coord, virtual_y_coord});
        virtual_routing_to_physical_routing_y.insert({virtual_y_coord, physical_y_coord});
    }

    // map physical harvested rows to virtual harvested rows
    std::unordered_map<int, int> virtual_harvested_row_to_physical_harvested_row;
    for (auto v_it = virtual_harvested_rows.begin(), p_it = row_coordinates_to_remove.begin();
         v_it != virtual_harvested_rows.end() and p_it != row_coordinates_to_remove.end();
         ++v_it, ++p_it) {
        virtual_routing_to_physical_routing_y.insert({*v_it, *p_it});
        this->physical_routing_to_virtual_routing_y.insert({*p_it, *v_it});
    }

    for (const auto& [virtual_noc_core, core_desc] : this->cores) {
        CoreCoord physical_noc_core = virtual_noc_core;
        CoreDescriptor phys_core_desc = core_desc;
        if (core_desc.type == CoreType::WORKER or core_desc.type == CoreType::HARVESTED) {
            physical_noc_core.y = virtual_routing_to_physical_routing_y.at(virtual_noc_core.y);
            phys_core_desc.coord = physical_noc_core;
            if (row_coordinates_to_remove.find(physical_noc_core.y) != row_coordinates_to_remove.end()) {
                phys_core_desc.type = CoreType::HARVESTED;
                this->physical_harvested_workers.push_back(physical_noc_core);
            } else {
                phys_core_desc.type = CoreType::WORKER;
                this->physical_workers.push_back(physical_noc_core);
            }
        }
        this->physical_cores.insert({physical_noc_core, phys_core_desc});
    }

    TT_ASSERT(
        this->physical_routing_to_virtual_routing_y.size() ==
            this->worker_grid_size.y + row_coordinates_to_remove.size() and
        this->physical_routing_to_virtual_routing_x.size() == this->worker_grid_size.x);
}

const std::string get_product_name(tt::ARCH arch, uint32_t num_harvested_noc_rows) {
    const static std::map<tt::ARCH, std::map<uint32_t, std::string>> product_name = {
        {tt::ARCH::GRAYSKULL, {{0, "E150"}, {2, "E75"}}},
        {tt::ARCH::WORMHOLE_B0, {{0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"}}}};

    return product_name.at(arch).at(num_harvested_noc_rows);
}

void metal_SocDescriptor::load_dispatch_and_banking_config(uint32_t harvesting_mask) {
    std::bitset<32> mask_bitset(harvesting_mask);
    uint32_t num_harvested_noc_rows = mask_bitset.count();

    if (num_harvested_noc_rows > 2) {
        TT_THROW("At most two rows can be harvested, but detected {} harvested rows", num_harvested_noc_rows);
    }
    if (num_harvested_noc_rows == 1 and this->arch == tt::ARCH::GRAYSKULL) {
        TT_THROW("One row harvested Grayskull is not supported");
    }

    YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
    auto product_to_config = device_descriptor_yaml["dispatch_and_banking"];
    auto product_name = get_product_name(this->arch, num_harvested_noc_rows);
    auto config = product_to_config[product_name];

    this->l1_bank_size = config["l1_bank_size"].as<int>();

    // TODO: Add validation for compute_with_storage, storage only, and dispatch core specification
    auto compute_with_storage_start = config["compute_with_storage_grid_range"]["start"];
    auto compute_with_storage_end = config["compute_with_storage_grid_range"]["end"];
    TT_ASSERT(compute_with_storage_start.IsSequence() and compute_with_storage_end.IsSequence());
    TT_ASSERT(compute_with_storage_end[0].as<size_t>() >= compute_with_storage_start[0].as<size_t>());
    TT_ASSERT(compute_with_storage_end[1].as<size_t>() >= compute_with_storage_start[1].as<size_t>());

    this->compute_with_storage_grid_size = CoreCoord({
        .x = (compute_with_storage_end[0].as<size_t>() - compute_with_storage_start[0].as<size_t>()) + 1,
        .y = (compute_with_storage_end[1].as<size_t>() - compute_with_storage_start[1].as<size_t>()) + 1,
    });

    // compute_with_storage_cores are a subset of worker cores
    // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
    for (auto x = 0; x < this->compute_with_storage_grid_size.x; x++) {
        for (auto y = 0; y < this->compute_with_storage_grid_size.y; y++) {
            const auto relative_coord = RelativeCoreCoord({.x = x, .y = y});
            this->compute_with_storage_cores.push_back(relative_coord);
        }
    }

    // storage_cores are a subset of worker cores
    // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
    for (const auto& core_node : config["storage_cores"]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
        } else {
            TT_THROW("Only logical relative coords supported for storage_cores cores");
        }
        this->storage_cores.push_back(coord);
    }

    // dispatch_cores are a subset of worker cores
    // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
    for (const auto& core_node : config["producer_cores"]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
        } else {
            TT_THROW("Only logical relative coords supported for producer_cores cores");
        }
        this->producer_cores.push_back(coord);
    }

    for (const auto& core_node : config["consumer_cores"]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
        } else {
            TT_THROW("Only logical relative coords supported for consumer_cores cores");
        }
        this->consumer_cores.push_back(coord);
    }
}

void metal_SocDescriptor::generate_logical_eth_coords_mapping() {
    this->physical_ethernet_cores = this->ethernet_cores;
    for (int i = 0; i < this->physical_ethernet_cores.size(); i++) {
        CoreCoord core = {.x = 0, .y = static_cast<size_t>(i)};
        this->logical_eth_core_to_chan_map.insert({core, i});
        this->chan_to_logical_eth_core_map.insert({i, core});
        this->logical_ethernet_cores.emplace_back(core);
    }
}

// UMD initializes and owns tt_SocDescriptor
// For architectures with translation tables enabled, UMD will remove the last x rows from the descriptors in
// tt_SocDescriptor (workers list and worker_log_to_routing_x/y maps) This creates a virtual coordinate system, where
// translation tables are used to convert virtual core coordinates to the true harvesting state. For architectures
// without translation tables enabled (Grayskull), UMD updates tt_SocDescriptor to contain the true harvesting state by
// removing the harvested physical coordiniates Metal needs the true harvesting state so we generate physical
// descriptors from virtual coordinates We also initialize additional lookup tables to translate physical coordinates to
// virtual coordinates because UMD APIs expect virtual coordinates.
metal_SocDescriptor::metal_SocDescriptor(const tt_SocDescriptor& other, uint32_t harvesting_mask) :
    tt_SocDescriptor(other) {
    this->trisc_sizes = {MEM_TRISC0_SIZE, MEM_TRISC1_SIZE, MEM_TRISC2_SIZE};  // TODO: Read trisc size from yaml
    this->generate_physical_descriptors_from_virtual(harvesting_mask);
    this->load_dram_metadata_from_device_descriptor();
    this->load_dispatch_and_banking_config(harvesting_mask);
    this->generate_logical_eth_coords_mapping();
}
