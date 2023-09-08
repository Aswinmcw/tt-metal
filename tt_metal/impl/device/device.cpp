// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"

#include "common/utils.hpp"
#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

size_t Device::detect_num_available_devices(const TargetDevice target_type) {
    return tt_cluster::detect_available_devices(target_type).size();
}
void Device::initialize_cluster() {
    ZoneScoped;
    llrt::utils::log_current_ai_clk(this->cluster(), this->pcie_slot_);
    this->cluster()->assert_risc_reset(pcie_slot_);
}

void Device::initialize_allocator(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    TT_ASSERT(cluster_is_initialized() && "Cluster needs to be initialized!");
    tt::log_assert(
        this->harvesting_initialized_,
        "Harvesting information needs to be initialized before allocator"
    );
    auto soc_desc = this->cluster()->get_soc_desc(this->pcie_slot_);
    // Construct allocator config from soc_desc
    AllocatorConfig config({
        .num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
        .dram_bank_size = soc_desc.dram_bank_size,
        .dram_bank_offsets = {},
        .worker_grid_size = this->post_harvested_worker_grid_size_,
        .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
        .l1_bank_size = static_cast<size_t>(soc_desc.l1_bank_size),
        .core_type_from_noc_coord_table = {}, // Populated later
        .logical_to_routing_coord_lookup_table=this->logical_to_routing_coord_lookup_table_,
        .l1_bank_remap = l1_bank_remap,
    });
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.cores) {
        config.core_type_from_noc_coord_table.insert({core.first, AllocCoreType::Invalid});
    }
    for (const auto& core : soc_desc.compute_with_storage_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, this->post_harvested_worker_grid_size_);
        const auto noc_coord = this->logical_to_routing_coord_lookup_table_[logical_coord];
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const auto& core : soc_desc.storage_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, this->post_harvested_worker_grid_size_);
        const auto noc_coord = this->logical_to_routing_coord_lookup_table_[logical_coord];
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const auto& core : soc_desc.dispatch_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, this->post_harvested_worker_grid_size_);
        const auto noc_coord = this->logical_to_routing_coord_lookup_table_[logical_coord];
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    static_assert(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    this->allocator_ = std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_harvesting_information() {
    ZoneScoped;
    if (not cluster_is_initialized()) {
        tt::log_fatal("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    auto &soc_desc = this->cluster()->get_soc_desc(this->pcie_slot_);
    uint32_t harvested_noc_rows = 0;
    if (this->target_type_ == tt::TargetDevice::Silicon) {
        harvested_noc_rows = this->cluster()->get_harvested_rows(this->pcie_slot_);
    }

    // Determine which noc-coords are harvested
    std::vector<unsigned int> noc_row_harvested(soc_desc.grid_size.y, 0);
    this->num_harvested_rows_ = 0;
    for (unsigned int r = 0; r < soc_desc.grid_size.y; r++) {
        bool row_harvested = (harvested_noc_rows>>r)&0x1;
        this->num_harvested_rows_ += row_harvested;
        noc_row_harvested[r] = row_harvested;
    }

    load_dispatch_and_banking_config(soc_desc, this->num_harvested_rows_);

    tt::log_assert(
        this->num_harvested_rows_ <= 2,
        tt::LogDevice,
        "this->pcie_slot_={} has this->num_harvested_rows_={}>2",
        this->pcie_slot_,
        this->num_harvested_rows_);
    tt::log_assert(
        (this->num_harvested_rows_ == 0) or (this->arch_ == tt::ARCH::WORMHOLE_B0),
        tt::LogDevice,
        "Harvested Rows={} -- Harvesting is only supported on WORMHOLE_B0",
        this->num_harvested_rows_);
    // Populate lookup table
    this->logical_to_routing_coord_lookup_table_.clear();
    unsigned int num_rows = soc_desc.worker_grid_size.y;
    unsigned int num_cols = soc_desc.worker_grid_size.x;
    unsigned int row_offset = 0;
    for (unsigned int r = 0; r < num_rows; r++) {
        for (unsigned int c = 0; c < num_cols; c++) {
            CoreCoord logical_coord({
                .x = c,
                .y = r,
            });
            CoreCoord noc_routing_coord({
                .x = static_cast<size_t>(soc_desc.worker_log_to_routing_x.at(logical_coord.x)),
                .y = static_cast<size_t>(soc_desc.worker_log_to_routing_y.at(logical_coord.y)),
            });
            CoreCoord translated_coord = noc_routing_coord;
            this->logical_to_routing_coord_lookup_table_.insert({
                logical_coord,
                translated_coord
            });
        }
    }
    this->post_harvested_worker_grid_size_ = CoreCoord{
        .x = soc_desc.worker_grid_size.x,
        .y = soc_desc.worker_grid_size.y
    };
    this->harvesting_initialized_ = true;
}

void Device::clear_l1_state() {
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }
}

bool Device::initialize(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}", this->pcie_slot_);
    TT_ASSERT(not this->initialized_, "Device {} has already been initialized!", this->pcie_slot_);
    this->initialize_cluster();
    this->initialize_harvesting_information();
    this->initialize_allocator(l1_bank_remap);
    this->clear_l1_state();
    tt_start_debug_print_server(this->cluster());
    this->initialized_ = true;
    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->pcie_slot_);
    TT_ASSERT(this->initialized_, "Cannot close device {} that has not been initialized!", this->pcie_slot_);
    tt_stop_debug_print_server(this->cluster());
    this->cluster()->assert_risc_reset(pcie_slot_);
    this->clear_l1_state();
    allocator::clear(*this->allocator_);
    this->initialized_ = false;
    return true;
}

Device::~Device() {}

bool Device::cluster_is_initialized() const {
    return detail::ClusterWrapper::inst(this->arch_, this->target_type_).cluster() != nullptr;
}

tt_cluster *Device::cluster() const {
    if (not this->cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return detail::ClusterWrapper::inst(this->arch_, this->target_type_).cluster();
}

int Device::num_dram_channels() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster()->get_soc_desc(pcie_slot_).get_num_dram_channels();
}

uint32_t Device::l1_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster()->get_soc_desc(pcie_slot_).worker_l1_size;
}
uint32_t Device::dram_bank_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster()->get_soc_desc(pcie_slot_).dram_bank_size;
}

CoreCoord Device::logical_grid_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->post_harvested_worker_grid_size_;
}

CoreCoord Device::compute_with_storage_grid_size() const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    return this->cluster()->get_soc_desc(pcie_slot_).compute_with_storage_grid_size;
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    TT_ASSERT(
        (logical_core.x < this->post_harvested_worker_grid_size_.x) and
        (logical_core.y < this->post_harvested_worker_grid_size_.y),
        "Bounds-Error -- Logical_core={} is outside of logical_grid_size={}",
        logical_core.str(),
        this->post_harvested_worker_grid_size_.str()
    );
    CoreCoord worker_core = this->logical_to_routing_coord_lookup_table_.at(logical_core);
    return worker_core;
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) {
    std::vector<CoreCoord> worker_cores;
    for (auto logical_core : logical_cores) {
        worker_cores.push_back(worker_core_from_logical_core(logical_core));
    }
    return worker_cores;
}

void Device::check_allocator_is_initialized() const {
    if (this->allocator_ == nullptr) {
        TT_THROW("No memory allocator! Device has not been initialized, did you forget to call InitializeDevice?");
    }
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::num_banks(*this->allocator_, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::core_from_dram_channel(uint32_t dram_channel) const {
    if (not cluster_is_initialized()) {
        TT_THROW("Device has not been initialized, did you forget to call InitializeDevice?");
    }
    log_assert(
        dram_channel < this->num_dram_channels(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        dram_channel,
        this->num_dram_channels()
    );
    return this->cluster()->get_soc_desc(pcie_slot_).get_preferred_worker_core_for_dram_channel(dram_channel);
}

int32_t Device::l1_bank_offset_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::l1_bank_offset_from_bank_id(*this->allocator_, bank_id);
}

int32_t Device::dram_bank_offset_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_bank_offset_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

std::vector<uint32_t> Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

std::vector<uint32_t> Device::bank_ids_from_logical_core(const CoreCoord &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::get_statistics(*this->allocator_, buffer_type);
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const {
    this->check_allocator_is_initialized();
    return allocator::dump_memory_blocks(*this->allocator_, buffer_type, out);
}

}  // namespace tt_metal

}  // namespace tt
