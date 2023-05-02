#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/common/tt_soc_descriptor.h"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

class BasicAllocator : public Allocator {
   public:
    BasicAllocator(const tt_SocDescriptor &soc_desc);

    ~BasicAllocator() {}

    // TODO: Update copy/move semantics
    BasicAllocator(const BasicAllocator &other) { }
    BasicAllocator& operator=(const BasicAllocator &other) { return *this; }

    BasicAllocator(BasicAllocator &&other) { }
    BasicAllocator& operator=(BasicAllocator &&other) { return *this; }

    uint32_t allocate_dram_buffer(int dram_channel, uint32_t size_bytes);

    uint32_t allocate_dram_buffer(int dram_channel, uint32_t start_address, uint32_t size_bytes);

    uint32_t allocate_sysmem_buffer(uint32_t size_bytes);

    uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes);

    uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes);


    void deallocate_dram_buffer(int dram_channel, uint32_t address);

    void deallocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t address);

    void deallocate_sysmem_buffer(uint32_t address);


    uint32_t get_address_for_interleaved_dram_buffer(const std::map<int, uint32_t> &size_in_bytes_per_bank) const;
    uint32_t get_address_for_l1_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const;


    void clear_dram();

    void clear_l1();

    void clear_sysmem();

    void clear();

   private:
    allocator::Algorithm &allocator_for_dram_channel(int dram_channel) const;

    allocator::Algorithm &allocator_for_logical_core(const tt_xy_pair &logical_core) const;

    allocator::Algorithm &allocator_for_sysmem() const;

    std::unordered_map<int, std::unique_ptr<allocator::Algorithm>> dram_manager_;
    std::unordered_map<tt_xy_pair, std::unique_ptr<allocator::Algorithm>> l1_manager_;
    std::unique_ptr<allocator::Algorithm> sysmem_manager_;
};

}  // namespace tt_metal

}  // namespace tt
