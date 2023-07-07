#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Semaphores are statically allocated withing range [SEMAPHORE_BASE, SEMAPHORE_BASE + SEMAPHORE_SIZE]
class Semaphore {
   public:
    Semaphore(
        const CoreRangeSet &core_range_set,
        uint32_t address,
        uint32_t initial_value) : core_range_set_(core_range_set), address_(address), initial_value_(initial_value) {}

    Semaphore(const Semaphore &other);

    Semaphore& operator=(const Semaphore &other);

    Semaphore(Semaphore &&other);

    Semaphore& operator=(Semaphore &&other);

    constexpr uint32_t size() const { return SEMAPHORE_SIZE / NUM_SEMAPHORES; }


    uint32_t address() const { return address_; }

    CoreRangeSet core_range_set() const { return core_range_set_; }

    uint32_t initial_value() const { return initial_value_; }

    bool initialized_on_logical_core(const CoreCoord &logical_core) const;

   private:
    CoreRangeSet core_range_set_;             // Ranges of cores where this semaphore is initialized
    uint32_t address_;
    uint32_t initial_value_;              // Initial value of semaphore
};

}  // namespace tt_metal

}  // namespace tt
