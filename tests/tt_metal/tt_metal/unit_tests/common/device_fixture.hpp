/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"

inline
bool is_multi_device_gs_machine(const tt::ARCH& arch, const size_t num_devices) {
    return arch == tt::ARCH::GRAYSKULL && num_devices > 1;
}

class DeviceFixture : public ::testing::Test  {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            tt::log_fatal("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int device_id = 0;

        num_devices_ = tt::tt_metal::Device::detect_num_available_devices();
        // TODO (aliu): this needs to change, detect_num_avilable_devices only returning 1
        for (unsigned int id = 0; id < num_devices_; id++) {
            auto * device = tt::tt_metal::CreateDevice(id);
            //device->initialize_cluster();
            //device->cluster()->clean_system_resources();
            devices_.push_back(device);

        }
        // TODO (aliu): remove later, and change all unit tests
        if (num_devices_ == 1) {
          device_ = devices_.at(0);
        }
        if (is_multi_device_gs_machine(arch_, num_devices_)) {
            GTEST_SKIP();
        }
    }

    void TearDown() override {
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    tt::tt_metal::Device* device_ = nullptr;
    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};
