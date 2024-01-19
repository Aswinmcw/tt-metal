// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher waypoints.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

// Some machines will run this test on different physical cores, so wildcard the exact coordinates
// and replace them at runtime.
std::vector<string> ordered_waypoints = {
    "Device *, Core (x=*,y=*):    GW,W,W,W,W  rmsg:H0D|bnt smsg:DDDD k_ids:0|0|0",
    "Device *, Core (x=*,y=*):    AAAA,W,W,W,W  rmsg:D0G|Bnt smsg:DDDD k_ids:3|0|0",
    "Device *, Core (x=*,y=*):    BBBB,W,W,W,W  rmsg:D0G|Bnt smsg:DDDD k_ids:3|0|0",
    "Device *, Core (x=*,y=*):    CCCC,W,W,W,W  rmsg:D0G|Bnt smsg:DDDD k_ids:3|0|0",
    "Device *, Core (x=*,y=*):    GW,W,W,W,W  rmsg:D0D|Bnt smsg:DDDD k_ids:3|0|0"
};

static void RunTest(WatcherFixture* fixture, Device* device) {
    // Set up program
    Program program = Program();

    // Run a kernel that posts waypoints and waits on certain gating values to be written before
    // posting the next waypoint.
    constexpr CoreCoord core = {0, 0}; // Run kernel on first core
    KernelHandle kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    // Run the program in a new thread, we'll have to update gate values in this thread.
    fixture->RunProgram(device, program);

    // Check the print log against golden output, but need to put the correct phys coords in.
    CoreCoord phys_core = device->worker_core_from_logical_core(core);
    for (int idx = 0; idx < ordered_waypoints.size(); idx++) {
        ordered_waypoints[idx][7] = '0' + device->id();
        ordered_waypoints[idx][18] = '0' + phys_core.x;
        ordered_waypoints[idx][22] = '0' + phys_core.y;
    }
    EXPECT_TRUE(
        FileContainsAllStringsInOrder(
            fixture->log_file_name,
            ordered_waypoints
        )
    );
}

TEST_F(WatcherFixture, TestWatcherWaypoints) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
