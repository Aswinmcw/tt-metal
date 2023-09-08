// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "common/constants.hpp"
#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        string arch_name = "";
        try {
            std::tie(arch_name, input_args) =
                test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
        }
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, device_id);

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // Mt, Nt, Kt = num tiles, B = batch
        uint32_t Mt = 3;
        uint32_t Kt = 2;
        uint32_t Nt = 4;
        uint32_t B = 5;
        Shape shapea = {B, 1, Mt*TILE_HEIGHT, Kt*TILE_WIDTH};
        Shape shapeb = {B, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};
        Shape shapeb1 = {1, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = tt::numpy::random::random(shapea).to(Layout::TILE).to(device);
        Tensor b = tt::numpy::zeros(shapeb, DataType::BFLOAT16).to(Layout::TILE).to(device);
        Tensor b1 = tt::numpy::zeros(shapeb1, DataType::BFLOAT16).to(Layout::TILE).to(device);

        Tensor mm = bmm(a, b).cpu();
        Tensor mm1 = matmul(a, b1).cpu();

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.cpu(); // Move tensor a to host to validate

        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
