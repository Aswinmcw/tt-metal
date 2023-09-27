// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "multi_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::dram::multichip {
/// @brief Does Dram --> Reader --> L1 on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& reader_core) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t dram_byte_address = input_dram_buffer.address();
    auto dram_noc_xy = input_dram_buffer.noc_coordinates();
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_dram_to_l1.cpp",
        reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        reader_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_core_data;
    // tt_metal::ReadFromBuffer(l1_buffer, dest_core_data);
    tt_metal::detail::ReadFromDeviceL1(device, reader_core, l1_byte_address, byte_size, dest_core_data);
    std::cout << " l1 byte address " << std::hex<< l1_byte_address << std::endl;
    pass &= (dest_core_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << reader_core.str() << std::endl;
    }
    return pass;
}

/// @brief Does L1 --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool writer_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& writer_core) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t dram_byte_address = output_dram_buffer.address();
    auto dram_noc_xy = output_dram_buffer.noc_coordinates();
    auto l1_bank_ids = device->bank_ids_from_logical_core(writer_core);
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_l1_to_dram.cpp",
        writer_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToDeviceL1(device, writer_core, l1_byte_address, inputs);
    // tt_metal::WriteToBuffer(l1_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        writer_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= (dest_buffer_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << writer_core.str() << std::endl;
    }
    return pass;
}

struct ReaderWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t l1_byte_address = 0;
    tt::DataFormat l1_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_writer(tt_metal::Device* device, const ReaderWriterConfig& test_config) {

    bool pass = true;

    const uint32_t cb_index = 0;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t output_dram_byte_address = output_dram_buffer.address();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_cb = tt_metal::CreateCircularBuffer(
        program,
        cb_index,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_data_format,
        test_config.l1_byte_address);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
struct ReaderDatacopyWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t l1_input_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    size_t l1_output_byte_address = 0;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Datacopy --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_datacopy_writer(tt_metal::Device* device, const ReaderDatacopyWriterConfig& test_config) {

    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t output_dram_byte_address = output_dram_buffer.address();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input_cb = tt_metal::CreateCircularBuffer(
        program,
        input0_cb_index,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_input_data_format,
        test_config.l1_input_byte_address);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program,
        output_cb_index,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_output_data_format,
        test_config.l1_output_byte_address);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {input0_cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {output_cb_index}});

    vector<uint32_t> compute_kernel_args = {
        uint(test_config.num_tiles)  // per_core_tile_cnt
    };
    auto datacopy_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        test_config.core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
}  // namespace unit_tests::dram::multichip

TEST_F(MultiDeviceFixture, SingleCoreDirectDramReaderMultichip) {
    ASSERT_TRUE(unit_tests::dram::multichip::reader_only(devices_.at(0), 1 * 1024, L1_UNRESERVED_BASE, {.x = 1, .y = 6}));
    //ASSERT_TRUE(unit_tests::dram::multichip::reader_only(device_, 2 * 1024, L1_UNRESERVED_BASE, {.x = 0, .y = 0}));
    //ASSERT_TRUE(unit_tests::dram::multichip::reader_only(device_, 16 * 1024, L1_UNRESERVED_BASE, {.x = 0, .y = 0}));
}
