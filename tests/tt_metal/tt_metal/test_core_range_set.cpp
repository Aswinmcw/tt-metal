#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

void check_program_is_mapped_to_correct_cores(const tt_metal::Program &program, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &compute_kernel_args) {
    // every kernel, semaphore and CB should be mapped to each core in the core ranges of core_range_set
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                auto logical_core = CoreCoord{x, y};
                for (auto kernel : program.kernels()) {
                    TT_ASSERT(kernel->is_on_logical_core(logical_core));
                    // Check that compute kernel compile time args are mapped to the correct cores
                    if (kernel->kernel_type() == tt_metal::KernelType::Compute) {
                        auto kernel_compile_time_args = kernel->compile_time_args();
                        TT_ASSERT(kernel_compile_time_args == compute_kernel_args);
                    }
                }
                for (auto cb : program.circular_buffers()) {
                    TT_ASSERT(cb->is_on_logical_core(logical_core));
                }
                for (auto semaphore : program.semaphores()) {
                    TT_ASSERT(semaphore->initialized_on_logical_core(logical_core));
                }
            }
        }
    }
}

void check_semaphores_are_initialized(tt_metal::Device *device, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &golden_sem_values) {
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                auto logical_core = CoreCoord{x, y};
                std::vector<uint32_t> res;
                tt_metal::ReadFromDeviceL1(device, logical_core, SEMAPHORE_BASE, SEMAPHORE_SIZE, res);
                std::vector<uint32_t> filtered_res;
                constexpr static uint32_t num_u32_to_skip = UINT32_WORDS_PER_SEMAPHORE * sizeof(uint32_t);
                for (int i = 0; i < res.size(); i+=num_u32_to_skip) {
                    filtered_res.push_back(res.at(i));
                }
                TT_ASSERT(filtered_res == golden_sem_values);
            }
        }
    }
}

bool test_program_specified_with_core_range_set(tt_metal::Device *device, tt_metal::Program &program, const CoreRangeSet &core_range_set) {
    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 32;
    uint32_t buffer_size = single_tile_size * num_tiles;

    int dram_src_channel_id = 0;
    auto src_dram_buffer = tt_metal::Buffer(device, buffer_size, dram_src_channel_id, buffer_size, tt_metal::BufferType::DRAM);
    auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();

    std::map<CoreCoord, tt_metal::Buffer> core_to_l1_buffer;
    for (auto core_range : core_range_set.ranges()) {
        auto start = core_range.start;
        auto end = core_range.end;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({.x=x, .y=y});
                auto l1_bank_ids = device->bank_ids_from_logical_core(logical_core);
                TT_ASSERT(not l1_bank_ids.empty());
                auto l1_bank_id = l1_bank_ids.at(0);
                auto dst_l1_buffer = tt_metal::Buffer(device, buffer_size, l1_bank_id, buffer_size, tt_metal::BufferType::L1);
                core_to_l1_buffer.emplace(logical_core, dst_l1_buffer);
            }
        }
    }

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core_range_set,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 1;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core_range_set,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_push_4.cpp",
        core_range_set,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core_range_set,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // Each core range shares the same compute kernel args
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy_3m.cpp",
        core_range_set,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    auto size_per_semaphore = SEMAPHORE_SIZE / NUM_SEMAPHORES;
    std::vector<uint32_t> golden_sem_values;
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        uint32_t initial_value = i;
        auto semaphore = tt_metal::CreateSemaphore(program, device, core_range_set, initial_value);
        golden_sem_values.push_back(initial_value);
        pass &= semaphore->address() == SEMAPHORE_BASE + (size_per_semaphore * i);
    }

    check_program_is_mapped_to_correct_cores(program, core_range_set, compute_kernel_args);

    pass &= tt_metal::CompileProgram(device, program);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

    check_semaphores_are_initialized(device, core_range_set, golden_sem_values);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::WriteToBuffer(src_dram_buffer, src_vec);

    // Reader kernel on all cores reads from same location in DRAM
    std::vector<uint32_t> reader_rt_args = {
        src_dram_buffer.address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        num_tiles
    };
    std::vector<std::vector<uint32_t>> reader_args_for_core_ranges;
    for (auto core_range : core_range_set.ranges()) {
        reader_args_for_core_ranges.push_back(reader_rt_args);
    }
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core_range_set,
        reader_args_for_core_ranges);

    // Write runtime args for each writer separately because they may be writing to different location in their cores L1
    for (const auto &[core, dst_l1_buffer] : core_to_l1_buffer) {
        auto l1_dst_noc_xy = dst_l1_buffer.noc_coordinates();
        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dst_l1_buffer.address(),
            (std::uint32_t)l1_dst_noc_xy.x,
            (std::uint32_t)l1_dst_noc_xy.y,
            num_tiles});
    }

    pass &= tt_metal::LaunchKernels(device, program);

    for (const auto &[core, dst_l1_buffer] : core_to_l1_buffer) {
        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_l1_buffer, result_vec);
        bool copied_data_correctly = src_vec == result_vec;
        TT_ASSERT(copied_data_correctly);
        pass &= copied_data_correctly;
    }

    return pass;
}

bool test_overlapping_core_range_set() {
    bool pass = true;
    CoreRange core_range_one = {.start={0, 0}, .end={2, 2}};
    CoreRange core_range_two = {.start={3, 3}, .end={5, 4}};

    CoreRangeSet valid_ranges = CoreRangeSet({core_range_one, core_range_two});
    TT_ASSERT(valid_ranges.ranges().size() == 2);

    CoreRange overlapping_range = {.start={1, 2}, .end={3, 3}};

    try {
        CoreRangeSet invalid_ranges = CoreRangeSet({core_range_one, core_range_two, overlapping_range});
        pass &= false;
        TT_ASSERT(false);
    } catch (const std::exception &e) {}

    CoreRange single_core = {.start={1, 1}, .end={1, 1}};
    CoreRangeSet second_valid_ranges = CoreRangeSet({single_core, core_range_two});

    try {
        CoreRangeSet second_invalid_ranges = CoreRangeSet({single_core, core_range_two, core_range_one});
        pass &= false;
        TT_ASSERT(false);
    } catch (const std::exception &e) {}

    return pass;
}

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
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        tt_metal::Program program = tt_metal::Program();
        CoreRange core_range_one = {.start={0, 0}, .end={1, 1}};
        CoreRange core_range_two = {.start={2, 2}, .end={3, 3}};
        CoreRangeSet core_ranges = CoreRangeSet({core_range_one, core_range_two});

        pass &= test_program_specified_with_core_range_set(device, program, core_ranges);

        pass &= test_overlapping_core_range_set();

        ////////////////////////////////////////////////////////////////////////////
        //                              Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CloseDevice(device);

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
