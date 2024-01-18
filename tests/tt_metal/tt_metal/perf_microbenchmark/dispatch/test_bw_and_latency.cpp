// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 2;
constexpr uint32_t DEFAULT_PAGE_SIZE = 32;
constexpr uint32_t DEFAULT_SIZE_K = 512;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Test read bw and latency from host/dram/l1
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
CoreRange worker_g = {{0, 0}, {0, 0}};;
uint32_t page_size_g;
uint32_t page_count_g;
uint32_t source_mem_g;
bool latency_g;
bool lazy_g;
bool time_just_finish_g;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -s: size in K of data to xfer in one iteration (default {}K)", DEFAULT_SIZE_K);
        log_info(LogTest, "  -p: page size (default {})", DEFAULT_PAGE_SIZE);
        log_info(LogTest, "  -m: source mem, 0:PCIe, 1:DRAM, 2:L1, (default 0)");
        log_info(LogTest, "  -l: measure latency (default is bandwidth)");
        log_info(LogTest, "  -x: X of core to issue read (default {})", 1);
        log_info(LogTest, "  -y: Y of core to issue read (default {})", 1);
        log_info(LogTest, "  -f: time just the finish call (use w/ lazy mode) (default disabled)");
        log_info(LogTest, "  -z: enable dispatch lazy mode (default disabled)");
        exit(0);
    }

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-x", 0);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-y", 0);
    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    lazy_g = test_args::has_command_option(input_args, "-z");
    time_just_finish_g = test_args::has_command_option(input_args, "-f");
    source_mem_g = test_args::get_command_option_uint32(input_args, "-m", 0);
    uint32_t size = test_args::get_command_option_uint32(input_args, "-s", DEFAULT_SIZE_K) * 1024;
    latency_g = test_args::has_command_option(input_args, "-l");
    page_size_g = test_args::get_command_option_uint32(input_args, "-p", DEFAULT_PAGE_SIZE);
    page_count_g = size / page_size_g;

    worker_g = {.start = {core_x, core_y}, .end = {core_x, core_y}};
}

int main(int argc, char **argv) {
    init(argc, argv);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = tt::tt_metal::detail::GetCommandQueue(device);

        tt_metal::Program program = tt_metal::CreateProgram();

        string src_mem;
        switch (source_mem_g) {
        case 0:
            src_mem = "FROM_PCIE";
            break;
        case 1:
            src_mem = "FROM_DRAM";
            break;
        case 2:
            src_mem = "FROM_L1";
            break;
        }

        std::map<string, string> defines = {
            {src_mem, "1"},
            {"ITERATIONS", std::to_string(iterations_g)},
            {"PAGE_SIZE", std::to_string(page_size_g)},
            {"PAGE_COUNT", std::to_string(page_count_g)},
            {"LATENCY", std::to_string(latency_g)}
        };

        tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(page_size_g * page_count_g, {{0, tt::DataFormat::Float32}})
            .set_page_size(0, page_size_g);
        auto cb = tt_metal::CreateCircularBuffer(program, worker_g, cb_config);

        auto dm0 = tt_metal::CreateKernel(
                                          program,
                                          "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/bw_and_latency.cpp",
                                          worker_g,
                                          tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = defines});

        // Cache stuff
        for (int i = 0; i < warmup_iterations_g; i++) {
            EnqueueProgram(cq, program, false);
        }
        Finish(cq);

        if (lazy_g) {
            tt_metal::detail::SetLazyCommandQueueMode(true);
        }

        auto start = std::chrono::system_clock::now();
        EnqueueProgram(cq, program, false);
        if (time_just_finish_g) {
            start = std::chrono::system_clock::now();
        }
        Finish(cq);
        auto end = std::chrono::system_clock::now();

        log_info(LogTest, "Reader core: {}", worker_g.start.str());
        log_info(LogTest, "Reading: {}", src_mem);
        log_info(LogTest, "Lazy: {}", lazy_g);
        log_info(LogTest, "Size: {}", page_count_g * page_size_g);
        log_info(LogTest, "Page size: {}", page_size_g);

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
        if (latency_g) {
            log_info(LogTest, "Latency: {} us", elapsed_seconds.count() / (page_count_g * iterations_g) * 1000.0 * 1000.0);
        } else {
            float bw = page_count_g * page_size_g * iterations_g / (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << bw;
            log_info(LogTest, "BW: {} GB/s", ss.str());
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
