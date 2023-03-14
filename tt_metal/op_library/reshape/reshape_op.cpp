#include "tt_metal/op_library/reshape/reshape_op.hpp"
#include "common/test_tiles.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace eltwise_unary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor reshape(Tensor &a, int N, int C, int H, int W) {

    if (a.layout() == Layout::TILE) {
        // Don't need to do a check here to see the H and W both divisible by 32
        // since handled within the tensor reshape method
        a.reshape(N, C, H, W);
        return a;
    }

    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Only tile and row major reshape supported!");

    // Reshape for row major data requires rebanking if and only if the last
    // dimension is changed. W dictates the stick size in DRAM banks
    if (W == a.shape()[3]) {
        a.reshape(N, C, H, W);
        return a;
    }

    TT_ASSERT(N != -1 and C != -1 and H != -1 and W != -1, "-1 reshape not yet supported for rebanking row major reshape");

    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};

    TT_ASSERT(not a.on_host(), "Operand to reshape needs to be on device");
    TT_ASSERT(a.buffer() != nullptr, "Operand to reshape needs to be allocated in a buffer on device!");
    TT_ASSERT(a.volume() == N*C*H*W, "New shape volume must match old shape volume");

     // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);
    tt_metal::InterleavedDramBuffer *src0_dram_buffer = a.buffer();
    tt_metal::InterleavedDramBuffer *dst_dram_buffer = output.buffer();

    uint32_t single_tile_size = 2 * TILE_HW; // Assuming bfloat16 dataformat
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = (a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW);
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = (C * H * W / TILE_HW);
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t num_old_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t old_stick_size = a.shape()[3] * 2; // Assuming bfloat16 data format

    uint32_t num_new_sticks = N*C*H;
    uint32_t new_stick_size = W * 2; // Assuming bfloat16 data format


    // Reader compile-time args
    bool old_stick_size_is_power_of_two = (ceil(log2(old_stick_size)) == floor(log2(old_stick_size)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(), num_old_sticks, old_stick_size};
    DataMovementKernelArgs *reader_compile_time_args;
    if (old_stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(old_stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1});
    } else {
        reader_compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {0});
    }

    // Writer compile-time args
    bool new_stick_size_is_power_of_two = (ceil(log2(new_stick_size)) == floor(log2(new_stick_size)));
    vector<uint32_t> writer_kernel_args = {dst_dram_buffer->address(), num_new_sticks, new_stick_size};
    DataMovementKernelArgs *writer_compile_time_args;
    if (new_stick_size_is_power_of_two) {
        writer_kernel_args.push_back(log2(new_stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        writer_compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1});
    } else {
        writer_compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {0});
    }

    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
        core,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // No compute required, so using blank kernel
    vector<uint32_t> compute_args = {
        uint(a.volume() / TILE_HW), // per_core_block_cnt
        1 // per_core_block_size
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "kernels/compute/eltwise_copy.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    // Compile kernels
    bool skip_hlkc = false;
    tt_metal::CompileProgram(device, program, skip_hlkc);
    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        writer_kernel_args
    );

    tt_metal::LaunchKernels(device, program);

    output.reshape(N, C, H, W);
    delete program;
    return output;
}

} // namespace tt_metal
} // namespace tt
