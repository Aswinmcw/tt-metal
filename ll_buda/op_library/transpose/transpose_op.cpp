#include "ll_buda/op_library/transpose/transpose_op.hpp"

#include "ll_buda/host_api.hpp"
#include "constants.hpp"

#include "tools/cpuprof/cpuprof.h"

namespace hlk_copy_binary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t num_tensor_tiles;
};
}

namespace hlk_transpose_wh {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t NHtWt;
};
}

using u32 = std::uint32_t;
using namespace tt::constants;

namespace tt {

namespace ll_buda {

Tensor transpose_wh(const Tensor &a) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    ll_buda::Program *program = new ll_buda::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    ll_buda::InterleavedDramBuffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernel only needs the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);


    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], W, H};
    ll_buda::Tensor output = ll_buda::Tensor(output_shape, a.dtype(), tt::ll_buda::Layout::TILE, device);

    ll_buda::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = ll_buda::CreateCircularBuffer(
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
    uint32_t num_output_tiles = 2;
    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    ll_buda::DataMovementKernel *reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_transpose_wh_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new hlk_transpose_wh::hlk_args_t{ .NHtWt = int(Ht*Wt*NC) };
    ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(hlk_transpose_wh::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/transpose_wh.cpp",
        core,
        eltwise_binary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            reader_kernel,
            core,
            {
                src0_dram_buffer->address(),
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                num_tensor_tiles, NC, Ht, Wt, Ht*Wt
            }
        );

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            writer_kernel,
            core,
            {
                dst_dram_buffer->address(),
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tensor_tiles
            }
        );

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data

    return output;
}

Tensor transpose_wh_multi_core(const Tensor &a) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    ll_buda::Program *program = new ll_buda::Program();
    auto num_cores_c = Wt;
    auto num_cores_r = Ht;
    tt_xy_pair start_core = {0, 0};
    tt_xy_pair end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};;
    ll_buda::CoreRange all_cores(start_core, end_core);

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    ll_buda::InterleavedDramBuffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernel only needs the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);


    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], W, H};
    ll_buda::Tensor output = ll_buda::Tensor(output_shape, a.dtype(), tt::ll_buda::Layout::TILE, device);

    ll_buda::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);
    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            int core_index = i * num_cores_c + j;
            tt_xy_pair core = {(std::size_t) j, (std::size_t) i};
            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = 200 * 1024;
            uint32_t num_input_tiles = 2;
            auto cb_src0 = ll_buda::CreateCircularBuffer(
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
            uint32_t num_output_tiles = 2;
            auto cb_output = ll_buda::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                num_output_tiles,
                num_output_tiles * single_tile_size,
                output_cb_addr,
                DataFormat::Float16_b
            );
        }
    }

    ll_buda::DataMovementKernel *reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_transpose_wh_8bank_partitioned.cpp",
        all_cores,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank_start_id_batched.cpp",
        all_cores,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new hlk_transpose_wh::hlk_args_t{ .NHtWt = int(Ht*Wt*NC) };
    ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(all_cores, hlk_args, sizeof(hlk_transpose_wh::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/transpose_wh.cpp",
        all_cores,
        eltwise_binary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    //bool profile_kernel = true;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    ll_buda::ConfigureDeviceWithProgram(device, program);
    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            int core_index = i * num_cores_c + j;
            tt_xy_pair core = {(std::size_t) j, (std::size_t) i};
            ll_buda::WriteRuntimeArgsToDevice(
                device,
                reader_kernel,
                core,
                {
                    src0_dram_buffer->address(),
                    (std::uint32_t)dram_src0_noc_xy.x,
                    (std::uint32_t)dram_src0_noc_xy.y,
                    num_tensor_tiles, NC, Ht, Wt, Ht*Wt,
                    (std::uint32_t)(i * Wt + j), 1, 1, Wt
                }
            );

            ll_buda::WriteRuntimeArgsToDevice(
                device,
                writer_kernel,
                core,
                {
                    dst_dram_buffer->address(),
                    (std::uint32_t)dram_dst_noc_xy.x,
                    (std::uint32_t)dram_dst_noc_xy.y,
                    (std::uint32_t)(j * Ht + i), 1, 0, 1
                }
            );
        }
    }

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data

    return output;
}

Tensor transpose_hc(const Tensor &a) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    u32 HW = H*W;
    u32 CHW = C*H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(C % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && N > 0 && C > 0);
    TT_ASSERT(TILE_WIDTH == TILE_HEIGHT && "Tile width and height must match for this kernel!");

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;
    u32 Ct = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;

    ll_buda::Program *program = new ll_buda::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    ll_buda::InterleavedDramBuffer *src0_dram_buffer = a.buffer();

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernel only needs the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);

    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {N, H, C, W};
    ll_buda::Tensor output = ll_buda::Tensor(output_shape, a.dtype(), tt::ll_buda::Layout::TILE, device);

    ll_buda::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = ll_buda::CreateCircularBuffer(
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
    uint32_t num_output_tiles = 2;
    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    ll_buda::DataMovementKernel *reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/transpose_hc_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new hlk_copy_binary::hlk_args_t{ .num_tensor_tiles = int(num_tensor_tiles) };
    ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(hlk_transpose_wh::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/eltwise_copy.cpp",
        core,
        eltwise_binary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            reader_kernel,
            core,
            {
                src0_dram_buffer->address(),
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                W, H, C, HW, N, CHW
            }
        );

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            writer_kernel,
            core,
            {
                dst_dram_buffer->address(),
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tensor_tiles
            }
        );

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data

    return output;
}

Tensor transpose(const Tensor &a) {
    return transpose_wh(a);
}

}  // namespace ll_buda

}  // namespace tt
