// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/rotate_half/rotate_half_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotate_half_single_core(const Tensor &input, Tensor &output) {
    Program program{};

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt::DataFormat scalar_cb_data_format = DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt_metal::detail::TileSize(scalar_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;
    uint32_t num_rows = input.volume()  / input.shape()[-1] / TILE_HEIGHT;
    uint32_t half_row_size = input.shape()[-1] / TILE_WIDTH / 2;

    tt_metal::Device *device = input.device();

    // Used for half of tensor that is multiplied
    uint32_t src_mul_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src_mul = tt_metal::CreateCircularBuffers(
        program,
        src_mul_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );
    // Used for bcast scalar
    uint32_t src_scalar_cb_index = 1;
    uint32_t num_scalar_tiles = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src_scalar_cb_index,
        core,
        num_scalar_tiles,
        num_scalar_tiles * scalar_single_tile_size,
        cb_data_format
    );
    // Used for half of tensor that is not multiplied
    uint32_t src_no_mul_cb_index = 2;
    auto cb_src_no_mul = tt_metal::CreateCircularBuffers(
        program,
        src_no_mul_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_mul_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_mul_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );
    uint32_t output_no_mul_cb_index = src_no_mul_cb_index;

    const uint16_t bfloat16_scalar = bfloat16(-1.0f).to_uint16();

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src_no_mul_cb_index,
        (uint32_t)src_mul_cb_index,
        (uint32_t)src_scalar_cb_index,
        (uint32_t)src_is_dram,
        (uint32_t)bfloat16_scalar
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_no_mul_cb_index,
        (std::uint32_t) output_mul_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_rotate_half_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_rotate_half_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    std::map<string, string> bcast_compute_defines = {
        {"BCAST_OP", "mul_tiles_bcast"},
        {"BCAST_LLKOP", "ELWMUL"},
        {"BCAST_DIM", "BroadcastType::SCALAR"},
		{"BCAST_SCALAR", "1"}
	};

	// TODO(AP): add dimensions and op params
	// Ignore Ht and just read num_tiles_per_core
	vector<uint32_t> compute_kernel_args = {
		1, // B
		1, // Ht
		num_tiles / 2  // Wt
	};

	auto bcast_kernel_group_1_id = tt_metal::CreateComputeKernel(
		program,
		"tt_metal/kernels/compute/bcast_hw.cpp",
		core,
		tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = bcast_compute_defines}
	);

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            src_buffer->address(),
            num_rows,
            half_row_size,
            0
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            num_rows,
            half_row_size,
            0
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
