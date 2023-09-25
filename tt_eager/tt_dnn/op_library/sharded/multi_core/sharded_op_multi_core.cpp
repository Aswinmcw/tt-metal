// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tensor/tensor_utils.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(const Tensor &input, Tensor &output) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, unit_size;

    tt_metal::Device *device = input.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto shard_spec = output.shard_spec().value();

    if (input.layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
        unit_size = tt_metal::detail::TileSize(cb_data_format);
        num_units_per_shard = shard_spec.shard_shape.first * shard_spec.shard_shape.second / TILE_HW;
    } else {
        num_units = input.volume() / input.shape()[-1];
        unit_size = input.shape()[-1] * input.element_size();
        num_units_per_shard = shard_spec.shard_shape.first;
    }

    auto all_cores = shard_spec.shard_grid;
    uint32_t num_cores_x = device->compute_with_storage_grid_size().x;
    uint32_t num_cores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        num_cores += core_range.size();
    }

    uint32_t src0_cb_index = 0;
    uint32_t num_input_units = num_units_per_shard;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        all_cores,
        num_input_units,
        num_input_units * round_up_to_mul32(unit_size),
        cb_data_format,
        output.buffer()->address(), true
    );

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    tt_metal::KernelID unary_reader_kernel_id;
    if (input.layout() == Layout::TILE) {

        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) src_is_dram
        };

        unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unit_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(unit_size) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) src_is_dram,
            (std::uint32_t) src_stick_size_is_power_of_two,
            (std::uint32_t) src_log2_stick_size
        };

        unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_stick_layout_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});
    }

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        all_cores,
        {
            num_units_per_shard
        }
    );

    for (uint32_t i = 0, num_units_written = 0; i < num_cores; i++){
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        if (!all_cores.core_coord_in_core_ranges(core)) {
            TT_ASSERT("Unexpected sharded layout");
        }

        if (input.layout() == Layout::TILE) {
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    src_buffer->address(),
                    num_units_per_shard,
                    1,
                    num_units_written
                }
            );
        } else {
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    src_buffer->address(),
                    unit_size,
                    num_units_per_shard,
                    1,
                    num_units_written
                }
            );
        }

        num_units_written += num_units_per_shard;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_x
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i % num_cores_x, i / num_cores_x};

            {
                auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(const Tensor &input, Tensor &output) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, unit_size;

    tt_metal::Device *device = input.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto shard_spec = input.shard_spec().value();
    if (input.layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
        unit_size = tt_metal::detail::TileSize(cb_data_format);
        num_units_per_shard = shard_spec.shard_shape.first * shard_spec.shard_shape.second / TILE_HW;
    } else {
        num_units = input.volume() / input.shape()[-1];
        unit_size = input.shape()[-1] * input.element_size();
        num_units_per_shard = shard_spec.shard_shape.first;
    }

    auto all_cores = shard_spec.shard_grid;
    uint32_t num_cores_x = device->compute_with_storage_grid_size().x;
    uint32_t num_cores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        num_cores += core_range.size();
    }

    uint32_t src0_cb_index = 0;
    uint32_t num_input_units = num_units_per_shard;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        all_cores,
        num_input_units,
        num_input_units * round_up_to_mul32(unit_size),
        cb_data_format,
        input.buffer()->address(), true
    );

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_cb_index
    };

    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});


    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    tt_metal::KernelID unary_writer_kernel_id;
    if (input.layout() == Layout::TILE) {
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) dst_is_dram
        };

        unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});
    } else {
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(unit_size);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(unit_size) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) dst_is_dram,
            (std::uint32_t) dst_stick_size_is_power_of_two,
            (std::uint32_t) dst_log2_stick_size
        };

        unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_stick_layout_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
    }

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        all_cores,
        {
            num_units_per_shard
        }
    );

    for (uint32_t i = 0, num_units_written = 0; i < num_cores; i++){
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        if (!all_cores.core_coord_in_core_ranges(core)) {
            TT_ASSERT("Unexpected sharded layout");
        }

        if (input.layout() == Layout::TILE) {
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {
                    dst_buffer->address(),
                    num_units_per_shard,
                    1,
                    num_units_written
                }
            );
        } else {
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {
                    dst_buffer->address(),
                    unit_size,
                    num_units_per_shard,
                    1,
                    num_units_written
                }
            );
        }
        num_units_written+=num_units_per_shard;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_x
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i % num_cores_x, i / num_cores_x};
            {
                auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
