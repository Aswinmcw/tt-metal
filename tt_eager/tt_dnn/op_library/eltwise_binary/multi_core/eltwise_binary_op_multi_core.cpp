// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_binary_multi_core(const Tensor &a, const Tensor &b, Tensor& output, BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations) {

    Program program{};

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tiles, true);
    std::optional<ShardSpec> shard_spec = std::nullopt;
    if (a.memory_config().is_sharded()) {
        shard_spec = a.shard_spec().value();
    } else if (b.memory_config().is_sharded()) {
        shard_spec = b.shard_spec().value();
    } if (output.memory_config().is_sharded()) {
        shard_spec = output.shard_spec().value();
    }
    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().shard_grid;
        num_cores = 0;
        for (const auto& core_range : all_cores.ranges()) {
            num_cores += core_range.size();
        }
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet({});
        num_tiles_per_core_group_1 = shard_spec.value().shard_shape[0] * shard_spec.value().shard_shape[1] / TILE_HW;
        num_tiles_per_core_group_2 = 0;
    }

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    if (a.memory_config().is_sharded()) {
        uint32_t num_input_tiles = 2;
        auto cb_src0 = tt_metal::CreateCircularBuffers(
            program,
            src0_cb_index,
            all_cores,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_1 * src0_single_tile_size,
            src0_cb_data_format,
            a.buffer()->address(),
            true
        );
    } else {
        auto cb_src0 = tt_metal::CreateCircularBuffers(
            program,
            src0_cb_index,
            all_cores,
            num_input_tiles,
            num_input_tiles * src0_single_tile_size,
            src0_cb_data_format
        );
    }

    uint32_t src1_cb_index = 1;

    if (b.memory_config().is_sharded()) {
        auto cb_src1 = tt_metal::CreateCircularBuffers(
            program,
            src1_cb_index,
            all_cores,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_1 * src1_single_tile_size,
            src1_cb_data_format,
            b.buffer()->address(),
            true
        );
    } else {
        auto cb_src1 = tt_metal::CreateCircularBuffers(
            program,
            src1_cb_index,
            all_cores,
            num_input_tiles,
            num_input_tiles * src1_single_tile_size,
            src1_cb_data_format
        );
    }

    std::map<string, string> eltwise_defines = eltwise_binary_op_utils::get_defines(op_type, fused_activations);

    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN0_0") != eltwise_defines.end()) {
        auto cb_interm = tt_metal::CreateCircularBuffers(
            program,
            CB::c_intermed0,
            all_cores,
            1,
            1 * src0_single_tile_size,
            src0_cb_data_format
        );
    }
    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN1_0") != eltwise_defines.end()) {
        auto cb_interm2 = tt_metal::CreateCircularBuffers(
            program,
            CB::c_intermed1,
            all_cores,
            1,
            1 * src1_single_tile_size,
            src1_cb_data_format
        );
    }

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    if (output.memory_config().is_sharded()) {
        auto cb_output = tt_metal::CreateCircularBuffers(
            program,
            output_cb_index,
            all_cores,
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_1 * dst_single_tile_size,
            dst_cb_data_format,
            output.buffer()->address(),
            true
        );
    } else {
        auto cb_output = tt_metal::CreateCircularBuffers(
            program,
            output_cb_index,
            all_cores,
            num_output_tiles,
            num_output_tiles * dst_single_tile_size,
            dst_cb_data_format
        );
    }
    std::map<string, string> reader_defines;
    if (a.memory_config().is_sharded()) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    if (b.memory_config().is_sharded()) {
        reader_defines["IN1_SHARDED"] = "1";
    }
    std::map<string, string> writer_defines;
    if (output.memory_config().is_sharded()) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) src1_is_dram
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines = reader_defines});

    KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args, .defines = writer_defines});

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1, // per_core_block_cnt
        1 // per_core_block_size
    };

    auto eltwise_binary_kernel_group_1_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_1, .defines = eltwise_defines}
    );

    if(!core_group_2.ranges().empty()){
        vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2, // per_core_block_cnt
            1 // per_core_block_size
        };

        auto eltwise_binary_kernel_group_2_id = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_2, .defines = eltwise_defines}
        );
    }

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++){
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                src0_buffer->address(),
                src1_buffer->address(),
                num_tiles_per_core,
                num_tiles_read
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_tiles_per_core,
                num_tiles_read
            }
        );
        num_tiles_read+=num_tiles_per_core;
    }

    auto override_runtime_args_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_x
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer_a = input_buffers.at(0);

        auto src_buffer_b = input_buffers.at(1);

        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i % num_cores_x, i / num_cores_x};

            {
                auto runtime_args = GetRuntimeArgs(program, binary_reader_kernel_id, core);
                runtime_args[0] = src_buffer_a->address();
                runtime_args[1] = src_buffer_b->address();
                SetRuntimeArgs(program, binary_reader_kernel_id, core, runtime_args);
            }

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
