// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_single_core(
    const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output, std::optional<uint32_t> token_idx) {
    Program program{};

    CoreRangeSet core({CoreRange{.start = {0, 0}, .end = {0, 0}}});

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt_metal::detail::TileSize(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt_metal::detail::TileSize(sin_cb_data_format);

    tt::DataFormat scalar_cb_data_format = DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt_metal::detail::TileSize(scalar_cb_data_format);

    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;
    uint32_t num_rows = input.volume() / input.shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.shape()[-1] * sizeof(bfloat16);

    tt_metal::Device *device = input.device();

    uint32_t input_cb_index = CB::c_in0;
    uint32_t num_input_tiles = 2 * Wt;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size);
    auto cb_input = tt_metal::CreateCircularBuffer(program, core, cb_input_config);

    uint32_t rotated_input_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig cb_rotated_input_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{rotated_input_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_cb_index, input_single_tile_size);
    auto cb_rotated_input = tt_metal::CreateCircularBuffer(program, core, cb_rotated_input_config);

    // TODO: Debug why this can't be double buffered
    uint32_t num_cos_sin_tiles = token_idx.has_value() ? Wt : 2 * Wt;

    uint32_t cos_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig cb_cos_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size);
    auto cb_cos = tt_metal::CreateCircularBuffer(program, core, cb_cos_config);

    uint32_t sin_cb_index = CB::c_in3;
    tt_metal::CircularBufferConfig cb_sin_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size);
    auto cb_sin = tt_metal::CreateCircularBuffer(program, core, cb_sin_config);

    // Used for bcast scalar
    uint32_t src_scalar_cb_index = CB::c_in4;
    uint32_t num_scalar_tiles = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(
            num_scalar_tiles * scalar_single_tile_size, {{src_scalar_cb_index, scalar_cb_data_format}})
            .set_page_size(src_scalar_cb_index, scalar_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t num_interm_tiles = 1;
    uint32_t rotated_input_interm_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    auto cb_rotated_input_interm = tt_metal::CreateCircularBuffer(program, core, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    auto cb_cos_interm = tt_metal::CreateCircularBuffer(program, core, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    auto cb_sin_interm = tt_metal::CreateCircularBuffer(program, core, cb_sin_interm_config);

    uint32_t output_cb_index = CB::c_out0;  // output operands start at index 16
    uint32_t num_output_tiles = 2 * Wt;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    uint32_t untilized_cos_interm_cb_index = CB::c_intermed3;
    uint32_t untilized_cos_sync_cb_index = CB::c_in5;
    uint32_t untilized_sin_interm_cb_index = CB::c_intermed4;
    uint32_t untilized_sin_sync_cb_index = CB::c_in6;
    uint32_t retilized_cos_cb_index = CB::c_intermed5;
    uint32_t retilized_sin_cb_index = CB::c_intermed6;
    std::map<string, string> kernel_defines;
    if (token_idx.has_value()) {
        tt_metal::CircularBufferConfig cb_cos2_config =
            tt_metal::CircularBufferConfig(
                num_cos_sin_tiles * cos_single_tile_size, {{retilized_cos_cb_index, cos_cb_data_format}})
                .set_page_size(retilized_cos_cb_index, cos_single_tile_size);
        auto cb_cos2 = tt_metal::CreateCircularBuffer(program, core, cb_cos2_config);

        tt_metal::CircularBufferConfig cb_sin2_config =
            tt_metal::CircularBufferConfig(Wt * sin_single_tile_size, {{retilized_sin_cb_index, sin_cb_data_format}})
                .set_page_size(retilized_sin_cb_index, sin_single_tile_size);
        auto cb_sin2 = tt_metal::CreateCircularBuffer(program, core, cb_sin2_config);

        std::map<uint8_t, tt::DataFormat> cos_interim_data_format_spec = {
            {untilized_cos_interm_cb_index, scalar_cb_data_format},
            {untilized_cos_sync_cb_index, scalar_cb_data_format}};
        tt_metal::CircularBufferConfig cb_untilized_cos_interm_config =
            tt_metal::CircularBufferConfig(Wt * scalar_single_tile_size, cos_interim_data_format_spec)
                .set_page_size(untilized_cos_interm_cb_index, scalar_single_tile_size)
                .set_page_size(untilized_cos_sync_cb_index, scalar_single_tile_size);
        auto cb_untilized_cos_interm = tt_metal::CreateCircularBuffer(program, core, cb_untilized_cos_interm_config);

        std::map<uint8_t, tt::DataFormat> sin_interim_data_format_spec = {
            {untilized_sin_interm_cb_index, scalar_cb_data_format},
            {untilized_sin_sync_cb_index, scalar_cb_data_format}};
        tt_metal::CircularBufferConfig cb_untilized_sin_interm_config =
            tt_metal::CircularBufferConfig(Wt * scalar_single_tile_size, sin_interim_data_format_spec)
                .set_page_size(untilized_sin_interm_cb_index, scalar_single_tile_size)
                .set_page_size(untilized_sin_sync_cb_index, scalar_single_tile_size);
        auto cb_untilized_sin_interm = tt_metal::CreateCircularBuffer(program, core, cb_untilized_sin_interm_config);
        kernel_defines["DECODE_MODE"] = "1";
    }

    const uint16_t bfloat16_scalar = bfloat16(-1.0f).to_uint16();

    auto src_buffer = input.buffer();
    auto cos_buffer = cos.buffer();
    auto sin_buffer = sin.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool cos_is_dram = cos_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool sin_is_dram = sin_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)cos_is_dram,
        (std::uint32_t)sin_is_dram,
        (std::uint32_t)bfloat16_scalar,
        (std::uint32_t)Ht,
        (std::uint32_t)Wt,
        (std::uint32_t)HtWt,
        (std::uint32_t)half_Wt,
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    if (token_idx.has_value()) {
        writer_compile_time_args.insert(
            writer_compile_time_args.end(),
            {untilized_cos_interm_cb_index,
             untilized_cos_sync_cb_index,
             untilized_sin_interm_cb_index,
             untilized_sin_sync_cb_index,
             Wt,
             Wbytes});
    }

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/rotary_embedding/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args, .defines = kernel_defines});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/rotary_embedding/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args, .defines = kernel_defines});

    vector<uint32_t> compute_kernel_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)num_rows,
        (std::uint32_t)Wt,
        (std::uint32_t)half_Wt};
    if (token_idx.has_value()) {
        compute_kernel_args.insert(
            compute_kernel_args.end(),
            {(std::uint32_t)untilized_cos_interm_cb_index,
             (std::uint32_t)untilized_cos_sync_cb_index,
             (std::uint32_t)untilized_sin_interm_cb_index,
             (std::uint32_t)untilized_sin_sync_cb_index,
             (std::uint32_t)retilized_cos_cb_index,
             (std::uint32_t)retilized_sin_cb_index});
    }

    auto rotary_embedding_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/rotary_embedding/kernels/compute/rotary_embedding.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = kernel_defines});

    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src_buffer->address(), cos_buffer->address(), sin_buffer->address(), num_rows, 0, 0, cos_sin_start_id});

    SetRuntimeArgs(
        program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0, cos_sin_offset, Wt, Wbytes});

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, Wbytes, Wt](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        const auto token_idx = static_cast<const RotaryEmbedding *>(operation)->token_idx;

        auto src_buffer = input_tensors.at(0).buffer();
        auto cos_buffer = input_tensors.at(1).buffer();
        auto sin_buffer = input_tensors.at(2).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        uint32_t cos_sin_offset = 0;
        uint32_t cos_sin_start_id = 0;
        if (token_idx.has_value()) {
            cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
            cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
        }

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = cos_buffer->address();
            runtime_args[2] = sin_buffer->address();
            runtime_args[6] = cos_sin_start_id;
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[3] = cos_sin_offset;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
