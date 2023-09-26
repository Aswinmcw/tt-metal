// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_eager/tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks scale_mask_softmax_(const Tensor &input_tensor, const std::optional<const Tensor> mask, const Tensor &output_tensor, std::optional<float> scale) {

    const auto shape = input_tensor.shape();
    uint32_t W = shape[3], H = shape[2] * shape[1], NC = shape[0];
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    Program program = Program();

    uint32_t scalar_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::DataFormat in0_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t in0_tile_size = tt_metal::detail::TileSize(in0_cb_data_format);

    tt::DataFormat mask_cb_data_format = mask.has_value() ? tt_metal::datatype_to_dataformat_converter(mask.value().dtype()) : tt::DataFormat::Float16_b;
    uint32_t mask_tile_size = tt_metal::detail::TileSize(mask_cb_data_format);

    tt::DataFormat out_cb_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_cb_data_format);

    auto src0_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    int32_t num_tiles = input_tensor.volume()/TILE_HW;

    // This should allocate input_tensor DRAM buffer on the device
    Device *device = input_tensor.device();

    uint32_t block_size = find_max_divisor(Wt, 8);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  = block_size*2;
    uint32_t out0_t = block_size*2;
    uint32_t im1_t  = 1; // 1/sum(exp(x))
    uint32_t in2_t  = 1; // scaler for reduce coming from reader
    uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t  = div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled

    // cb_exps - keeps exps in CB in L1 to avoid recomputing
    uint32_t im0_t  = block_size*div_up(Wt, block_size);
    TT_ASSERT(im0_t == Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t  = block_size*(div_up(Wt, block_size)+1);
    TT_ASSERT(im3_t == Wt+block_size);

    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in4_t % block_size == 0);
    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

    uint32_t NCHt = NC*Ht;
    CoreGridDesc grid(input_tensor.device());
    uint32_t num_cores = grid.numcores_dividing_numtiles(NCHt);
    uint32_t partHt = NCHt/num_cores; // only used by fused_scale_mask variant

    // we are actually splitting blocks of Wt tiles, not tiles, so no checking for bank alignment is needed
    TilesSplit ts(num_cores, NCHt);
    auto wtpc = ts.get_tpc();
    TT_ASSERT(NCHt % wtpc == 0);
    TT_ASSERT(NCHt % num_cores == 0);
    TT_ASSERT(wtpc < Ht || (wtpc % Ht == 0));
    TT_ASSERT(NCHt % num_cores == 0);
    TT_ASSERT(partHt >= Ht || Ht % partHt == 0);
    //cout << "NUM CORES=" << num_cores << " WTPC=" << wtpc << " partHt=" << partHt << endl;

    // Parallelize across rows
    // TODO: Refactor by calling utility function?
    uint32_t num_full_rows = num_cores / grid.x_;
    uint32_t last_row_cores = num_cores % grid.x_;

    std::set<CoreRange> all_cores_set;
    if (num_full_rows) {
        all_cores_set.insert((CoreRange) {
            .start={0, 0}, .end={grid.x_ - 1, num_full_rows - 1}
        });
    }
    if (last_row_cores) {
        all_cores_set.insert((CoreRange) {
            .start={0, num_full_rows}, .end={last_row_cores - 1, num_full_rows}
        });
    }
    CoreRangeSet all_cores(all_cores_set);
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        src0_is_dram,
        block_size
    };
    if (mask.has_value()) {
        bool mask_is_dram = mask.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        reader_compile_time_args.push_back(mask_is_dram);
    }
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        dst_is_dram,
        block_size
    };
    std::map<string, string> softmax_defines;
    if (mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    auto reader_kernels_id = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/reader_unary_interleaved_sm.cpp", all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = softmax_defines
        });
        //DataMovementProcessor::RISCV_1, core.x < 6 ? NOC::RISCV_1_default : NOC::RISCV_0_default);

    auto writer_kernels_id = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp", all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = softmax_defines
        });
        //DataMovementProcessor::RISCV_0, core.x < 6 ? NOC::RISCV_0_default : NOC::RISCV_1_default);

    // for broadcasting in H direction we need to
    // NCHt, Nt, Wt
    // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
    // if wtpc >= Ht then tpc should be a multiple of Ht
    vector<uint32_t> compute_args = { wtpc, partHt, Wt, block_size };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto softmax_kernels_id = CreateComputeKernel(
        program, "kernels/compute/softmax.cpp", all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode,
            .compile_args = compute_args,
            .defines = softmax_defines
        });

    // Create circular buffers
    // see softmax.cpp for which buffers are needed
    CreateCircularBuffers( program, CB::c_in0,       all_cores, in0_t,  in0_t * in0_tile_size, in0_cb_data_format );
    CreateCircularBuffers( program, CB::c_out0,      all_cores, out0_t, out0_t * out_tile_size, out_cb_data_format );
    CreateCircularBuffers( program, CB::c_intermed1, all_cores, im1_t,  im1_t * in0_tile_size,  in0_cb_data_format );
    CreateCircularBuffers( program, CB::c_in2,       all_cores, in2_t,  in2_t * scalar_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffers( program, CB::c_intermed0, all_cores, im0_t,  im0_t * in0_tile_size,  in0_cb_data_format );
    if (mask.has_value()) {
        CreateCircularBuffers( program, CB::c_intermed3, all_cores, im3_t,  im3_t * in0_tile_size,  in0_cb_data_format );
        CreateCircularBuffers( program, CB::c_in3,       all_cores, in3_t,  in3_t * scalar_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffers( program, CB::c_in4,       all_cores, in4_t,  in4_t * mask_tile_size,  mask_cb_data_format );
    }
    uint32_t src_addr = src0_buffer->address();
    uint32_t mask_addr = mask.has_value() ? mask.value().buffer()->address() : 0;
    uint32_t dst_addr = dst_buffer->address();

    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);

        uint32_t tile_offset = wtpc*Wt*icore;
        union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
        // always in-place
        //                                                              0  1    2       3            4   5       6          7           8
        SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, 0, s.u, wtpc*Wt, tile_offset, partHt, Wt, mask_addr, 0x3f800000 }); // [8]=1.0f is scaler
        SetRuntimeArgs(program, writer_kernels_id, core, { dst_addr, wtpc*Wt, tile_offset });
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernels_id,
            writer_kernel_id=writer_kernels_id,
            num_cores,
            grid
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {
        TT_ASSERT(input_buffers.size() == 2);

        auto src_buffer = input_buffers.at(0);
        auto mask_buffer = input_buffers.at(1);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            auto core = grid.wrap_core(icore);

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                if (mask_buffer != nullptr) {
                    runtime_args[7] = mask_buffer->address();
                }
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
} // scale_mask_softmax_


void Softmax::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1 and optional_input_tensors.size() <= 1, "Must have 1 or 2 input tensors");
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr , "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(this->dim == input_tensor.shape().rank() - 1, "Only softmax on last dim is supported");
    if (optional_input_tensors.size() == 1) {
        if (optional_input_tensors.at(0).has_value()) {
            auto& mask = optional_input_tensors.at(0).value();
            TT_ASSERT(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
            TT_ASSERT(input_tensor.device() == mask.device());
            TT_ASSERT(input_tensor.dtype() == mask.dtype());
        } else {
            TT_ASSERT(not this->scale.has_value());
        }
    } else {
        TT_ASSERT(not this->scale.has_value());
    }


}

std::vector<Shape> Softmax::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> Softmax::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    if (this->in_place) {
        return {};
    } else {
        auto& input_tensor = input_tensors.at(0);
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Softmax::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& input_tensor = input_tensors.at(0);
    const auto& mask = optional_input_tensors.at(0);
    if (this->in_place) {
        return scale_mask_softmax_(input_tensor, mask, input_tensor, this->scale);
    } else {
        const auto& output_tensor = output_tensors.at(0);
        return scale_mask_softmax_(input_tensor, mask, output_tensor, this->scale);
    }

}

tt::stl::reflection::Attributes Softmax::attributes() const {
    return {
        {"scale", this->scale},
        {"dim", this->dim},
        {"in_place", this->in_place},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor softmax_in_place(Tensor& input_tensor) {
    return transformers::scale_mask_softmax_in_place(input_tensor, std::nullopt, std::nullopt);
}

namespace transformers {
Tensor scale_mask_softmax_in_place(Tensor& input_tensor, std::optional<float> scale, std::optional<const Tensor> mask) {
    operation::run(Softmax{.scale=scale, .dim=input_tensor.shape().rank() - 1, .in_place=true, .output_mem_config=input_tensor.memory_config()}, {input_tensor}, {mask});
    return input_tensor;
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations

namespace tt_metal {

Tensor softmax(const Tensor& input_tensor, const std::int64_t dim, const MemoryConfig& output_mem_config) {
    return scale_mask_softmax(input_tensor, dim, std::nullopt, std::nullopt, output_mem_config);
}

Tensor scale_mask_softmax(const Tensor& input_tensor, const std::int64_t dim, std::optional<float> scale, std::optional<const Tensor> mask, const MemoryConfig& output_mem_config) {
    uint32_t normalized_dim = input_tensor.shape().get_normalized_index(dim);
    uint32_t last_dim = input_tensor.shape().rank() - 1;
    bool transpose_input = normalized_dim != last_dim;
    Tensor transposed_input_tensor = input_tensor;
    if (transpose_input) {
        TT_ASSERT(!mask.has_value(), "Non-last dim softmax does not support mask input");
        transposed_input_tensor = transpose(input_tensor, normalized_dim, last_dim);
    }

    Shape pad_shape = AutoFormat::pad_to_tile_shape(transposed_input_tensor.shape());
    FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=-std::numeric_limits<float>::lowest(), .target_layout=Layout::TILE};
    Tensor output = operation::run_with_autoformat(operations::primary::Softmax{.scale=scale, .dim=last_dim, .in_place=false}, {transposed_input_tensor}, {input_format_params}, {Layout::TILE}, {mask}).at(0);
    if (transpose_input) {
        output = transpose(input_tensor, normalized_dim, last_dim);
    }
    return output;
}
}  // namespace tt_metal
}  // namespace tt
