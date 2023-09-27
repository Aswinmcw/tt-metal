// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations) {
    std::map<string, string> defines;
    string op_name = "sub_tiles";
    string op_code = "1";
    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_code = "0";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_code = "1";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_code = "2";
            break;
        case BinaryOpType::GT: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ)); break;
        case BinaryOpType::LT: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LTZ)); break;
        case BinaryOpType::GTE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GEZ)); break;
        case BinaryOpType::LTE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LEZ)); break;
        case BinaryOpType::EQ: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EQZ)); break;
        case BinaryOpType::NE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ)); break;
        case BinaryOpType::SQUARED_DIFFERENCE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::SQUARE)); break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_tiles";
            op_code = "2";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ));
            break;
        case BinaryOpType::BIAS_GELU:
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GELU, 0));
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP, {}, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP, {}, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LOG));
            break;
        case BinaryOpType::LOGICAL_OR:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, {}, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, {}, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ));
	    break;
        case BinaryOpType::LDEXP:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, {}, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_code = "2";
            break;
        case BinaryOpType::LOGADDEXP2:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, {}, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, {}, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LOG2));
            break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_CODE"] = op_code.c_str();
    if (fused_activations.has_value()) {
        defines.merge(eltwise_unary_op_utils::get_block_defines(fused_activations.value()));
    }
    return defines;
}



}  // namespace eltwise_binary_op_utils

namespace tt {

namespace tt_metal {


void EltwiseBinary::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE), "Inputs to eltwise binary must be tilized");
    TT_ASSERT(input_tensor_a.dtype() == input_tensor_b.dtype());
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);
    if (input_tensor_a.memory_config().is_sharded()) {
        TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_ASSERT(input_tensor_a.memory_config() == input_tensor_b.memory_config());
            TT_ASSERT(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value());
        }
        if (this->output_mem_config.is_sharded()) {
            TT_ASSERT(input_tensor_a.memory_config() == this->output_mem_config);
        } else {
            TT_ASSERT(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_ASSERT(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->output_mem_config.is_sharded()) {
            TT_ASSERT(input_tensor_b.memory_config() == this->output_mem_config);
        } else {
            TT_ASSERT(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_ASSERT(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->output_mem_config.is_sharded()) {
            TT_ASSERT(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_ASSERT(num_blocks < num_cores || num_blocks % num_cores == 0);

        } else {
            TT_ASSERT(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }
}

std::vector<Shape> EltwiseBinary::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseBinary::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->output_mem_config.is_sharded()) {
        ShardSpec shard_spec{.shard_grid=CoreRangeSet({}), .shard_shape={0, 0}};
        if (input_tensor_a.memory_config().is_sharded()) {
            shard_spec = input_tensor_a.shard_spec().value();
        } else if (input_tensor_b.memory_config().is_sharded()) {
            shard_spec = input_tensor_b.shard_spec().value();
        } else {
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_grid_cores = core_grid.x * core_grid.y;
            uint32_t target_num_cores = num_blocks < num_grid_cores ? num_blocks : num_grid_cores;
            shard_spec.shard_grid = num_cores_to_corerange_set(target_num_cores, input_tensor_a.device()->compute_with_storage_grid_size(), true);
            shard_spec.shard_shape = {num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.shape()[-1]};
        }
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor_a.dtype(), Layout::TILE, input_tensor_a.device(), this->output_mem_config, shard_spec)};
    }
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks EltwiseBinary::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type, this->fused_activations);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default: return eltwise_binary_single_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type, this->fused_activations);
    }
}


BinaryOpParallelizationStrategy EltwiseBinary::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    if(num_tiles > 1){
           return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
       return BinaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes EltwiseBinary::attributes() const {
    return {
        {"op_type", this->op_type},
        {"fused_activations", this->fused_activations},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
