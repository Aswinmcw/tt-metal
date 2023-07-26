#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace bcast_op_utils {
using namespace tt::tt_metal;
using namespace tt::constants;

// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy) {
    if (bcast_parallelization_strategy == BcastOpParallelizationStrategy::SINGLE_CORE) {
        if (bcast_dim == BcastOpDim::H) {
            return "tt_metal/kernels/dataflow/reader_bcast_h_8bank.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "tt_metal/kernels/dataflow/reader_bcast_w_8bank.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "tt_metal/kernels/dataflow/reader_bcast_hw_8bank.cpp";
        }
    }
    else {
        if (bcast_dim == BcastOpDim::H) {
            return "tt_metal/kernels/dataflow/reader_bcast_h_8bank_input_rows_partitioned.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "tt_metal/kernels/dataflow/reader_bcast_w_8bank_input_cols_partitioned.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "tt_metal/kernels/dataflow/reader_bcast_hw_8bank_partitioned.cpp";
        }
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastOpDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastOpDim::H:  return "tt_metal/kernels/compute/bcast_h.cpp";
        case BcastOpDim::W:  return "tt_metal/kernels/compute/bcast_w.cpp";
        case BcastOpDim::HW: return "tt_metal/kernels/compute/bcast_hw.cpp";
        default:  TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

void add_defines(ComputeKernel* k, BcastOpDim::Enum bcast_dim, BcastOpMath::Enum bcast_math)
{
    const char* math_to_op_define[] = { "add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast" };
    const char* math_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
    const char* bdim_to_llkdim_define[] = { "BroadcastType::ROW", "BroadcastType::COL", "BroadcastType::SCALAR" };
    k->add_define("BCAST_OP", math_to_op_define[int(bcast_math)]);
    k->add_define("BCAST_LLKOP", math_to_llkop_define[int(bcast_math)]);
    k->add_define("BCAST_DIM", bdim_to_llkdim_define[int(bcast_dim)]);
}

} // namespace bcast_op_utils


using namespace tt::tt_metal;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace tt_metal {

void EltwiseBinaryBroadcast::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    TT_ASSERT(input_tensor_a.device() != nullptr and input_tensor_b.device() != nullptr, "Operands to bcast need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to bcast need to be allocated in buffers on device!");

    TT_ASSERT((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE)
        || (input_tensor_a.layout() == Layout::TILE_CL && input_tensor_b.layout() == Layout::TILE_CL));
    TT_ASSERT(input_tensor_a.dtype() == input_tensor_b.dtype());
    TT_ASSERT(input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    auto true_input_shape_a = input_tensor_a.shape();
    auto true_input_shape_b = input_tensor_b.shape();
    if(input_tensor_a.layout() == Layout::TILE_CL) {
        true_input_shape_a = {true_input_shape_a[0], true_input_shape_a[2], true_input_shape_a[3], true_input_shape_a[1]};
        true_input_shape_b = {true_input_shape_b[0], true_input_shape_b[2], true_input_shape_b[3], true_input_shape_b[1]};
    }
    auto batch_size_a = true_input_shape_a[0];
    auto num_channels_a = true_input_shape_a[1];
    auto height_a = true_input_shape_a[2];
    auto width_a = true_input_shape_a[3];
    auto batch_size_b = true_input_shape_b[0];
    auto num_channels_b = true_input_shape_b[1];
    auto height_b = true_input_shape_b[2];
    auto width_b = true_input_shape_b[3];

    TT_ASSERT((batch_size_b * num_channels_b == 1 || (batch_size_b == batch_size_a && num_channels_b == num_channels_a)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");

    // validate input dimensions
    if (this->dim == BcastOpDim::W)
        TT_ASSERT(height_a == height_b && width_b == TILE_WIDTH);
    if (this->dim == BcastOpDim::H)
        TT_ASSERT(width_a == width_b && height_b == TILE_HEIGHT);
    if (this->dim == BcastOpDim::HW)
        TT_ASSERT(width_b == TILE_WIDTH && height_b == TILE_HEIGHT);
}


std::vector<Shape> EltwiseBinaryBroadcast::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}


std::vector<Tensor> EltwiseBinaryBroadcast::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.layout() == Layout::TILE || input_tensor.layout() == Layout::TILE_CL);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks EltwiseBinaryBroadcast::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case BcastOpParallelizationStrategy::MULTI_CORE_H:
            return bcast_multi_core_h(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
        case BcastOpParallelizationStrategy::MULTI_CORE_W:
            return bcast_multi_core_w(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
        case BcastOpParallelizationStrategy::MULTI_CORE_HW:
            return bcast_multi_core_hw(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
        case BcastOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bcast_single_core(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
    }
}

operation::Hash EltwiseBinaryBroadcast::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    return fmt::format("{}_{}_{}", *this, input_tensor_a, input_tensor_b);
}

tt::stl::reflection::Attributes EltwiseBinaryBroadcast::attributes() const {
    return {
        {"math_op", this->math_op},
        {"dim", this->dim},
        {"output_mem_config", this->output_mem_config},
    };
}

BcastOpParallelizationStrategy::Enum EltwiseBinaryBroadcast::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);

    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    uint32_t Ht = input_tensor_a.shape()[2] / TILE_HEIGHT;
    uint32_t Wt = input_tensor_a.shape()[3] / TILE_WIDTH;

    if(Ht > 1 and this->dim == BcastOpDim::H){
        return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(Wt > 1 and this->dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(num_tiles > 1 and this->dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        return BcastOpParallelizationStrategy::SINGLE_CORE;
    }
}

}  // namespace tt_metal

}  // namespace tt
