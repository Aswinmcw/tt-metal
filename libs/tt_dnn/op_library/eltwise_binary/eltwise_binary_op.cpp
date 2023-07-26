#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

void add_defines(ComputeKernel* eltwise_binary_kernel, BinaryOpType::Enum op_type) {
    string op_name = "sub_tiles";
    string op_code = "1";
    string compare = "1";
    string compare_init = "";
    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_code = "0";
            compare = "0";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_code = "1";
            compare = "0";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_code = "2";
            compare = "0";
            break;
        case BinaryOpType::GT: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::GTZ); break;
        case BinaryOpType::LT: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::LTZ); break;
        case BinaryOpType::GTE: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::GEZ); break;
        case BinaryOpType::LTE: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::LEZ); break;
        case BinaryOpType::EQ: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::EQZ); break;
        case BinaryOpType::NE: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::NEZ); break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_code.c_str());
    if ( compare == "1" ) {
      eltwise_binary_kernel->add_define("ELTWISE_COMPARE_BINARY_OP", compare);
      eltwise_binary_kernel->add_define("SFPU_OP_AND_PACK", compare_init);
    }
}



}  // namespace eltwise_binary_op_utils

namespace tt {

namespace tt_metal {


void EltwiseBinary::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    cout << "input tensor a shape - " << endl;
    cout << input_tensor_a.shape()[1] << " " << input_tensor_a.shape()[2] << " " << input_tensor_a.shape()[3] << endl;
    cout << "input tensor b shape - " << endl;
    cout << input_tensor_b.shape()[1] << " " << input_tensor_b.shape()[2] << " " << input_tensor_b.shape()[3] << endl;
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE)
        || (input_tensor_a.layout() == Layout::TILE_CL && input_tensor_b.layout() == Layout::TILE_CL), "Inputs to eltwise binary must be tilized, either TILE or TILE_CL");
    TT_ASSERT(input_tensor_a.dtype() == input_tensor_b.dtype());
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);
    TT_ASSERT((input_tensor_a.buffer()->buffer_type() == BufferType::DRAM && input_tensor_b.buffer()->buffer_type() == BufferType::DRAM));
}

std::vector<Shape> EltwiseBinary::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseBinary::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.layout() == Layout::TILE || input_tensor.layout() == Layout::TILE_CL);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), MemoryConfig{.interleaved = true});
}

operation::ProgramWithCallbacks EltwiseBinary::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default: return eltwise_binary_single_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type);
    }
}

operation::Hash EltwiseBinary::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    return fmt::format("{}_{}_{}", *this, input_tensor_a, input_tensor_b);
}


BinaryOpParallelizationStrategy::Enum EltwiseBinary::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
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
    };
}

}  // namespace tt_metal

}  // namespace tt
