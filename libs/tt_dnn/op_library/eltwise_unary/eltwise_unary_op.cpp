#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using namespace tt::constants;

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

static const unordered_map<UnaryOpType::Enum ,string> optype_to_name = {
    {UnaryOpType::Enum::EXP,"EXP"},
    {UnaryOpType::Enum::RECIP,"SURECIPB"},
    {UnaryOpType::Enum::GELU,"GELU"},
    {UnaryOpType::Enum::RELU,"RELU"},
    {UnaryOpType::Enum::SQRT,"SQRT"},
    {UnaryOpType::Enum::SIGMOID,"SIGMOID"},
    {UnaryOpType::Enum::LOG,"LOG"},
    {UnaryOpType::Enum::TANH,"TANH"}
};

string get_op_name(UnaryOpType::Enum op_type) {
    string op_name;
    switch (op_type) {
        case UnaryOpType::EXP: op_name = "exp_tile_init(); exp_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RECIP: op_name = "recip_tile_init(); recip_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::GELU: op_name = "gelu_tile_init(); gelu_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RELU: op_name = "pack_relu_tile_to_stream(0, CB::c_out0);"; break;
        case UnaryOpType::SQRT: op_name = "sqrt_tile_init(); sqrt_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::SIGMOID: op_name = "sigmoid_tile_init(); sigmoid_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::LOG: op_name = "log_tile_init(); log_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::TANH: op_name = "tanh_tile_init(); tanh_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            op_name = "log_with_base_tile_init(); log_with_base_tile(0,0x36f3); pack_tile(0,CB::c_out0);";
            break;
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            op_name = "log_with_base_tile_init(); log_with_base_tile(0,0x3dc5); pack_tile(0,CB::c_out0);";
            break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    return op_name;
}

static std::string op_type_to_string(UnaryOpType::Enum op_type) {
    switch (op_type) {
        case UnaryOpType::EXP: return "EXP";
        case UnaryOpType::RECIP: return "RECIP";
        case UnaryOpType::GELU: return "GELU";
        case UnaryOpType::RELU: return "RELU";
        case UnaryOpType::SQRT: return "SQRT";
        case UnaryOpType::SIGMOID: return "SIGMOID";
        case UnaryOpType::LOG: return "LOG";
        case UnaryOpType::TANH: return "TANH";
        case UnaryOpType::LOG10: return "LOG10";
        case UnaryOpType::LOG2: return "LOG2";
    }
    throw std::runtime_error("Undefined op type");
}

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type){
    string op_name = get_op_name(op_type);
    eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", op_name);
    bool is_relu = (op_type == UnaryOpType::RELU);
    eltwise_unary_kernel->add_define("INIT_RELU", is_relu ? "pack_relu_config(1);" : "");
    eltwise_unary_kernel->add_define("DEINIT_RELU", is_relu ? "pack_relu_config(0);" : "");
    return;
}

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a){
    uint32_t num_tiles = a.volume() / TILE_HW;
    if(num_tiles > 1){
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return UnaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

}  // namespace eltwise_unary_op_utils

namespace tt {

namespace tt_metal {

ProgramHash EltwiseUnary::compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    const auto& input_shape = input_tensor.shape();

    auto op_type = eltwise_unary_op_utils::op_type_to_string(this->op_type);
    return fmt::format("{}_[{},{},{},{}]", op_type, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
}

void EltwiseUnary::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {}

std::vector<Shape> EltwiseUnary::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseUnary::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors);
}

Program EltwiseUnary::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);

    TT_ASSERT(eltwise_unary_op_utils::optype_to_name.find(this->op_type) != \
            eltwise_unary_op_utils::optype_to_name.end(), "Eltwise op not defined");

    profiler::set_preferred_name(eltwise_unary_op_utils::optype_to_name.at(this->op_type));
    switch (eltwise_unary_op_utils::get_parallelization_strategy(input_tensor)){
        case UnaryOpParallelizationStrategy::MULTI_CORE:
            profiler::set_parallelization_strategy("MULTI_CORE");
            return eltwise_unary_multi_core(input_tensor, output_tensor, this->op_type);
            break;
        case UnaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            profiler::set_parallelization_strategy("SINGLE_CORE");
            return eltwise_unary_single_core(input_tensor, output_tensor, this->op_type);
    }

}

}  // namespace tt_metal

}  // namespace tt
