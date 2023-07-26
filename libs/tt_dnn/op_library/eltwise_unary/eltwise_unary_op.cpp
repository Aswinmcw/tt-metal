#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;


union Converter {
public:
  float f;
  uint32_t u;

  Converter(float f_) : f(f_) {};

  static
  std::string to_hex(float f_) {
    Converter obj(f_);
    std::stringstream ss;
    ss << "0x" << std::hex << obj.u;
    return ss.str();
  }
};

inline
string get_op_name_parameterized(UnaryOpType::Enum op_type,float param0) {
    string op_name;
    TT_ASSERT( is_parametrized_type(op_type) && "operator should support one parameter" );

    switch (op_type) {
    case UnaryOpType::RELU_MAX: op_name = "relu_max_tile_init(); relu_max_tile(0,"+Converter::to_hex(param0)+"u ); pack_tile(0, tt::CB::c_out0);"; break;
    case UnaryOpType::RELU_MIN: op_name = "relu_min_tile_init(); relu_min_tile(0,"+Converter::to_hex(param0)+"u ); pack_tile(0, tt::CB::c_out0);"; break;
    case UnaryOpType::POWER: op_name = "power_tile_init(); power_tile(0," + std::to_string( (uint32_t) param0) + " ); pack_tile(0, tt::CB::c_out0);"; break;
    case UnaryOpType::LEAKY_RELU: op_name = "leaky_relu_tile_init(); leaky_relu_tile(0,"+Converter::to_hex(param0)+"u); pack_tile(0, tt::CB::c_out0);"; break;
    case UnaryOpType::ELU: op_name = "elu_tile_init(); elu_tile(0,"+Converter::to_hex(param0)+"u); pack_tile(0, tt::CB::c_out0);"; break;
    case UnaryOpType::GELU: op_name = "gelu_tile_init(); gelu_tile(0,"+std::to_string((uint32_t)param0)+"u); pack_tile(0, tt::CB::c_out0);"; break;
    case UnaryOpType::HEAVISIDE: op_name = "heaviside_tile_init(); heaviside_tile(0,"+Converter::to_hex(param0)+"u); pack_tile(0, tt::CB::c_out0);"; break;
    default:
	  TT_ASSERT( false && "unexpected parameterized type");
    };
    return op_name;
}

inline
string get_op_name_default(UnaryOpType::Enum op_type) {
    string op_name;
    switch (op_type) {
        case UnaryOpType::EXP: op_name = "exp_tile_init(); exp_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::RECIP: op_name = "recip_tile_init(); recip_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::RELU: op_name = "relu_min_tile_init(); relu_min_tile(0,0x0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::SQRT: op_name = "sqrt_tile_init(); sqrt_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::SIGMOID: op_name = "sigmoid_tile_init(); sigmoid_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::LOG: op_name = "log_tile_init(); log_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::TANH: op_name = "tanh_tile_init(); tanh_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::SIN: op_name = "sin_tile_init(); sin_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::COS: op_name = "cos_tile_init(); cos_tile(0); pack_tile(0, tt::CB::c_out0);"; break;
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            op_name = "log_with_base_tile_init(); log_with_base_tile(0,0x36f3); pack_tile(0,tt::CB::c_out0);";
            break;
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            op_name = "log_with_base_tile_init(); log_with_base_tile(0,0x3dc5); pack_tile(0,tt::CB::c_out0);";
            break;
        case UnaryOpType::ABS:
            op_name = "abs_tile_init(); abs_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::SIGN:
            op_name = "sign_tile_init(); sign_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::SQUARE:
            op_name = "square_tile_init(); square_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::EQZ:
            op_name = "eqz_tile_init(); eqz_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::NEZ:
            op_name = "nez_tile_init(); nez_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::LTZ:
            op_name = "ltz_tile_init(); ltz_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::GTZ:
            op_name = "gtz_tile_init(); gtz_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::LEZ:
            op_name = "lez_tile_init(); lez_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::GEZ:
            op_name = "gez_tile_init(); gez_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::EXP2:
            op_name = "exp2_tile_init(); exp2_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        case UnaryOpType::EXPM1:
            op_name = "expm1_tile_init(); expm1_tile(0); pack_tile(0,tt::CB::c_out0);"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    return op_name;
}

bool get_op_approx_mode(UnaryOpType::Enum op_type) {
    switch (op_type) {
        default:
            return false;
    }
}

static
void add_defines_impl(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type, std::string op_name){
    eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", op_name);
    return;
}

string get_op_name(UnaryOpType::Enum op_type,std::optional<float> param0) {
   return is_parametrized_type(op_type) ? get_op_name_parameterized(op_type, param0.value()) : get_op_name_default(op_type);
}

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type,std::optional<float> param0) {
    std::string op_name = get_op_name(op_type,param0);
    add_defines_impl(eltwise_unary_kernel,op_type,op_name);
    return;
}


} // namespace eltwise_unary_op_utils

namespace tt {

namespace tt_metal {

void EltwiseUnary::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to eltwise unnary need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to eltwise unary need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::TILE_CL), "Inputs to eltwise unary must be tilized");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);
    TT_ASSERT((input_tensor_a.buffer()->buffer_type() == BufferType::DRAM));
}

std::vector<Shape> EltwiseUnary::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseUnary::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.layout() == Layout::TILE || input_tensor.layout() == Layout::TILE_CL);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), MemoryConfig{.interleaved = true});
}

operation::ProgramWithCallbacks EltwiseUnary::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    switch (parallelization_strategy){
        case UnaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_unary_multi_core(input_tensor, output_tensor, this->op_type,param);
            break;
        case UnaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            return eltwise_unary_single_core(input_tensor, output_tensor, this->op_type,param);
    }
}

operation::Hash EltwiseUnary::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}


UnaryOpParallelizationStrategy::Enum EltwiseUnary::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    if (num_tiles > 1) {
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return UnaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes EltwiseUnary::attributes() const {
    return {
        {"op_type", this->op_type},
        {"param", this->param},
    };
}

//unary op version tie
template<BcastOpMath::Enum OP>
Tensor tie_binop_to_unary(const Tensor& input_tensor, float value) {
  Tensor t_value = mk_scalar(value);
  return bcast(input_tensor,t_value,OP, BcastOpDim::HW);
}

Tensor div_unary(const Tensor& input_tensor, float value) {
    return tie_binop_to_unary<BcastOpMath::MUL>(input_tensor,1.0f/value);
}

Tensor div_unary(float value,const Tensor& input_tensor) {
    Tensor inv = tie_binop_to_unary<BcastOpMath::MUL>(input_tensor,value);
    return recip(inv);
}


Tensor mul_unary(const Tensor& input_tensor,float value) {
    return tie_binop_to_unary<BcastOpMath::MUL>(input_tensor,value);
}

Tensor sub_unary(const Tensor& input_tensor,float value) {
    return tie_binop_to_unary<BcastOpMath::SUB>(input_tensor,value);
}

Tensor sub_unary(float value, const Tensor& input_tensor) {
  return add_unary(value,neg(input_tensor));
}

Tensor add_unary(const Tensor& input_tensor,float value) {
    return tie_binop_to_unary<BcastOpMath::ADD>(input_tensor,value);
}

// symmetric
Tensor add_unary(float value, const Tensor& input_tensor) {
    return add_unary(input_tensor,value);
}

Tensor mul_unary(float value, const Tensor& input_tensor) {
    return mul_unary(input_tensor,value);
}

}  // namespace tt_metal

}  // namespace tt
