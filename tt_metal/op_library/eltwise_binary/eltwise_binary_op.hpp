#pragma once

#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

struct BinaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

Tensor eltwise_binary (const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type, bool profile_device=false);
Tensor eltwise_binary_single_core (const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type, bool profile_device=false);
Tensor eltwise_binary_multi_core (const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type, bool profile_device=false);

inline Tensor add     (const Tensor &a, const Tensor &b, bool profile_device=false) { return eltwise_binary(a, b, BinaryOpType::ADD, profile_device); }
inline Tensor sub     (const Tensor &a, const Tensor &b, bool profile_device=false) { return eltwise_binary(a, b, BinaryOpType::SUB, profile_device); }
inline Tensor mul     (const Tensor &a, const Tensor &b, bool profile_device=false) { return eltwise_binary(a, b, BinaryOpType::MUL, profile_device); }

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

string get_op_name(BinaryOpType::Enum op_type);

void set_compute_kernel_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type);

BinaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b);

} // namespace eltwise_binary_op_utils
