#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
struct ConvOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, MULTI_CORE_REUSE = 1, MULTI_CORE_REUSE_MCAST = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE }; }
};

Tensor conv(const Tensor& A, const Tensor& B, vector<int> conv_params, bool untilize_out=true); // Tilizes a, untilizes b
Tensor conv_as_large_bmm_single_core_single_block(const Tensor& A, const Tensor& B, bool untilize_out, bool use_single_bank_reader); // Allows support for tilizing a, untilize b
std::pair<bool, string> conv_fits_on_single_core(vector<uint32_t> conv_act_shape, vector<uint32_t> conv_weight_shape, vector<int> conv_params);
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, string> compute_conv_op_block_info(uint32_t M, uint32_t K, uint32_t N);
}  // namespace tt_metal

}  // namespace tt
