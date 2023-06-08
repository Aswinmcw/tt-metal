#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

struct TransposeOpDim {
    enum Enum { WH = 0, HC = 1, CN = 2 };
    static const vector<Enum> all() { return { WH, HC, CN }; }
};

struct TransposeOpParallelizationStrategy {
    enum Enum { MULTI_CORE_WH = 0, MULTI_CORE_HC = 1, SINGLE_CORE = 2 };
    static const vector<Enum> all() { return { MULTI_CORE_WH, MULTI_CORE_HC, SINGLE_CORE }; }
};

// TODO: Accept parallelization
Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim=TransposeOpDim::WH);
Tensor _transpose(const Tensor &a, TransposeOpDim::Enum transpose_dim=TransposeOpDim::WH);
inline Tensor transpose(const Tensor &a) { return _transpose(a, TransposeOpDim::WH); }
inline Tensor transpose_wh(const Tensor &a) { return _transpose(a, TransposeOpDim::WH); }
inline Tensor transpose_hc(const Tensor &a) { return _transpose(a, TransposeOpDim::HC); }
inline Tensor transpose_cn(const Tensor &a) { return _transpose(a, TransposeOpDim::CN); }

Tensor transpose_single_core(const Tensor &a, TransposeOpDim::Enum transpose_dim);
Tensor transpose_wh_multi_core(const Tensor &a, uint32_t call_count = 0);
Tensor transpose_hc_multi_core(const Tensor &a, uint32_t call_count = 0);

}  // namespace tt_metal

}  // namespace tt

namespace transpose_op_utils {

using namespace tt::tt_metal;

TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, TransposeOpDim::Enum transpose_dim);

} // namespace transpose_op_utils
