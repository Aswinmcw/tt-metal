#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void UpdateCache::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE, "Operands to update_cache need to be on device!");
    TT_ASSERT(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_ASSERT(input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr, "Operands to update_cache need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.layout() == Layout::TILE && cache_tensor.layout() == Layout::TILE), "Inputs to update_cache must be tilized");
    TT_ASSERT(input_tensor.dtype() == cache_tensor.dtype());
    TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B);

    TT_ASSERT(input_tensor.shape()[-1] == cache_tensor.shape()[-1]);
    TT_ASSERT(cache_tensor.shape()[1] == 1);
    TT_ASSERT(input_tensor.shape()[0] * input_tensor.shape()[1] == 1);

    if (this->op_type == UpdateCacheOpType::FILL) {
        TT_ASSERT(this->batch_idx < cache_tensor.shape()[0]);
        TT_ASSERT(input_tensor.shape()[-2] <= cache_tensor.shape()[-2]);
    } else if (this->op_type == UpdateCacheOpType::UPDATE) {
        TT_ASSERT(cache_tensor.shape()[0] == input_tensor.shape()[-2]);
    }
}

std::vector<Shape> UpdateCache::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

std::vector<Tensor> UpdateCache::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::ProgramWithCallbacks UpdateCache::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);

    switch(this->get_parallelization_strategy(input_tensors)) {
        case UpdateCacheOpParallelizationStrategy::MULTI_CORE:
            if (this->op_type == UpdateCacheOpType::FILL) {
                return fill_cache_multi_core(cache_tensor, input_tensor, this->batch_idx, this->update_idx);
            } else {
                TT_ASSERT(false, "Unsupported parallelization strategy for op");
            }
        case UpdateCacheOpParallelizationStrategy::SINGLE_CORE:
        default:
            if (this->op_type == UpdateCacheOpType::FILL) {
                return fill_cache_single_core(cache_tensor, input_tensor, this->batch_idx, this->update_idx);
            } else {
                return update_cache_single_core(cache_tensor, input_tensor, this->update_idx);
            }
    };
    return {};
}


UpdateCacheOpParallelizationStrategy UpdateCache::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(1);
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    if (this->op_type == UpdateCacheOpType::FILL) {
        if (num_tiles > 1) {
            return UpdateCacheOpParallelizationStrategy::MULTI_CORE;
        }
        else{
            return UpdateCacheOpParallelizationStrategy::SINGLE_CORE;
        }
    } else {
        return UpdateCacheOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes UpdateCache::attributes() const {
    return {
        {"batch_idx", this->batch_idx},
        {"update_idx", this->update_idx},
        {"op_type", this->op_type},
    };
}

}  // namespace tt_metal

}  // namespace tt
