/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
enum class MatmulParallelizationStrategy {
    MULTI_CORE = 0,
    MULTI_CORE_REUSE = 1,
    MULTI_CORE_REUSE_MCAST = 2,
    MULTI_CORE_REUSE_GENERALIZED = 3,
    MULTI_CORE_REUSE_MCAST_GENERALIZED = 4,
    MULTI_CORE_REUSE_PADDING = 5,
    MULTI_CORE_REUSE_MCAST_PADDING = 6,
    SINGLE_CORE = 7
};


/*
 * GENERAL MATMUL AND BMM
 */
operation::ProgramWithCallbacks matmul_single_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);

struct Matmul {
    bool bcast_batch;
    const MemoryConfig output_mem_config;
    const DataType output_dtype; // TODO: Uplift output_dtype as an option for general matmul/bmm

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    MatmulParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};


inline Tensor matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor_a.dtype() == input_tensor_b.dtype());
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_b.shape()[0]*input_tensor_b.shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=input_tensor_a.dtype()}, {input_tensor_a, input_tensor_b}).at(0);
}
inline Tensor bmm    (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor_a.dtype() == input_tensor_b.dtype());
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_a.shape()[1] == input_tensor_b.shape()[1] && input_tensor_a.shape()[0] == input_tensor_b.shape()[0]
        && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    return operation::run_with_autoformat(Matmul{.bcast_batch=false, .output_mem_config=mem_config, .output_dtype=input_tensor_a.dtype()}, {input_tensor_a, input_tensor_b}).at(0);
}

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool gelu);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Shape &ashape, const Shape &bshape, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);


/**
 * Bert large matmuls using operations::primary::matmul + program_config
 */
Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);

/**
 * Falcon matmuls using operations::primary::matmul + program_config
 */
Tensor falcon_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_dense_4h_to_h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_dense_h_to_4h_matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, bool fuse_gelu_activation = false, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_lm_head_matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);

/**
 * Generalized blocked matmul with support for tilize and untilize and mixed-prec
 */
struct BMMTilizeUntilize {
    const DataType out_dt_;
    const uint32_t in0_nblocks_h_, in0_nblocks_w_, in1_nblocks_w_;
    const uint32_t in0_block_ntiles_h_, in0_block_ntiles_w_, in1_block_ntiles_w_;
    const uint32_t out_subblock_ntiles_h_, out_subblock_ntiles_w_;
    const bool tilize_in0_, untilize_out_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    stl::reflection::Attributes attributes() const;
};

/**
 * Blocked Matmul, with support for tilize a and untilize output.
 * NOTE: Takes blocks and subblock information as arguments.
 */
Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b, DataType out_dt,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_in0, bool untilize_out);
operation::ProgramWithCallbacks bmm_single_core_tilize_untilize(
                                    const Tensor &in0, const Tensor &in1, DataType out_dt,
                                    uint32_t in0_height_nblocks, uint32_t in0_width_nblocks, uint32_t in1_width_nblocks,
                                    uint32_t in0_block_height_ntiles, uint32_t in0_block_width_ntiles, uint32_t in1_block_width_ntiles,
                                    uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                                    bool tilize_in0, bool untilize_out,
                                    Tensor &out);

}  // namespace tt_metal




namespace operations {

namespace primary {

using namespace tt_metal;

struct MatmulDefaultProgramConfig{
    tt::stl::reflection::Attributes attributes() const { return {}; };
};

struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;

    tt::stl::reflection::Attributes attributes() const;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<UnaryWithParam> fused_activation;

    tt::stl::reflection::Attributes attributes() const;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    bool fuse_gelu_activation;

    tt::stl::reflection::Attributes attributes() const;
};

using MatmulProgramConfig = std::variant<
    MatmulDefaultProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig
>;


struct Matmul {
    MatmulProgramConfig program_config;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;
};


inline Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const MatmulProgramConfig& program_config = MatmulDefaultProgramConfig{},
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype=std::nullopt
) {
    return operation::run(Matmul{program_config, mem_config, output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
}

inline Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b, std::optional<const Tensor> bias,
    const MatmulProgramConfig& program_config = MatmulDefaultProgramConfig{},
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype=std::nullopt
) {
    return operation::run(Matmul{program_config, mem_config, output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}

Tensor matmul_1d(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig> program_config = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

constexpr std::array<tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},
    {7, 1}, {1, 7},
    {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5},
    {2, 2}, {4, 1}, {1, 4},
    {3, 1}, {1, 3},
    {2, 1}, {1, 2},
    {1, 1},
}};

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool fuse_batch = false, bool fuse_gelu_activation = false);
}  // namespace bmm_op_utils
