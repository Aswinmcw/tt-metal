#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

#include "constants.hpp"

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

BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, BcastOpDim::Enum bcast_dim){
    uint32_t num_tiles = a.volume() / TILE_HW;
    uint32_t Ht = a.shape()[2] / TILE_HEIGHT;
    uint32_t Wt = a.shape()[3] / TILE_WIDTH;

    if(Ht > 1 and bcast_dim == BcastOpDim::H){
        return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(Wt > 1 and bcast_dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(num_tiles > 1 and bcast_dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        return BcastOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace bcast_op_utils


using namespace tt::tt_metal;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace tt_metal {


Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {

    Device * device;

    // Get the device
    if (a.on_host() && b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!a.on_host()){
        device = a.device();
    } else {
        device = b.device();
    }
    // Bring tensor to host if it isn't already, pad and convert layout, send to device
    auto input1 = AutoPad::format_input_tensor(a, device);
    auto input2 = AutoPad::format_input_tensor(b, device);

    const auto ashape = input1.shape();
    const auto bshape = input2.shape();
    u32 N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    u32 bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    u32 NC = N*C;
    u32 HW = H*W;

    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(input1.volume() % TILE_HW == 0);

    TT_ASSERT((bN*bC == 1 || (bN == N && bC == C)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");
    // validate input dimensions
    if (bcast_dim == BcastOpDim::W)
        TT_ASSERT(H == bH && bW == TILE_WIDTH);
    if (bcast_dim == BcastOpDim::H)
        TT_ASSERT(W == bW && bH == TILE_HEIGHT);
    if (bcast_dim == BcastOpDim::HW)
        TT_ASSERT(bW == TILE_WIDTH && bH == TILE_HEIGHT);

    Tensor output = Tensor({1, 1, 1, 1}, Initialize::ZEROS, DataType::BFLOAT16, Layout::ROW_MAJOR); // No Default Tensor Constructor, create dummy

    switch (bcast_op_utils::get_parallelization_strategy(input1, bcast_dim)){
        case BcastOpParallelizationStrategy::MULTI_CORE_H:
            output = bcast_multi_core_h(input1, input2, bcast_math, bcast_dim);
            break;
        case BcastOpParallelizationStrategy::MULTI_CORE_W:
            output = bcast_multi_core_w(input1, input2, bcast_math, bcast_dim);
            break;
        case BcastOpParallelizationStrategy::MULTI_CORE_HW:
            output = bcast_multi_core_hw(input1, input2, bcast_math, bcast_dim);
            break;
        case BcastOpParallelizationStrategy::SINGLE_CORE:
        default:
            output = bcast_single_core(input1, input2, bcast_math, bcast_dim);
    }

    // Convert tensor back to original
    output = AutoPad::format_output_tensor(a, output, a.shape(), device);

    return output;
}

}  // namespace tt_metal

}  // namespace tt
