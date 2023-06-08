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

static Profiler op_profiler = Profiler();
static uint32_t call_count = 0;
static const string op_name = "bcast";
static const string perf_folder = "/tmp/tt_perf/ops/";
static string prepend_name = " ";

Tensor bcast_(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {
    const auto ashape = a.shape();
    const auto bshape = b.shape();
    u32 N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    u32 bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    u32 NC = N*C;
    u32 HW = H*W;

    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(a.volume() % TILE_HW == 0);

    TT_ASSERT((bN*bC == 1 || (bN == N && bC == C)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");
    // validate input dimensions
    if (bcast_dim == BcastOpDim::W)
        TT_ASSERT(H == bH && bW == TILE_WIDTH);
    if (bcast_dim == BcastOpDim::H)
        TT_ASSERT(W == bW && bH == TILE_HEIGHT);
    if (bcast_dim == BcastOpDim::HW)
        TT_ASSERT(bW == TILE_WIDTH && bH == TILE_HEIGHT);

    switch (bcast_op_utils::get_parallelization_strategy(a, bcast_dim)){
        case BcastOpParallelizationStrategy::MULTI_CORE_H:
            prepend_name = "MULTI_CORE_H";
            return bcast_multi_core_h(a, b, bcast_math, bcast_dim, call_count);
            break;
        case BcastOpParallelizationStrategy::MULTI_CORE_W:
            prepend_name = "MULTI_CORE_W";
            return bcast_multi_core_w(a, b, bcast_math, bcast_dim, call_count);
            break;
        case BcastOpParallelizationStrategy::MULTI_CORE_HW:
            prepend_name = "MULTI_CORE_HW";
            return bcast_multi_core_hw(a, b, bcast_math, bcast_dim, call_count);
            break;
        case BcastOpParallelizationStrategy::SINGLE_CORE:
        default:
            prepend_name = "SINGLE_CORE";
            return bcast_single_core(a, b, bcast_math, bcast_dim, call_count);
    }
}

Tensor _bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {

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

    if (bcast_dim == BcastOpDim::W)
        TT_ASSERT(a.shape()[2] == b.shape()[2]);
    else if (bcast_dim == BcastOpDim::H)
        TT_ASSERT(a.shape()[3] == b.shape()[3]);


    u32 N  = a.shape()[0], C  = a.shape()[1];
    u32 bN = b.shape()[0], bC = b.shape()[1];


    TT_ASSERT((bN*bC == 1 || (bN == N && bC == C)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");

    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape());
    auto b_pad_shape = AutoPad::pad_to_tile_shape(b.shape());
    auto out_shape = a.shape();

    auto no_pad_a = AutoPad::check_input_tensor_format(a, a_pad_shape);
    auto no_pad_b = AutoPad::check_input_tensor_format(b, b_pad_shape);
    if (no_pad_a && no_pad_b) {
        return bcast_(a, b, bcast_math, bcast_dim);
    } else if (no_pad_a) {
        auto output = bcast_(a, AutoPad::format_input_tensor(b, device, b_pad_shape, 0), bcast_math, bcast_dim);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else if (no_pad_b) {
        auto output = bcast_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), b, bcast_math, bcast_dim);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else {
        auto output = bcast_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), AutoPad::format_input_tensor(b, device, b_pad_shape, 0), bcast_math, bcast_dim);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    }
}

Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {
    op_profiler.markStart(op_name);
    op_profiler.setOutputDir(perf_folder + op_name);
    call_count ++;

    Tensor ret = _bcast(a,b,bcast_math,bcast_dim);

    op_profiler.markStop(op_name);
    op_profiler.dumpHostResults(to_string(call_count) + "-" + prepend_name);

    return ret;
}

}  // namespace tt_metal

}  // namespace tt
