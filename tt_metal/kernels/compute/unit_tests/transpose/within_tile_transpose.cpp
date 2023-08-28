#include <cstdint>

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    transpose_wh_init(tt::CB::c_in0);
    for(uint32_t b=0;b<per_core_tile_cnt;++b)
    {
        acquire_dst(tt::DstMode::Half);

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CB::c_in0, 1);
        cb_reserve_back(tt::CB::c_out0, 1);
        transpose_wh_tile(tt::CB::c_in0, 0, 0);
        pack_tile(0, tt::CB::c_out0);

        cb_pop_front(tt::CB::c_in0, 1);
        cb_push_back(tt::CB::c_out0, 1);

        release_dst(tt::DstMode::Half);
    }
}
}
