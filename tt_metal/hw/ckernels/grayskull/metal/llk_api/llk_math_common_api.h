// // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #pragma once
// #include "ckernel.h"
// #include "ckernel_defs.h"
// #include "ckernel_globals.h"
// #include "ckernel_template.h"
// #include "cmath_common.h"
// #include "llk_defs.h"
// #include "llk_io.h"
// #include "llk_math_common.h"
// #include "llk_operands.h"
// #include "llk_param_structs.h"

// /*************************************************************************
//  * LLK MATH COMMON
//  *************************************************************************/

// template <DstSync Dst>
// inline void llk_math_wait_for_dest_available() {
//     _llk_math_wait_for_dest_available_<Dst>();
// }

// template <DstSync Dst = SyncFull, bool is_fp32_dest_acc_en = false /*not used*/>
// inline void llk_math_dest_section_done() {
//     _llk_math_dest_section_done_<Dst, is_fp32_dest_acc_en>();
// }

// template <DstSync Dst, bool is_fp32_dest_acc_en = false /*not used*/>
// inline void llk_math_pack_sync_init() {
//     _llk_math_pack_sync_init_<Dst, is_fp32_dest_acc_en>();
// }

// template <bool mail2math = true, bool mail2pack = true>
// inline void llk_math_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t *p_tile) {
//     _llk_math_get_tile_<mail2math, mail2pack>(tile_index, p_tile);
// }

// template <bool mail2math = true, bool mail2pack = true>
// inline void llk_math_release_tile(std::uint32_t operand) {
//     _llk_math_release_tile_<mail2math, mail2pack>();
// }

// inline void llk_math_debug_dump(std::uint8_t *data, std::uint32_t byte_size) { _llk_math_debug_dump_(data, byte_size); }

// inline void llk_math_debug_dump_seek(std::uint8_t offset) { _llk_math_debug_dump_seek_(offset); }

// //The following functions are only needed for WHB0, they call empty functions for GS
// inline void llk_math_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
//     _llk_math_reconfig_data_format_srca_();
// }

// inline void llk_math_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
//     _llk_math_reconfig_data_format_srcb_();
// }

// inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
//     _llk_math_reconfig_data_format_();
// }

// inline void llk_math_reconfig_data_format(
//     const std::uint32_t srca_old_operand,
//     const std::uint32_t srca_new_operand,
//     const std::uint32_t srcb_old_operand,
//     const std::uint32_t srcb_new_operand) {
//     _llk_math_reconfig_data_format_();
// }

// inline void llk_math_reconfig_data_format_srca(
//     const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
//     _llk_math_reconfig_data_format_srca_();
// }

// inline void llk_math_reconfig_data_format_srcb(
//     const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
//     _llk_math_reconfig_data_format_srcb_();
// }

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ckernel_defs.h"
#include "ckernel_include.h"
#include "cmath_common.h"
#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

#include "hostdevcommon/common_runtime_address_map.h"

using namespace ckernel::math;

template <DstSync Dst>
inline void llk_math_wait_for_dest_available() {
    // These liteweight functions for sync with packer imply
    // no mode change - entire epoch is either double buffer or single buffer
#ifdef PERF_DUMP
    if constexpr(MATH_PACK_DECOUPLE == 0) {
        math_dest_wait();
    }
#else
    math_dest_wait();
#endif
}

template <DstSync Dst = SyncFull>
inline void llk_math_dest_section_done() {
#ifdef PERF_DUMP
    if constexpr(MATH_PACK_DECOUPLE) {
        return;
    }
#endif
    set_math_semaphores();
    if constexpr (Dst == DstSync::SyncHalf) {
        dest_section_flip();
    } else if constexpr (Dst == DstSync::SyncTile16) {
        math_sync_tile_dst_index++;
        math_sync_tile_dst_index &= 0xF;
    } else if constexpr (Dst == DstSync::SyncTile2) {
        math_sync_tile_dst_index = math_sync_tile_dst_index + 8;
        math_sync_tile_dst_index &= 0xF;
    }
}

template <DstSync Dst>
inline void llk_math_pack_sync_init() {
#ifdef PERF_DUMP
    if constexpr(MATH_PACK_DECOUPLE) {
        return;
    }
#endif
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    };  // Wait for previous packs to finish before claiming all dest
    if constexpr (Dst == DstSync::SyncFull) {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    } else if constexpr (Dst == DstSync::SyncHalf) {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    } else if constexpr (Dst == DstSync::SyncTile2) {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
        math_sync_tile_dst_index = 0;
    } else {
        TTI_SEMINIT(15, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
        math_sync_tile_dst_index = 0;
    }
}

inline void llk_math_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    debug_dump(data, byte_size);
}
