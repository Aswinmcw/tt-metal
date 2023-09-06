/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_tilize.h"
#endif

#include "debug_print.h"


namespace ckernel {

ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb = 16)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    #else
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    #endif

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_tilize_hw_configure_disaggregated(icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
    #else
    UNPACK(( llk_unpack_tilize_hw_configure_disaggregated<>(icb, block) ));
    UNPACK(( llk_unpack_tilize_init() ));
    #endif
}

ALWI void tilize_init_short(uint32_t icb, uint32_t block)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
    #else
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    UNPACK(( llk_unpack_tilize_init() ));
    #endif
}

ALWI void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb)
{

    UNPACK(( llk_unpack_tilize_block(icb, block) ));

    // UNPACK(( DPRINT << "WLKRHFJOLWSNDJN W" << ENDL() ));

    for (uint32_t t = 0; t < block; t++) {

        // Acquire dst
        MATH(( llk_math_wait_for_dest_available<SYNC>() ));
        PACK(( llk_packer_wait_for_math_done() ));

        // Datacopy
        MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0) ));
        PACK(( llk_pack<false, SYNC, false >(0, ocb)  ));

        // Release dest
        MATH(( llk_math_dest_section_done<SYNC>() ));
        PACK(( llk_pack_dest_section_done<SYNC>() ));
    }
    // PACK(( DPRINT << 'p' << ENDL() ));

}

ALWI void tilize_uninit()
{
    UNPACK(( llk_unpack_tilize_uninit() ));
}


}
