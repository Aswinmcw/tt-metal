// // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #pragma once
// #include "llk_math_common_api.h"
// #include "llk_math_eltwise_binary.h"

// /*************************************************************************
//  * LLK ELTWISE BINARY
//  *************************************************************************/

// // Version with no operand
// template <
//     EltwiseBinaryType eltwise_binary_type,
//     BroadcastType src_b_bcast_type,
//     int NUM_FIDELITY_PHASES = 0,
//     EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
// inline void llk_math_eltwise_binary_init(const std::uint32_t transpose = 0, const std::uint32_t acc_to_dest = 0) {

//     _llk_math_eltwise_binary_init_<
//         eltwise_binary_type,
//         src_b_bcast_type,
//         NUM_FIDELITY_PHASES,
//         binary_reuse_dest>(
//         transpose, acc_to_dest);
// }

// // Version with operands
// //TODO: is this needed? operands arent used for anything
// template <
//     EltwiseBinaryType eltwise_binary_type,
//     BroadcastType src_b_bcast_type,
//     int NUM_FIDELITY_PHASES = 0,
//     EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
// inline void llk_math_eltwise_binary_init_with_operands(
//     const std::uint32_t operand_A,
//     const std::uint32_t operand_B,
//     const std::uint32_t transpose = 0,
//     const std::uint32_t acc_to_dest = 0) {

//     _llk_math_eltwise_binary_init_<
//         eltwise_binary_type,
//         src_b_bcast_type,
//         NUM_FIDELITY_PHASES,
//         binary_reuse_dest>(
//         transpose, acc_to_dest);
// }

// template <
//     EltwiseBinaryType eltwise_binary_type,
//     BroadcastType src_b_bcast_type,
//     DstSync Dst = DstSync::SyncFull,
//     int NUM_FIDELITY_PHASES = 0,
//     EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
//     bool is_fp32_dest_acc_en = false>
// inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = false) {
//     const std::uint32_t num_faces = 4;

//     _llk_math_eltwise_binary_<
//         eltwise_binary_type,
//         src_b_bcast_type,
//         Dst,
//         NUM_FIDELITY_PHASES,
//         binary_reuse_dest,
//         is_fp32_dest_acc_en>(num_faces, num_faces, dst_index, clear_fp32_dst_acc);
// }

// template <
//     EltwiseBinaryType eltwise_binary_type,
//     BroadcastType src_b_bcast_type,
//     DstSync Dst = DstSync::SyncFull,
//     int NUM_FIDELITY_PHASES = 0,
//     EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
//     bool is_fp32_dest_acc_en = false>
// inline void llk_math_eltwise_binary(
//     const std::uint32_t operand_A,
//     const std::uint32_t operand_B,
//     uint dst_index,
//     const bool clear_fp32_dst_acc = false) {
//     const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
//     const std::uint32_t num_faces = get_operand_num_faces(operand_id);

//     _llk_math_eltwise_binary_<
//         eltwise_binary_type,
//         src_b_bcast_type,
//         Dst,
//         NUM_FIDELITY_PHASES,
//         binary_reuse_dest,
//         is_fp32_dest_acc_en>(num_faces, num_faces, dst_index, clear_fp32_dst_acc);
// }


// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_param_structs.h"

#include "ckernel_include.h"
#include "ckernel_template.h"

#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

// local function declarations
inline void eltwise_binary_configure_addrmod();
inline void eltwise_binary_configure_mop(uint total_rows);

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst = DstSync::SyncFull,
    int NUM_FIDELITY_PHASES = 0,
    bool acc_to_dest =  false>
inline void llk_math_eltwise_binary(uint dst_index, bool clear_dest_acc=false) {
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);

        if constexpr (eltwise_binary_type == ELWMUL) {
            TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 0);
            TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 1);
            TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 2);
            TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 3);
        } else if constexpr (acc_to_dest == true) {
            if (clear_dest_acc) {
               TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 0);
               TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 1);
               TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 2);
               TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + 3);
            }
        }


    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB)) {
        if constexpr (src_b_bcast_type == BroadcastType::COL) {
            // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
            constexpr uint32_t outerloop = acc_to_dest ? 2 : 1;
            #pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                if constexpr (acc_to_dest) {
                    move_d2a_fixed_face(ADDR_MOD_1);
                }
                ckernel_template::run(instrn_buffer);
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            #pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                if constexpr (acc_to_dest) {
                    move_d2a_fixed_face(ADDR_MOD_1);
                }
                ckernel_template::run(instrn_buffer);
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        } else {
            constexpr uint32_t outerloop = acc_to_dest ? 4 : 1;
            #pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                if constexpr (acc_to_dest) {
                    move_d2a_fixed_face(ADDR_MOD_1);
                }
                ckernel_template::run(instrn_buffer);
            }
            // Manually clear B once mop is done for scaler bcast
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR) {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        if constexpr (src_b_bcast_type == BroadcastType::COL) {
            // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
            constexpr uint32_t outerloop = acc_to_dest ? 2 : 1;
            if constexpr (NUM_FIDELITY_PHASES > 0) {
                #pragma GCC unroll 0
                for (std::uint32_t n = 0; n < 2; n++) {  // N-num faces
                    if constexpr (acc_to_dest == true) {
                        move_d2a_fixed_face(ADDR_MOD_1);
                        TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, ((get_dest_buffer_base()>>4) + (dst_index<<2)) + n);
                    }
                    ckernel_template::run(instrn_buffer);
                }
            } else {
                #pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                    if constexpr (acc_to_dest == true) {
                        move_d2a_fixed_face(ADDR_MOD_1);
                        TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, ((get_dest_buffer_base()>>4) + (dst_index<<2)) + n);
                    }
                    ckernel_template::run(instrn_buffer);
                }
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            if constexpr (NUM_FIDELITY_PHASES > 0) {
                #pragma GCC unroll 0
                for (std::uint32_t n = 0; n < 2; n++) {  // N-num faces
                    if constexpr (acc_to_dest == true) {
                        move_d2a_fixed_face(ADDR_MOD_1);
                        TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, ((get_dest_buffer_base()>>4) + (dst_index<<2)) + 2 + n);
                    }
                    ckernel_template::run(instrn_buffer);
                }
            } else {
                #pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                    if constexpr (acc_to_dest == true) {
                        move_d2a_fixed_face(ADDR_MOD_1);
                        TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, ((get_dest_buffer_base()>>4) + (dst_index<<2)) + 2 + n);
                    }
                    ckernel_template::run(instrn_buffer);
                }
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        } else {
            // Row and no broadcasted behaves similarly
            constexpr uint32_t outerloop = acc_to_dest ? 4 : 1;
            if constexpr (NUM_FIDELITY_PHASES > 0) {
                #pragma GCC unroll 0
                for (std::uint32_t n = 0; n < 4; n++) {  // N-num faces
                    if constexpr (acc_to_dest == true) {
                        move_d2a_fixed_face(ADDR_MOD_1);
                        TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, ((get_dest_buffer_base()>>4) + (dst_index<<2)) + n);
                    }
                    ckernel_template::run(instrn_buffer);
                }
            } else {
                #pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                    if constexpr (acc_to_dest == true) {
                        move_d2a_fixed_face(ADDR_MOD_1);
                        TT_ZEROACC(p_zeroacc::CLR_16, ADDR_MOD_1, ((get_dest_buffer_base()>>4) + (dst_index<<2)) + n);
                    }
                    ckernel_template::run(instrn_buffer);
                }
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR) {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    } else {
        FWASSERT("Unsupported op!", false);
    }
    math::clear_dst_reg_addr();
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type>
inline void eltwise_binary_configure_addrmod() {
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        if constexpr (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) {
            addr_mod_t{
                .srca = {.incr = 4},
                .srcb = {.incr = 4},
                .dest = {.incr = 4},
            }
                .set(ADDR_MOD_0);
        } else if constexpr (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR) {
            addr_mod_t{
                .srca = {.incr = 4},
                .srcb = {.incr = 0},
                .dest = {.incr = 4},
            }
                .set(ADDR_MOD_0);
        }
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_t{
            .srca = {.incr = 0, .clr = 1},
            .srcb = {.incr = 0, .clr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = 1}}
            .set(ADDR_MOD_2);

        addr_mod_t{
            .srca = {.incr = 0, .clr = 1},
            .srcb = {.incr = 0, .clr = 1},
            .dest = {.incr = 4, .clr = 0, .cr = 0, .c_to_cr = 1},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_3);
    }
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, int NUM_FIDELITY_PHASES = 0, bool acc_to_dest = false>
inline void eltwise_binary_configure_mop() {
    const uint addr_mod = ADDR_MOD_0;
    uint innerloop = 16 >> 2;  // 4 rows per eltwise op at a time.
    uint outerloop = 4;
    auto broadcast_type = p_elwise::SRCB_NO_BCAST;
    if constexpr (bcast_type == BroadcastType::COL) {
        // The mop only runs for 2 outer loops and mop is called twice for col broadcast
        outerloop = 2;
        broadcast_type = p_elwise::SRCB_BCAST_COL;
    } else if constexpr (bcast_type == BroadcastType::ROW) {
        broadcast_type = p_elwise::SRCB_BCAST_ROW;
    } else if constexpr (bcast_type == BroadcastType::SCALAR) {
        broadcast_type = p_elwise::SRCB_BCAST_ALL;
    }

    if constexpr (acc_to_dest) {
        outerloop = 1;
    }

    // Scalar and Col broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    if constexpr (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) {
        if constexpr (eltwise_binary_type == ELWADD) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWSUB) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWSUB(0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWMUL) {
            ckernel_template tmp(
                NUM_FIDELITY_PHASES > 0 ? NUM_FIDELITY_PHASES : outerloop,
                innerloop,
                TT_OP_ELWMUL(0, broadcast_type, addr_mod, 0));
            if constexpr (NUM_FIDELITY_PHASES > 0) {
                tmp.set_last_inner_loop_instr(
                    TT_OP_ELWMUL(0, broadcast_type, ADDR_MOD_2, 0));  // Incr fidelity last inst of inner loop
                tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, broadcast_type, ADDR_MOD_3, 0));
            } else {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            }
            tmp.program(instrn_buffer);
        }
    } else {
        if constexpr (eltwise_binary_type == ELWADD) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWSUB) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWSUB(0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWMUL) {
            ckernel_template tmp(
                NUM_FIDELITY_PHASES > 0 ? NUM_FIDELITY_PHASES : outerloop,
                innerloop,
                TT_OP_ELWMUL(0, broadcast_type, addr_mod, 0));
            if constexpr (NUM_FIDELITY_PHASES > 0) {
                tmp.set_last_inner_loop_instr(
                    TT_OP_ELWMUL(0, broadcast_type, ADDR_MOD_2, 0));  // Incr fidelity last inst of inner loop
                tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_AB, broadcast_type, ADDR_MOD_3, 0));
            } else {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            }
            tmp.program(instrn_buffer);
        }
    }
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, int NUM_FIDELITY_PHASES = 0, bool acc_to_dest = false>
inline void llk_math_eltwise_binary_init() {
    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type>();

    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        eltwise_binary_configure_mop<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, acc_to_dest>();
    } else {
        FWASSERT("Unsupported op!", false);
    }

    math::reset_counters(p_setrwc::SET_ABD_F);
}
