// // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #pragma once
// #include "ckernel.h"
// #include "ckernel_defs.h"
// #include "ckernel_template.h"
// #include "cpack_common.h"
// #include "ckernel_globals.h"
// #include "circular_buffer.h"

#include "llk_io.h"
// #include "llk_defs.h"
#include "llk_outputs.h"
// #include "llk_param_structs.h"
// #include "llk_pack.h"
// #include "llk_pack_common.h"

// /*************************************************************************
// * LLK PACK
// *************************************************************************/

// template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
// inline void llk_pack_mop_config(const uint32_t output) {

//     _llk_pack_mop_config_<untilize, zero_output, FaceLayout, false>();
// }

// template <bool untilize = false, bool is_fp32_dest_acc_en = false /*not used*/>
// inline void llk_pack_hw_configure(const llk_pack_params_t *pack_params) {

//     const std::uint32_t output_id = get_output_id(pack_params->pack_output);
//     const std::uint32_t tile_size = cb_interface[output_id].fifo_page_size;

//     _llk_pack_hw_configure_<untilize>(
//         pack_src_format[output_id],
//         pack_dst_format[output_id],
//         tile_size,
//         pack_params->relu_config.val
//     );
// }

// template <bool untilize = false, bool is_fp32_dest_acc_en = false /*not used*/, ReluType relu_type = ReluType::NO_RELU, std::uint32_t relu_threshold = 0>
// inline void llk_pack_hw_configure_disaggregated(std::uint32_t pack_output) {
//     llk_pack_params_t llk_pack_params = {
//         .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold,}}};
//     llk_pack_hw_configure<untilize, is_fp32_dest_acc_en>(&llk_pack_params);
// }

// template <bool untilize = false, PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false /*not used*/>
// inline void llk_pack_reduce_hw_configure(const llk_pack_params_t *pack_params) {
//     const std::uint32_t output_id = get_output_id(pack_params->pack_output);
//     const std::uint32_t tile_size = cb_interface[output_id].fifo_page_size;

//     _llk_pack_reduce_hw_configure_<untilize, type, dim>(
//         pack_src_format[output_id],
//         pack_dst_format[output_id],
//         tile_size,
//         pack_params->relu_config.val
//     );
// }

// template <bool untilize = false, PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false, ReluType relu_type = ReluType::NO_RELU, std::uint32_t relu_threshold = 0>
// inline void llk_pack_reduce_hw_configure_disaggregated(std::uint32_t pack_output) {
//     llk_pack_params_t llk_pack_params = {
//         .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
//     llk_pack_reduce_hw_configure<untilize, type, dim, is_fp32_dest_acc_en>(&llk_pack_params);
// }

// template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
// inline void llk_pack_init(const std::uint32_t pack_output = 16) {

//     const std::uint32_t output_id = get_output_id(pack_output);

//     _llk_pack_init_<untilize, zero_output, FaceLayout>();
// }

// template <bool out_of_order_output, bool untilize>
// inline std::uint32_t get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index) {

//     std::uint16_t pack_tile_addr;
//     if constexpr (out_of_order_output) {
//         pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
//                          MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[output_id], (std::uint16_t)output_tile_index);
//     } else {
//         if constexpr (untilize) {
//             // TODO: uplift this option from BBE
//         } else {
//             pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr;
//             cb_interface[output_id].fifo_wr_tile_ptr += GET_L1_TILE_SIZE((std::uint8_t)pack_dst_format[output_id]);
//         }
//     }
//     return pack_tile_addr - 1;
// }

// template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false /* unused*/>
// inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
//     std::uint8_t output_id = get_output_id(output);

//     static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

//     std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(output_id, output_tile_index);

//     _llk_pack_<out_of_order_output, Dst, untilize, is_fp32_dest_acc_en>(
//         tile_index,
//         pack_dst_format[output_id],
//         pack_tile_addr
//     );
// }

// // template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false /* unused*/>
// // inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
// //     std::uint8_t output_id = get_output_id(output);

// //     static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

// //     std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(output_id, output_tile_index);

// //     if constexpr (Dst == DstSync::SyncTile16) {
// //         // Z-counter points to the next tile in dest
// //     } else if constexpr (Dst == DstSync::SyncTile2) {
// //         TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
// //         pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
// //     } else {
// //         TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
// //     }

// //     program_packer_untilized_destination(pack_tile_addr, pack_dst_format);

// //     mop_run(1, 1);

// //     if constexpr (untilize) {
// //         TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
// //         TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
// //     }
// // }

// /*************************************************************************
// * LLK PACK COMMON
// *************************************************************************/


// inline void llk_packer_wait_for_math_done() {
//     _llk_packer_wait_for_math_done_();
// }

// inline void llk_packer_set_math_semaphore() {
//     _llk_packer_set_math_semaphore_();  // Indicate that packer is done and header is written into L1
// }

// template <DstSync Dst, bool is_fp32_dest_acc_en = false>
// inline void llk_pack_dest_section_done() {
//     _llk_pack_dest_section_done_<Dst, is_fp32_dest_acc_en>();
// }

// template <DstSync Dst, DstTileFaceLayout FaceLayout, bool untilize = false>
// inline void llk_init_packer_dest_offset_registers(const std::uint32_t pack_output = 16) {
//     _llk_init_packer_dest_offset_registers_<Dst, FaceLayout, untilize>();
// }

// template <DstSync Dst, DstTileFaceLayout FaceLayout = RowMajor, bool untilize = false, bool is_fp32_dest_acc_en = false /*unused*/>
// inline void llk_pack_dest_init(const std::uint32_t pack_output = 16) {
//     _llk_pack_dest_init_<Dst, FaceLayout, untilize, is_fp32_dest_acc_en>();
// }

// template <bool mail2math=true, bool mail2pack=true>
// inline void llk_pack_get_tile(std::uint32_t output, std::uint32_t tile_index, std::uint32_t *p_tile) {
//     _llk_pack_get_tile_<mail2math, mail2pack>(tile_index, p_tile);
// }

// template <bool mail2math=true, bool mail2pack=true>
// inline void llk_pack_release_tile(std::uint32_t output) {
//     _llk_pack_release_tile_<mail2math, mail2pack>();
// }

// inline void llk_pack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
//     _llk_pack_debug_dump_(data, byte_size);
// }

// inline void llk_pack_debug_dump_seek(std::uint8_t offset) {
//     _llk_pack_debug_dump_seek_(offset);
// }

// template <bool is_fp32_dest_acc_en = false /*unused*/, bool is_tile_dim_reconfig_en = false /*unused*/, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor /*unused*/>
// inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
//     std::uint32_t output_id = get_output_id(new_output);

//     _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, is_tile_dim_reconfig_en, FaceLayout>(
//         pack_dst_format[output_id],
//         cb_interface[output_id].fifo_page_size
//     );
// }

// template <bool is_fp32_dest_acc_en = false /*unused*/, bool is_tile_dim_reconfig_en = false /*unused*/, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor /*unused*/>
// inline void llk_pack_reconfig_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
//     std::uint32_t old_output_id = get_output_id(old_output);
//     std::uint32_t new_output_id = get_output_id(new_output);

//     if((pack_dst_format[old_output_id] != pack_dst_format[new_output_id])
//        && (pack_dst_format[old_output_id] != (uint)DataFormat::Invalid)
//        && (pack_dst_format[new_output_id] != (uint)DataFormat::Invalid)) {
//         llk_pack_reconfig_data_format<is_fp32_dest_acc_en, is_tile_dim_reconfig_en, FaceLayout>(new_output);
//     }
// }

// TT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) {
//     _llk_pack_relu_config_(config);
// }

// inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) {
//     _llk_pack_reconfig_l1_acc_(enable);
// }

// template <bool untilize = false, ReduceDim dim>
// inline void llk_pack_reduce_mask_config() {
//     _llk_pack_reduce_mask_config_<untilize, dim>();
// }

// inline void llk_pack_reduce_mask_clear() {
//     _llk_pack_reduce_mask_clear_();
// }

// //TODO: review the following 2 functions
// template <ReduceDim dim, bool at_kernel_start = false, bool revert=false>
// inline void llk_pack_reduce_config_v2(uint32_t operand) {

//     const bool untilize = false;
//     if constexpr (at_kernel_start) {

//         const std::uint32_t output_id = get_output_id(operand);
//         const std::uint32_t tile_size = cb_interface[output_id].fifo_page_size;
//         const llk_relu_config_u relu_config = {.f = {.ApplyRelu = (std::uint32_t)ReluType::NO_RELU, .Threshold = 0,}};

//         _llk_pack_hw_configure_<untilize>(
//             pack_src_format[output_id],
//             pack_dst_format[output_id],
//             tile_size,
//             relu_config.val
//         );
//     }

//     if constexpr (revert) {
//         _llk_pack_reduce_mask_clear_();
//     } else {
//         _llk_pack_reduce_mask_config_<untilize, dim>();
//     }

// //The snippet below is the GS implementation of reduce_config_v2
//     // if constexpr (at_kernel_start)
//     //     configure_pack(get_output_id(icb_out), false);
//     // else {
//     //     TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
//     //     tensix_sync();
//     // }

//     // volatile uint *cfg = get_cfg_pointer();
//     // if constexpr (dim == ReduceDim::REDUCE_ROW) {
//     //     for (uint i = 0; i < 4; i++)
//     //         //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+i);
//     //         cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + i] = revert ? 0xFFFFffff : 0x1;
//     // } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
//     //     //TTI_WRCFG(revert ? 0xFFFFffff : 0x0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0);
//     //     //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1);
//     //     //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
//     //     cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = revert ? 0xFFFFffff : 0x0;
//     //     cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = revert ? 0xFFFFffff : 0x1;
//     //     cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = revert ? 0xF : 0x1;
//     // } else {
//     //     //TTI_WRCFG(revert ? 0xFFFFffff : 0x0,    p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0);
//     //     //TTI_WRCFG(revert ? 0xFFFFffff : 0xFFFF, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1);
//     //     cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = revert ? 0xFFFFffff : 0x0;
//     //     cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = revert ? 0xFFFFffff : 0x0000ffff;
//     //     cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = revert ? 0xF : 0x1;
//     // }
// }

// template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false>
// inline void llk_matmul_pack(std::uint32_t start_tile_index, std::uint32_t output, uint32_t ntiles, std::uint32_t output_tile_index = 0) {
//     std::uint8_t output_id = get_output_id(output);
//     const std::uint8_t OUTPUT_BASE_ID = (std::uint8_t) get_output_base_id();

//     static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

//     for (uint32_t tile_index=start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {

//         std::uint16_t pack_tile_addr;
//         if constexpr (out_of_order_output) {
//             pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
//                             MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID], (std::uint16_t)output_tile_index);
//         } else {
//             // in-order pack: 1) start with wr_ptr and then increment fifo_wr_tile_ptr tile by tile
//             // note: packer is programmed to automatically skip the tile header
//             // however, since there is no tile header we need to -1 the pack address (in terms of 16B words) to offset packer's +1
//             pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr - 1;
//             cb_interface[output_id].fifo_wr_tile_ptr += GET_L1_TILE_SIZE((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID]);
//         }

//         if constexpr (Dst == DstSync::SyncTile16) {
//             // Z-counter points to the next tile in dest
//         } else if constexpr (Dst == DstSync::SyncTile2) {
//             TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
//             pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
//         } else {
//             TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
//         }

//         // program_packer_untilized_destination(pack_tile_addr, (std::uint32_t)pack_dst_format[OUTPUT_BASE_ID]);
//         program_packer_destination(pack_tile_addr, (std::uint32_t)pack_dst_format[OUTPUT_BASE_ID]);

//         mop_run(1, 1);

//         if constexpr (untilize) {
//             TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
//             TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
//         }
//     }
// }

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "debug/fw_debug.h"
#include "cpack_common.h"
#include "llk_param_structs.h"

#include "hostdevcommon/common_runtime_address_map.h"


using namespace ckernel;
using namespace ckernel::packer;

// wait until math is done and has produced something to pack
inline void llk_packer_wait_for_math_done() {
    TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}

// Tell math that it can write again
inline void llk_packer_set_math_semaphore() {
    t6_semaphore_get(semaphore::MATH_PACK);  // Indicate that packer is done and header is written into L1
}

// Wait for all writes to complete in L1 (header + data)
// Tell math it can write again
// Clear dest
template <DstSync Dst>
inline void llk_pack_dest_section_done() {
    if constexpr ((Dst == DstSync::SyncTile16)) {
        llk_packer_set_math_semaphore();
    } else if constexpr (Dst == DstSync::SyncTile2) {
        // Tell math that it can write again
        TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::PACK);  // stall sem update until pack is done
        llk_packer_set_math_semaphore();
    } else {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);  // wait for pack to finish

        if constexpr (Dst == DstSync::SyncFull) {
            TT_ZEROACC(p_zeroacc::CLR_ALL, ADDR_MOD_1, 0);
        } else {
            TT_ZEROACC(p_zeroacc::CLR_HALF, ADDR_MOD_1, (dest_offset_id) % 2);
        }

        // Tell math that it can write again
        llk_packer_set_math_semaphore();

        if constexpr (Dst == DstSync::SyncHalf) {
            flip_packer_dest_offset_id();
            select_packer_dest_registers<Dst>();
        }
    }
}

template <DstSync Dst, DstTileFaceLayout FaceLayout, bool untilize = false>
inline void llk_init_packer_dest_offset_registers() {
    TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::PACK);  // wait for pack to finish
    if constexpr (untilize) {
       if constexpr (FaceLayout == ColMajor) {
          // Packer0 :  0,32,  1,33 ...  7, 39
          // Packer1 :  8,40,  9,41 ... 15, 47
          // Packer2 : 16,48, 17,49 ... 23, 55
          // Packer3 : 23,56, 24,57 ... 31, 63
          TT_SETDMAREG(0, 0x000 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
          TT_SETDMAREG(0, 0x000 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
          TT_SETDMAREG(0, 0x000 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
          TT_SETDMAREG(0, 0x000 + 0x18, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
          TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
          TT_SETDMAREG(0, 0x200 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
          TT_SETDMAREG(0, 0x200 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
          TT_SETDMAREG(0, 0x200 + 0x18, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       } else {
          // Packer0 :  0,16,  1,17 ...  7, 23
          // Packer1 :  8,24,  9,25 ... 15, 31
          // Packer2 : 32,48, 33,49 ... 39, 55
          // Packer3 : 40,56, 41,57 ... 47, 63
          TT_SETDMAREG(0, 0x000 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
          TT_SETDMAREG(0, 0x000 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
          TT_SETDMAREG(0, 0x000 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
          TT_SETDMAREG(0, 0x000 + 0x28, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
          TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
          TT_SETDMAREG(0, 0x200 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
          TT_SETDMAREG(0, 0x200 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
          TT_SETDMAREG(0, 0x200 + 0x28, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       }
    } else {
       if constexpr (FaceLayout == ColMajor) {
           TT_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
           TT_SETDMAREG(0, 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
           TT_SETDMAREG(0, 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
           TT_SETDMAREG(0, 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
           TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
           TT_SETDMAREG(0, 0x200 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
           TT_SETDMAREG(0, 0x200 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
           TT_SETDMAREG(0, 0x200 + 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       } else {  // Default to row major layout
           TT_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
           TT_SETDMAREG(0, 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
           TT_SETDMAREG(0, 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
           TT_SETDMAREG(0, 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
           TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
           TT_SETDMAREG(0, 0x200 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
           TT_SETDMAREG(0, 0x200 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
           TT_SETDMAREG(0, 0x200 + 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       }
    }
    select_packer_dest_registers<Dst>();
}
template <DstSync Dst, DstTileFaceLayout FaceLayout = RowMajor, bool untilize = false>
inline void llk_pack_dest_init() {
    tensix_sync();
    reset_dest_offset_id();
    llk_init_packer_dest_offset_registers<Dst, FaceLayout, untilize>();
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

inline void llk_pack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    debug_dump(data, byte_size);
}

inline void llk_pack_relu_config(std::uint32_t config) {
    ReluType mode = (config&0xf) == 0 ? ReluType::NO_RELU : ((config&0xf) == 3 ? ReluType::MAX_THRESHOLD_RELU : ReluType::MIN_THRESHOLD_RELU);
    uint32_t threshold = (config>>16) << STACC_RELU_ReluThreshold_SHAMT;
    TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0,((uint32_t)mode), 0, HI_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, threshold, 0, LO_16(p_gpr_pack::TMP1));
    TTI_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP1));
	TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TMP0,  p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1,  p_cfg::WRCFG_32b, STACC_RELU_ReluThreshold_ADDR32);
    TTI_NOP; TTI_NOP;
}

inline void llk_pack_reconfig_data_format(std::uint32_t new_operand) {
    reconfig_packer_data_format(new_operand);
   //reconfig_packer_data_format(pack_dst_format[get_output_id(new_operand)],cb_interface[get_output_id(new_operand)].fifo_page_size);
}


// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0



#include "llk_io_pack.h"
#include "llk_defs.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_pack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_mop_config() {


    addr_mod_pack_t{
        .y_src = {.incr = untilize ? 0 : 1},
        .y_dst = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 1, .clr = 0},
        .z_dst = {.incr = 1, .clr = 0},
    }
    .set(ADDR_MOD_1);


    if constexpr (untilize) {
       addr_mod_pack_t{
           .y_src = { .incr = 1, .clr = 0, .cr = 1  },
           .y_dst = { .incr = 1, .clr = 0, .cr = 0  },
       }.set(ADDR_MOD_2);
    }

    const uint MOP_INNER_LOOP = 16;
    const uint MOP_UNTILIZE_INNER_LOOP = FaceLayout == DstTileFaceLayout::ColMajor ? 8 : 4;
    const uint MOP_OUTER_LOOP = 1;
    const uint MOP_UNTILIZE_OUTER_LOOP = 8;
    const uint PACKCNT = 4;
    const uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    ckernel::ckernel_template tmp(
        untilize ? MOP_UNTILIZE_OUTER_LOOP : MOP_OUTER_LOOP, untilize ? MOP_UNTILIZE_INNER_LOOP : MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));


    if constexpr (!untilize) {
        tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));

        // there's no tile headers, so we don't need to do this
        // Write header to l1
        //tmp.set_end_op(TT_OP_STOREIND(
        //    1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));
    } else {
        tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
        tmp.set_end_op(TT_OP_PACR(ADDR_MOD_2, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
        tmp.set_last_outer_loop_instr(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
    }

    tmp.program(instrn_buffer);
}

template <bool untilize = false>
inline void llk_pack_hw_configure(const llk_pack_params_t *pack_params) {
    configure_pack(get_output_id(pack_params->pack_output), pack_params->relu_config.val);

    // configure_pack<untilize>(
    //     pack_src_format[get_output_id(pack_params->pack_output)],
    //     pack_dst_format[get_output_id(pack_params->pack_output)],
    //     cb_interface[get_output_id(pack_params->pack_output)].fifo_page_size,
    //     pack_params->relu_config.val);

    // std::uint32_t output = get_output_id(pack_params->pack_output);
    // if constexpr (untilize) {
    //     regfile[p_gpr_pack::ONE_MSG_RECEIVED] =
    //         ((1 * GET_L1_HEADERLESS_TILE_SIZE((uint)pack_dst_format[output])) << 12) |
    //         1; /*SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE=12*/
    // }
}

inline void llk_pack_reconfig_data_format(const std::uint32_t old_operand, const std::uint32_t new_operand) {
    std::uint32_t old_operand_id = get_output_id(old_operand);
    std::uint32_t new_operand_id = get_output_id(new_operand);

    if((pack_dst_format[old_operand_id] != pack_dst_format[new_operand_id])
       && (pack_dst_format[old_operand_id] != (uint)DataFormat::Invalid)
       && (pack_dst_format[new_operand_id] != (uint)DataFormat::Invalid)) {
        reconfig_packer_data_format(new_operand_id);
        //reconfig_packer_data_format(pack_dst_format[get_output_id(new_operand)],cb_interface[get_output_id(new_operand)].fifo_page_size);
    }
}

template <bool untilize = false, ReluType relu_type=ReluType::NO_RELU, std::uint32_t relu_threshold=0>
inline void llk_pack_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_params_t llk_pack_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
    llk_pack_hw_configure<untilize>(&llk_pack_params);
}

// FIXME: Remove once edge mask spec is defined
template <bool untilize = false, PoolType type, ReduceDim dim>
inline void llk_pack_reduce_hw_configure(const llk_pack_params_t *pack_params) {
      configure_pack(get_output_id(pack_params->pack_output), pack_params->relu_config.val);
    // configure_pack<untilize>(
    //     pack_src_format[get_output_id(pack_params->pack_output)],
    //     pack_dst_format[get_output_id(pack_params->pack_output)],
    //     cb_interface[get_output_id(pack_params->pack_output)].fifo_page_size,
    //     pack_params->relu_config.val);
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        for (uint i = 0; i < 4; i++) cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + i] = 0x00000001;
    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = 0x00000000;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0x00000001;
        cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x00000001;
    } else {
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = 0x00000000;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0x0000ffff;

        if constexpr (untilize) {
            cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x00000005;
        } else {
            cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x00000001;
        }
    }

    // if constexpr (untilize) {
    //     std::uint32_t output = get_output_id(pack_params->pack_output);
    //     regfile[p_gpr_pack::ONE_MSG_RECEIVED] =
    //         ((1 * GET_L1_HEADERLESS_TILE_SIZE((uint)pack_dst_format[output])) << 12) |
    //         1; /*SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE=12*/
    // }
}

template <bool untilize = false, PoolType type, ReduceDim dim, ReluType relu_type=ReluType::NO_RELU, std::uint32_t relu_threshold=0>
inline void llk_pack_reduce_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_params_t llk_pack_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
    llk_pack_reduce_hw_configure<untilize, type, dim>(&llk_pack_params);
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_init() {
    llk_pack_mop_config<untilize, zero_output, FaceLayout>();
}

template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false>
inline void llk_matmul_pack(std::uint32_t start_tile_index, std::uint32_t output, uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);
    const std::uint8_t OUTPUT_BASE_ID = (std::uint8_t) get_output_base_id();

    static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

    for (uint32_t tile_index=start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {

        std::uint16_t pack_tile_addr;
        if constexpr (out_of_order_output) {
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
                            MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID], (std::uint16_t)output_tile_index);
        } else {
            // in-order pack: 1) start with wr_ptr and then increment fifo_wr_tile_ptr tile by tile
            // note: packer is programmed to automatically skip the tile header
            // however, since there is no tile header we need to -1 the pack address (in terms of 16B words) to offset packer's +1
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr - 1;
            cb_interface[output_id].fifo_wr_tile_ptr += GET_L1_TILE_SIZE((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID]);
        }

        if constexpr (Dst == DstSync::SyncTile16) {
            // Z-counter points to the next tile in dest
        } else if constexpr (Dst == DstSync::SyncTile2) {
            TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
            pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
        } else {
            TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
        }

        program_packer_destination(pack_tile_addr, OUTPUT_BASE_ID);

        mop_run(1, 1);

        if constexpr (untilize) {
            TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
        }
    }
}

template <bool out_of_order_output = false, bool untilize = false>
inline std::uint16_t get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index) {

    std::uint16_t pack_tile_addr;
    if constexpr (out_of_order_output) {
        pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
                         MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[output_id], (std::uint16_t)output_tile_index);
    } else {
        if constexpr (untilize) {
            // TODO: uplift this option from BBE
        } else {
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr;
            cb_interface[output_id].fifo_wr_tile_ptr += GET_L1_TILE_SIZE((std::uint8_t)pack_dst_format[output_id]);
        }
    }
    return pack_tile_addr - 1;
}

template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false /* unused*/>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    // Todo: figure out tile dims based on output
    std::uint8_t output_id = get_output_id(output);

    static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

    std::uint16_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(output_id, output_tile_index);

    if constexpr (Dst == DstSync::SyncTile16) {
        // Z-counter points to the next tile in dest
    } else if constexpr (Dst == DstSync::SyncTile2) {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
        pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
    } else {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
    }

    program_packer_destination(pack_tile_addr, output_id);

    mop_run(1, 1);

    if constexpr (untilize) {
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
    }
}


template <ReduceDim dim, bool at_kernel_start = false, bool revert=false>
inline void llk_pack_reduce_config_v2(uint32_t icb_out) {

    if constexpr (at_kernel_start) {
        configure_pack(get_output_id(icb_out), false);
    } else {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
        tensix_sync();
    }

    volatile uint *cfg = get_cfg_pointer();
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        for (uint i = 0; i < 4; i++)
            //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+i);
            cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + i] = revert ? 0xFFFFffff : 0x1;
    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0);
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1);
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = revert ? 0xFFFFffff : 0x0;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = revert ? 0xFFFFffff : 0x1;
        cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = revert ? 0xF : 0x1;
    } else {
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x0,    p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0);
        //TTI_WRCFG(revert ? 0xFFFFffff : 0xFFFF, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1);
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = revert ? 0xFFFFffff : 0x0;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = revert ? 0xFFFFffff : 0x0000ffff;
        cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = revert ? 0xF : 0x1;
    }
}
