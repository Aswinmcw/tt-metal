/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"
#include "src/firmware/riscv/common/risc_attribs.h"

#define SYNC SyncHalf

#define ALWI inline __attribute__((always_inline))

#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_reduce.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_common.h"
#include "llk_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_reduce.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_untilize.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif


namespace ckernel {

ALWI void rsqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void rsqrt_tile(uint32_t idst,bool fast_and_approx=true) {
  if (fast_and_approx) {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt<true, SyncHalf>(idst) ));
  } else {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt<false, SyncHalf>(idst) ));
  }
}

//for these files include eltwise_uanry/erf_erfc.h
//ALWI void erf_tile_init() {
//ALWI void erf_tile(uint32_t idst,bool fast_and_approx=true) {
//ALWI void erfc_tile_init() {
//ALWI void erfc_tile(uint32_t idst,bool fast_and_approx=true) {

ALWI void sigmoid_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>() )); // TODO(AP): move out init
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void sigmoid_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid<APPROX, SyncHalf>(idst) ));
}

ALWI void log_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_log_init<APPROX>() )); // TODO(AP): move out init
}

/**
 *  Please refer to documentation for log_tile.
 */
ALWI void log_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_log<APPROX, SyncHalf>(idst) ));
}

ALWI void log_with_base_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX>()));  // TODO(AP): move out init
}

/**
 *  Please refer to documentation for log_with_base_tile.
 */
ALWI void log_with_base_tile(uint32_t idst,uint32_t base_scale) {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base<APPROX, SyncHalf>(idst, base_scale)));
}

ALWI void tanh_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_tanh_init<APPROX>() )); // TODO(AP): move out init
}

ALWI void signbit_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_signbit_init<APPROX>() ));
}

ALWI void signbit_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_signbit<APPROX, SyncHalf>(idst) ));
}

ALWI void sin_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sin_init<APPROX>() ));
}

ALWI void cos_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_cos_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void tanh_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_tanh<APPROX, SyncHalf>(idst) ));
}

ALWI void sin_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sin<APPROX, SyncHalf>(idst) ));
}

ALWI void cos_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_cos<APPROX, SyncHalf>(idst) ));
}



//abs
ALWI void abs_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_abs<APPROX, SyncHalf>(idst) ));
}

ALWI void abs_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_abs_init<APPROX>() ));
}

//sign
ALWI void sign_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sign<APPROX, SyncHalf>(idst) ));
}

ALWI void sign_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sign_init<APPROX>() ));
}

//square
ALWI void square_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_square<APPROX, SyncHalf>(idst) ));
}

ALWI void square_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_square_init<APPROX>() ));
}

//compare to zero operators

ALWI void ltz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_ltz<APPROX, SyncHalf>(idst) ));
}

ALWI void ltz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_ltz_init<APPROX>() ));
}


ALWI void eqz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_eqz<APPROX,SyncHalf>(idst) ));
}

ALWI void eqz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_eqz_init<APPROX>() ));
}

ALWI void lez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_lez<APPROX, SyncHalf>(idst) ));
}

ALWI void lez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_lez_init<APPROX>() ));
}

ALWI void gtz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gtz<APPROX, SyncHalf>(idst) ));
}

ALWI void gtz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gtz_init<APPROX>() ));
}

ALWI void nez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_nez<APPROX, SyncHalf>(idst) ));
}

ALWI void nez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_nez_init<APPROX>() ));
}

ALWI void gez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gez<APPROX, SyncHalf>(idst) ));
}

ALWI void gez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gez_init<APPROX>() ));
}


//relu, relu-min, relu-max operators are implemented in
//compute_kernel_api/eltwise_unary/relu.h

//POWER : y = x^(const param0)
ALWI void power_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_power<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void power_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_power_init<APPROX>() ));
}

ALWI void get_next_op_info(tt::op_info_t& op_info)
{
    MATH(( llk_get_next_op_info(op_info) ));
    PACK(( llk_get_next_op_info(op_info) ));
    UNPACK(( llk_get_next_op_info(op_info) ));
}

ALWI void graph_interpreter_init() // TODO(AP): probably duplicated, remove
{
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));
    PACK(( llk_pack_init() ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(0,0) ));
}

//leaky_relu implemented in compute_kernel_api/eltwise_unary/relu.h
//elu implemented in same header as @leaky_relu

//exp2 : y = 2 ^ x  ==> [y = exp(x * log(2))]
ALWI void exp2_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_exp2<true, SyncHalf>(idst) ));
}

ALWI void exp2_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exp2_init<true>() ));
}

//heaviside : y = 0 if x < 0 , 1 if x > 0 , else value
ALWI void heaviside_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void heaviside_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>() ));
}

//expm1 : (exp(x) - 1)
ALWI void expm1_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_expm1<true, SyncHalf>(idst) ));
}

ALWI void expm1_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_expm1_init<true>() ));
}

//arcsine
ALWI void asin_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_asin<true, SyncHalf>(idst) ));
}

ALWI void asin_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_asin_init<true>() ));
}

//arctan
ALWI void atan_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_atan<true, SyncHalf>(idst) ));
}

ALWI void atan_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_atan_init<true>() ));
}

//arccosine
ALWI void acos_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_acos<true, SyncHalf>(idst) ));
}

ALWI void acos_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_acos_init<true>() ));
}
} // namespace ckernel
