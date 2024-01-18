// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

FORCE_INLINE void generate_bcast_scaler_w() {
    constexpr uint32_t cb_in_2 = tt::CB::c_in2;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(1);
    union { float f; uint32_t u; } u2; u2.u = get_arg_val<uint32_t>(2);
    cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[(k << 8) + j] = uint16_t(u.u>>16);
    cb_push_back(cb_in_2, 1);
}

FORCE_INLINE void generate_bcast_scaler_c() {
    constexpr uint32_t cb_in_4 = tt::CB::c_in4;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(0);
    cb_reserve_back(cb_in_4, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_4));

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[(k << 8) + j] = uint16_t(u.u>>16);
    cb_push_back(cb_in_4, 1);
}



FORCE_INLINE void generate_epsilon() {
    constexpr uint32_t eps_cb_id = tt::CB::c_in3;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(2);
    cb_reserve_back(eps_cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(eps_cb_id));

    for (int k = 0; k < 4; k+=2)
    for (int j = 0; j < 16; j++)
        ptr[(k << 8) + (j << 4)] = uint16_t(u.u>>16);
    cb_push_back(eps_cb_id, 1);
}

void kernel_main() {
    generate_bcast_scaler_w();
    generate_epsilon();
}
