// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// RELU MAX

template <bool APPROXIMATION_MODE>
void relu_max_init() {
    ;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void relu_max(uint uint_threshold)
{
    vFloat threshold = Converter::to_float(uint_threshold);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a > threshold) {
            a = threshold;
        }
        v_endif;
        v_if(a < 0.0f) {
            a = 0.0f;
        }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}


// RELU MIN

template <bool APPROXIMATION_MODE>
void relu_min_init() {
    ;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void relu_min(uint uint_threshold)
{
    vFloat threshold = Converter::to_float(uint_threshold);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a < threshold) {
            a = threshold;
        }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

//Leaky Relu

// LRELU = LEAKY RELU

template <bool APPROXIMATION_MODE>
void lrelu_init() {
    ;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_lrelu(uint slope)
{
    // SFPU microcode
    Converter c_slope;
    c_slope.u = slope;
    vFloat s = c_slope.f;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v *= s;
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}


}  // namespace sfpu
}  // namespace ckernel
