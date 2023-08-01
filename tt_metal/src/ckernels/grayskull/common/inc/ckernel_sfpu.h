#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

sfpi_inline vInt sfpu_is_fp16_zero(const vFloat& v, uint exponent_size_8)
{
    if (exponent_size_8) {
        // fp16b
        return v == 0.0F;
    } else {
        // fp16a
        // if math data format is fp16, SFPU will convert 5 bit exp to 8 bit exp
        // in grayskull, this unconditionally adds bias value to exp (even for zero)
        vInt tmp = 0x3800; // loads {0, 8'd112, 10'b0}
        tmp += reinterpret<vInt>(v);

        return tmp == 0;
    }
}

sfpi_inline vFloat sfpu_exp(vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    vInt exp = exexp(val);
    v_if (exp >= 0) {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    vFloat tmp = val * vConst0p8369 + 0.8634F;
    val = val * tmp + vConst1;

    v_if (exp >= 0) {
        val = val * val;
        #pragma GCC unroll 0
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

template <bool save_reg, int max_iter = 3>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in)
{
    vInt orig_exp;

    if constexpr (max_iter == 1) {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        orig_exp = exexp(in);
    }

    // Force sign to 1 (make number negative)
    vFloat val = setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
    vFloat two;
    if (!save_reg) {
        two = 2.0f;
    }
    vFloat result = vConst1p4424 * (val * vConst1p4424 + (save_reg ? 2.0f : two));

    for (int s_iter = 0; s_iter < (max_iter-1); s_iter++) {
        result = result * (val * result + (save_reg ? 2.0f : two));
    }

    vInt new_exp = exexp(result);
    if constexpr (max_iter != 1) {
        orig_exp = exexp(dst_reg[0]);
    }

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0) {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

inline void init_dropout_seed(uint16_t p2)
{
    FWLOG1("calculate_dropout() -- input seed:%x", p2);

    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    FWLOG1("calculate_dropout() -- calculated seed:%x", per_tensix_input_seed);

    vInt result = l_reg[LRegs::LReg3];

    vInt tmp = vConstTileId << 13;
    vInt ptis = reinterpret<vInt>(vFloat(per_tensix_input_seed));
    result = ~(tmp & ptis) & (tmp | ptis);

    l_reg[LRegs::LReg3] = result;
}

template <bool APPROXIMATION_MODE>
inline void sfpu_init(SfpuType operation, uint param0 = 0)
{
    uint imm0;
    uint imm1;
    uint imm2;
    switch (operation) {
    case SfpuType::tanh:
    case SfpuType::tanh_derivative:
        imm0 = 0x1DFF; //0.90625*x
        imm1 = 0x481A; //0.09375*x + 0.8125
        imm2 = 0xFF00; //1
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::sigmoid_appx:
        imm0 = 0x3DFF;
        imm1 = 0x21D8;
        imm2 = 0xFF10;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::gelu:
        imm0 = 0x18FF;
        imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
        imm2 = 0xFF00;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::sqrt:
        if (APPROXIMATION_MODE) {
            TTI_SFPLOADI(2, 0, 127 << 7);
        }
        break;
    case SfpuType::sigmoid:
        break;
    case SfpuType::exponential:
        if constexpr(APPROXIMATION_MODE) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, p_exp::C23_73);
            TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
        }
        break;
    case SfpuType::elu:
    case SfpuType::gelu_derivative:
    case SfpuType::expm1:
    case SfpuType::exp2:
        if constexpr(APPROXIMATION_MODE) {
            TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
        }
        break;
    case SfpuType::dropout:
        init_dropout_seed(param0);
        break;
    default:
        // Should result in compile time error??
        break;
    }
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat calculate_exponential_body(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat val = in * vConst1p4424 + p_exp::C23_73;

        // Remove Exponent of 7 and bias the Mantissa to 127.
        // LREG2 already holds 2's complement value so we simply do REG2 + REG3
        vInt val_short = p_exp::ADJ_EXP + reinterpret<vInt>(val);

        // SHL to move integer bits to exponent
        val_short <<= 10 - p_exp::FRAC_BITS;
        out = reinterpret<vFloat>(val_short);

        // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
        // without using Relu in Packer to clamp -ve Infinity to 0.
        if constexpr (ZERO_NEGATIVE)
        {
            v_if (val_short < 0) {
                out = vConst0;
            }
            v_endif;
        }
    }
    else
    {
        // Force sign to 0 (make number positive)
        vFloat exp = sfpu_exp(setsgn(in, 0));

        // Load input value, to determine whether reciprocal needs to be run
        vFloat val = dst_reg[0];

        // store tentatively e^x
        // reciprocal function relies on reloading input
        dst_reg[0] = exp;

        v_if (val < 0) {
            dst_reg[0] = sfpu_reciprocal<true>(exp);
        }
        v_endif;
    }
    return out;
}

/*
template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN>
void calculate_cube(uint16_t exp_base_scale_factor = 0)
{
    for (int d = 0; d < ITERATIONS; d++)
    {

        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0); // load from dest
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_NOP; TTI_NOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_NOP; TTI_NOP;
        TTI_SFPSTORE(p_sfpu::LREG2, 0, 0); // Store from lreg[1] into dest registers
        TTI_INCRWC(0, 4, 0, 0);
    }
}
*/
template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat calculate_exponential_body_improved(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {

        vInt adj_exp;
        adj_exp = l_reg[LRegs::LReg2];

        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat val = in * vConst1p4424 + s2vFloat16b(p_exp::C23_73);

        // Remove Exponent of 7 and bias the Mantissa to 127.
        // LREG2 already holds 2's complement value so we simply do REG2 + REG3
        vInt val_short =  adj_exp + reinterpret<vInt>(val);

        // SHL to move integer bits to exponent
        val_short <<= 10 - p_exp::FRAC_BITS;
        out = reinterpret<vFloat>(val_short);

        // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
        // without using Relu in Packer to clamp -ve Infinity to 0.
        if constexpr (ZERO_NEGATIVE)
        {
            v_if (val_short < 0) {
                out = vConst0;
            }
            v_endif;
        }
        l_reg[LRegs::LReg2] = adj_exp;
    }
    else
    {
        // Force sign to 0 (make number positive)
        vFloat exp = sfpu_exp(setsgn(in, 0));

        // Load input value, to determine whether reciprocal needs to be run
        vFloat val = dst_reg[0];

        // store tentatively e^x
        // reciprocal function relies on reloading input
        out = exp;

        v_if (val < 0) {
            //dst_reg[0] = sfpu_reciprocal<true>(exp);
            out = sfpu_reciprocal<true>(exp);
        }
        v_endif;
    }
    return out;
}


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_expm1()
{
   // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        vFloat out = calculate_exponential_body_improved<true, true>(v);
        dst_reg[0] = out - 1.0f;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_asine_maclaurin_series(vFloat val)
{
    // input for [-1:1]
    // Mclauren series
    // arcsin(x) = x + [(1/2) *x^3/3] + [(1 * 3) / (2 * 4) * x^5 / 5] + [(1 * 3 * 5) / (2 * 4 * 6) * x^7 / 7 ] + ...
    // arcsin(x) ≈ x + (1/6) * x^3 + (3/40) * x^5 + (5/112) * x^7 + (35/1152) * x^9 + (63/2816) * x^11

    vFloat tmp = val;
    vFloat val_square = val * val;
    // x
    vFloat output = tmp;
    // (1/6) * x^3
    tmp = tmp * val_square;
    output += 0.166666666 * tmp;
    // (3/40) * x^5
    tmp = tmp * val_square;
    output +=  0.075 * tmp;

    //(5/112) * x^7
    tmp = tmp * val_square;
    output += 0.044642857 * tmp;

    // (35/1152) *x^9
    tmp = tmp * val_square;
    output += 0.03038194 * tmp;

    //(63/2816) * x^11
    tmp = tmp * val_square;
    output += 0.02237216 * tmp;

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_asin()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        dst_reg[0] = v;
        dst_reg++;
    }
}


#define PI_2 (1.570796326794)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_acos()
{
    // SFPU microcode
    // acos = (pi/2 - asin)
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        v = PI_2 - v;
        dst_reg[0] = v;
        dst_reg++;
    }
}


template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN, int ITERATIONS>
inline void calculate_exponential(int16_t exp_base_scale_factor = 0)
{
    vFloat c23_73;
    vInt adj_exp;

    if constexpr (APPROXIMATION_MODE)
    {
        c23_73 = l_reg[LRegs::LReg0];
        adj_exp = l_reg[LRegs::LReg2];
    }

    #pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr(SCALE_EN){
            val = val * s2vFloat16a(exp_base_scale_factor);
            dst_reg[0] = val;
        }
        if constexpr (APPROXIMATION_MODE)
        {
            // * by 1/ln2 and add convert to 7.3 FxP format
            val = val * vConst1p4424 + c23_73;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            // LREG2 already holds 2's complement value so we simply do REG2 + REG3
            vInt val_short = adj_exp + reinterpret<vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            dst_reg[0] = reinterpret<vFloat>(val_short);

            // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
            // without using Relu in Packer to clamp -ve Infinity to 0.
            if constexpr (ZERO_NEGATIVE)
            {
                v_if (val_short < 0) {
                    dst_reg[0] = vConst0;
                }
                v_endif;
            }
        }
        else
        {
            // Force sign to 0 (make number positive)
            val = sfpu_exp(setsgn(val, 0));

            vFloat orig = dst_reg[0];

            // Loaded by reciprocal
            dst_reg[0] = val;
            v_if (orig < 0) {
                dst_reg[0] = sfpu_reciprocal<false>(val);
            }
            v_endif;
        }

        dst_reg++;
    }

    if constexpr (APPROXIMATION_MODE)
    {
        l_reg[LRegs::LReg0] = c23_73;
        l_reg[LRegs::LReg2] = adj_exp;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_gelu_core(vFloat in)
{
    constexpr uint imm0 = 0x18FF;
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;

    // SFPU microcode:
    // result = (APPROX_MODE == 1)
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
    vFloat result;
    if constexpr (APPROXIMATION_MODE) {
        result = in;
    } else {
        // f = (0.044715*x^3 + x)
        result = in * in * in;
        result = result * 0.044715f + in;

        result *= 0.79788f;
    }

    result = lut(result, imm0, imm1, imm2);

    result = result * 0.5f + 0.5f;

    return result;
}

#define POLYVAL5(coef4,coef3,coef2,coef1,coef0,val) ( (((coef4*val + coef3)*val + coef2)*val + coef1)*val + coef0 )

inline
vFloat calculate_pos_cdf_appx(vFloat val) {
  //(0,2.5) interpolation polynomial coeffs  [ 0.0122792,  -0.05281024, -0.03048313,  0.41314081,  0.49866379]
  //(2.5,5) interpolation polynomial coeffs  [0.44656975,  0.58216001]

  // FIXME:
  // reuse LREG0-3 for storing coefficients and do product computation
  // const float coef_2dot5_to_5[4] = {-0.00221304f, -0.03253934f, -0.18027954f, -0.44656975f };
  // TTI_SFPLOADI(p_sfpu::LREG0, 0, 0xbb1108a6);
  // TTI_SFPLOADI(p_sfpu::LREG1, 0, 0xbd0547f9);
  // TTI_SFPLOADI(p_sfpu::LREG2, 0, 0xbe389b33);
  // TTI_SFPLOADI(p_sfpu::LREG2, 0, 0xbee4a4ca);

  vFloat result;
  v_if( val < 2.5f ) {
    result = POLYVAL5(0.0122792f,  -0.05281024f, -0.03048313f,  0.41314081f,  0.49866379f, val);
  } v_else {
    // assume v >= 2.5f - 5
    //result = POLYVAL5(result,-0.00221304f,  0.03253934f, -0.18027954f,  0.44656975f,  0.58216001f, val);
    //result = ((vFloat)l_reg[LRegs::LReg0])*val + (vFloat)l_reg[LRegs::LReg1];
    //result = result*val + (vFloat)l_reg[LRegs::LReg2];
    //result = result*val + (vFloat)l_reg[LRegs::LReg3];
    result = 0.44656975f*val + 0.58216001f;
  }
  v_endif;

  v_if(result > 1.0f) {
    result = 1.0f;
  }
  v_endif;
  return result;
}

// compute the approximate value of CDF of normal distribution
inline
vFloat calculate_cdf_appx(vFloat val,bool scaled = false) {
    vFloat result = 0.0f;
    vFloat val2 = 0.0;
    v_if ( val < 0.0f ) {
         val2 = -val;
    } v_else {
         val2 = val;
    }
    v_endif;

    result = calculate_pos_cdf_appx(val2);

    v_if ( val < 0.0f ) {
        result = 1.0f - result;
    }
    v_endif;

    if ( scaled ) {
      result *= val; //scale
    }
    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu_appx()
{
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;
    vUInt l0 = l_reg[LRegs::LReg0];

    // SFPU microcode
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vUInt l1;
        vUInt l2;
        vFloat result;

        if constexpr (APPROXIMATION_MODE)
        {
            l1 = imm1;
            l2 = imm2;
            result = val;
        } else {
            // f = (0.044715*x^3 + x)
            result = (val * val * val) * 0.044715f + val;

            // result = result * sqrt(2/pi)
            result *= 0.7969f;

            // Reload l1, l2 for lut
            l1 = imm1;
            l2 = imm2;
        }

        result = lut(result, l0, l1, l2);

        val = dst_reg[0];

        result = val * result + val;
        result *= 0.5f;

        dst_reg[0] = result;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_negative()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        dst_reg[0] = -val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu()
{

    if constexpr (APPROXIMATION_MODE) {
	calculate_gelu_appx<APPROXIMATION_MODE,ITERATIONS>();
    } else {
      constexpr bool scaled = true;
      // SFPU microcode
      for (int d = 0; d < ITERATIONS; d++)
	{
	  vFloat val = dst_reg[0];
	  vFloat result = calculate_cdf_appx(val,scaled);
	  dst_reg[0] = result;
	  dst_reg++;
	}
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS>
inline void calculate_rsqrt()
{

    for (int d = 0; d < ITERATIONS; d++)
    {

        vFloat in = dst_reg[0];
        vFloat result = sfpu_reciprocal<false>(in);

        for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
        {
            // y = y * (1.5 - 0.5 * x * y * y) Newton's method iteration.
            result = result * (1.5F - 0.5F  * dst_reg[0] * result * result);
        }

        dst_reg[0] = result;

        dst_reg++;

    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_add1()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        dst_reg[0] = 1.0f + val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sigmoid_appx()
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        val = lut(val, l0, l1, l2);

        dst_reg[0] = val + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tanh()
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        val = lut(val, l0, l1, l2);
        dst_reg[0] = val;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

// TODO: Implement using bitwise comparision
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        v_if (val <= -0.0f) {
            val = 1.0f;
        } v_elseif (val >= 0.0f) {
            val = 0.0f;
        }
        v_endif;

        dst_reg[0] = val;

       dst_reg++;
    }

}

template <bool APPROXIMATION_MODE>
inline void calculate_hardtanh(uint param0, uint param1, uint param2, int ITERATIONS)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    vFloat p0 = s2vFloat16(param0);
    vFloat p1 = s2vFloat16(param1);
    vFloat p2 = s2vFloat16(param2);
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        val += p0;// 12 bits
        v_if (val < 0.0f) {
            val = 0.0f;
        }
        v_endif;

        val += p1;// 12 bits
        v_if (val >= 0.0f) {
            val = 0.0f;
        }
        v_endif;

        val += p2;// 12 bits

        dst_reg[0] = val;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH, int ITERATIONS>
inline void calculate_tanh_derivative()
{
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    // tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (!WITH_PRECOMPUTED_TANH) {
            val = lut(val, l0, l1, l2);
        }

        val = val * val;
        val = vConst1 - val;
        dst_reg[0] = val;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu_derivative()
{
    // SFPU microcode:
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = val * val * vConstNeg0p5;

        // exp = e^(val) * 1/sqrt(2*pi)
        if constexpr(APPROXIMATION_MODE) {
            vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        } else {
            dst_reg[0] = result;
            vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        }
        result = calculate_gelu_core<APPROXIMATION_MODE>(val);

        dst_reg[0] = dst_reg[0] + result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_reciprocal()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat out = sfpu_reciprocal<false, APPROXIMATION_MODE ? 2 : 3>(in);

        // Reload to reduce register pressure
        v_if (dst_reg[0] < 0.0F) {
            // Invert sign on calculated value if CC=1 (number is negative)
            out = -out;
        }
        v_endif;

        dst_reg[0] = out;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS=2>
inline void calculate_sqrt()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            vUInt magic = l_reg[LRegs::LReg2];

            //sqrt initial approximation
            // adjust bias
            vUInt val_s = magic + reinterpret<vUInt>(val);

            // approximation of square root
            val_s >>= 1;
            dst_reg[0] = reinterpret<vFloat>(val_s);

            l_reg[LRegs::LReg2] = magic;
        }
        else
        {
            // Recip root method
            //// Init approx
            //u.i = SQRT_MAGIC_F - (u.i >> 1);
            vUInt magic = reinterpret<vUInt>(vFloat(s2vFloat16b(0x5f37)));
            vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));

            // Re-load to save a MOV
            val = dst_reg[0];

            //Reciproot iterations
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                //x*r*(1.5f - xhalf*r*r);
                approx = (approx * approx * val * vConstNeg0p5 + vConst1 + 0.5F) * approx;
            }

            dst_reg[0] = approx * val;
        }

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_dropout(uint prob, uint scale)
{
    // SFPU microcode

    FWLOG1("calculate_dropout() -- prob:%x", prob);
    FWLOG1("calculate_dropout() -- scale:%x", scale);

    vUInt rand = l_reg[LRegs::LReg3];
    vUInt mask = reinterpret<vUInt>(vFloat(s2vFloat16b(0xa94b)));

    #pragma GCC unroll 0
    for (int d=0; d<4; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        vUInt tmp = rand >> 3;
        v_if (tmp < vUInt(prob)) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        tmp = rand << 1;

        // Mask = 0x593CA -> 29e4d
        // Mask = 0xd295 -> a94b
        // PRNG SHL by one
        v_if (tmp < 0) {
            rand ^= mask;
        }
        v_endif;

        dst_reg++;
    }

    l_reg[LRegs::LReg3] = rand;
}

union Converter {
  float f;
  uint32_t u;
  static float to_float(uint32_t _v) {
    Converter c{};
    c.u = _v;
    return c.f;
  }
};

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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_elu(uint slope)
{
    // SFPU microcode
    constexpr bool zero_negative = true;
    Converter c_slope;
    c_slope.u = slope;
    vFloat s = c_slope.f;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
         vFloat v_exp = calculate_exponential_body_improved<true,zero_negative>(v);

         v = s*(v_exp - 1.0f);
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE,int ITERATIONS>
inline void calculate_power_iterative(uint exponent)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;
        for (uint i = 2; i < exponent; i++) {
            result *= in;
        }

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_power(uint exponent)
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = 1.0f;
	    constexpr uint SIZE = 4*sizeof(uint);//till power of 64
        vFloat b[SIZE] = {1.0f,};
        // kind of a LUT
        b[0] = in;
        #pragma GCC unroll 32
        for (uint i =  1; i < SIZE; i++) {
            b[i] = b[i-1]*b[i-1];
        }

        //reduce with product
        #pragma GCC unroll 32
        for (uint i = 0; i < SIZE; i++) {
            if ( exponent & (1<<i) )
                result *= b[i];
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_square()
{
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool HAS_BASE_SCALING>
sfpi_inline void calculate_log_body(const int log_base_scale_factor)
{
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    vFloat in = dst_reg[0];
    vFloat x = setexp(in, 127);    // set exp to exp bias (put in range of 1-2)

    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D':
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    vFloat a = s2vFloat16a(0.1058F);
    vFloat series_result = x * (x * (x * a + s2vFloat16a(-0.7122f)) + s2vFloat16a(2.0869)) + s2vFloat16a(-1.4753f);

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    // Extract exponent and calculate abs value.  Save sign into partial reg
    vInt exp = 0;
    v_if (in != 0.0F) {
        exp = exexp(in);
        v_if (exp < 0) {
            exp = sfpi::abs(exp);
            in = setsgn(in, 1);
        }
        v_endif;
    }
    v_endif;

    // Calculate exponent of the exponent value. Done by using LZ
    // Get leading zero.  If not zero, we do 19 + ~LZ to get exponent value (mathematically == 19 - LZ - 1)
    vInt new_exp = 0;
    v_if (exp != 0) {
        new_exp = lz(exp);
        new_exp = ~new_exp;
        new_exp += 19;
        v_if (new_exp >= 0) {
            new_exp += 127;
        }
        v_endif;
    }
    v_endif;

    vFloat result = setexp(in, new_exp);
    vInt shift = lz(exp) + 1;
    result = setman(result, shft(reinterpret<vUInt>(exp), shift));
    result = result * vConst0p6929 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= s2vFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (dst_reg[0] == 0.0F) { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    dst_reg[0] = result;

    dst_reg++;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void calculate_log(uint log_base_scale_factor)
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_log_body<HAS_BASE_SCALING>(log_base_scale_factor);
    }
}

sfpi_inline void calculate_comp_init_flag(bool check, vFloat& flag1, vFloat& flag2, float init)
{
    flag1 = init;
    if (check) {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS>
inline void calculate_comp(uint exponent_size_8)
{
   const vFloat zero = 0.0f;
   const vFloat one = 1.0f;
   for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        vFloat flag1, flag2;

	//a[i] == 0
	if constexpr(COMP_MODE == SfpuType::equal_zero) {
	    v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
	  }

	//a[i] != 0
	if constexpr(COMP_MODE == SfpuType::not_equal_zero) {
	    v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
	      v = zero;
	    } v_else {
	      v = one;
	    }
	    v_endif;
        }

	//a[i] < 0
	if constexpr(COMP_MODE == SfpuType::less_than_zero) {
	    v_if (v >= 0.0f) {
	      v = zero;
	    } v_else {
	      v = one;
	    }
	    v_endif;
        }

	//a[i] >= 0
	if constexpr(COMP_MODE == SfpuType::greater_than_equal_zero) {
	    v_if (v >= 0.0f) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
        }

	//a[i] > 0
	if constexpr(COMP_MODE == SfpuType::greater_than_zero) {
	    v_if (v > 0.0f) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
        }

	//a[i] <= 0
	if constexpr(COMP_MODE == SfpuType::less_than_equal_zero) {
	    v_if (v > 0.0f) {
	      v = zero;
	    } v_else {
	      v = one;
	    }
	    v_endif;
        }

	dst_reg[0] = v;
	dst_reg++;
    }
}


template <bool APPROXIMATION_MODE>
inline void calculate_clamp(uint param0, uint param1, uint param2, int ITERATIONS)
{
    // All params are in FP16 format
    // param0 = min
    // param1 = max

    //uint format = (param0 >> 16)&0x1;
    s2vFloat16::Format format = s2vFloat16::fp16a;

    // SFPU microcode
    vFloat min = s2vFloat16(param0, format);
    vFloat max = s2vFloat16(param1, format);
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        v_if (val < min) {
            val = s2vFloat16(param0, format);
        } v_elseif (val >= max) {
            val = s2vFloat16(param1, format);
        }
        v_endif;

        dst_reg[0] = val + s2vFloat16b(param2); // 12 bits

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_abs()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_exp2()
{
    constexpr bool zero_negative = true;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        // y = exp(x * log(2))
        // log(2) = 0.6931471805;
        v = v * 0.6931471805f;
        vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, zero_negative>(v);
        dst_reg[0] = exp;
        dst_reg++;
    }
}


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sign()
{
    // All params are in FP16 format
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
	vFloat result = vConst1;
        v_if (v < 0.0f) {
           result = vConstNeg1;
        } v_elseif(v > 0.0f) {
	  result = vConst1;
	} v_else {
	  result = vConst0;
        }
        v_endif;

	dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_max()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[16];
        v_if(a < b) {
            dst_reg[0] = b;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_min()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[16];
        v_if(a > b) {
            dst_reg[0] = b;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_sine_maclaurin_series(vFloat val)
{
    // Good for [-pi:pi]
    // Mclauren series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    vFloat tmp = val;
    // x
    vFloat output = tmp;
    // x^3/3!
    tmp = tmp*val*val;
    output += -0.166666666*tmp;
    // x^5/5!
    tmp = tmp*val*val;
    output +=  0.0083333333*tmp;

    // x^7/7!
    tmp = tmp*val*val;
    output += -0.0001984126*tmp;

    // x^9/9!
    tmp = tmp*val*val;
    output +=  0.0000027557*tmp;

    if constexpr (not APPROXIMATION_MODE) {
	// x^11/11!
        tmp = tmp*val*val;
        output += -0.00000002505*tmp;

	// x^13/13!
	tmp = tmp*val*val;
	output += 1.6059043836821613e-10*(tmp);
      }

    // Write out output
    return output;
}
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_cosine_maclaurin_series(vFloat val)
{
    // Good for [-pi:pi]
    // Mclauren series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    vFloat output = 1.0f;
    // x^2/2!
    vFloat tmp = val*val;
    output += -0.5*tmp;
    // x^4/4!
    tmp = tmp*val*val;
    output +=  0.0416666666*tmp;
    // x^6/6!
    tmp = tmp*val*val;
    output += -0.0013888888*tmp;

    // x^8/8!
    tmp = tmp*val*val;
    output +=  0.0000248015*tmp;

    // x^10/10!
    tmp = tmp*val*val;
    output += -0.0000002755*tmp;

    if constexpr (not APPROXIMATION_MODE) {

	// x^12/12!
	tmp = tmp*val*val;
	output += 2.08767569878681e-9*tmp;

	// x^14/14!
	tmp = tmp*val*val;
	output += -1.1470745597729725e-11*tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];

        // Assume v is bound [0:2pi]
        // phase shift [0:2pi] to [-pi:pi] and multiply result by -1
        v = v - 3.14159264f;
        v = sfpu_sine_maclaurin_series<APPROXIMATION_MODE>(v);

        // Use symmetrical properties of trig
        v *= -1;

        // Write Output
        dst_reg[0] = v;
        dst_reg++;
    }
}
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];

        // Assume v is bound [0:2pi]
        // phase shift [0:2pi] to [-pi:pi] and multiply result by -1
        v = v - 3.14159264f;
        v = sfpu_cosine_maclaurin_series<APPROXIMATION_MODE>(v);

        // Use symmetrical properties of trig
        v *= -1;

        // Write Output
        dst_reg[0] = v;
        dst_reg++;
    }
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


inline
vFloat sigmoid_piecewise_linear_positive(vFloat val) {
        vFloat result = 0.0f;
	v_if ( val >= +5.0f)  {
	  result = 1.0f;
	} v_elseif ( val > 1.0f && val < 5.0f ) {
	  result = POLYVAL5(0.00144462f, -0.01055479f, -0.01203685f,  0.24300185f,  0.50437757f,val);
	} v_else {
	  result = 0.229f*val + 0.5f; // linear appx as y = 0.229x + 0.5
	}
	v_endif;
	return result;
}

//sigmoid is anti-symmetric and offset by 1
//sigmoid[-x] = 1 - sigmoid[x]
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sigmoid()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = 0.0f;

        v_if ( val < 0.0f ) {
  	   val = -val;
        }
        v_endif;

	result = sigmoid_piecewise_linear_positive(val);

	val = dst_reg[0];
        v_if ( val < 0.0f ) {
            result = 1.0f - result;
        }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }

    return;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_heaviside(uint value)
{
    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat s = c_value.f;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v = 0.0f;
        }v_elseif (v > 0.0f) {
            v = 1.0f;
        }v_else {
            v = s;
        }
        v_endif;

       dst_reg[0] = v;

        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int SfpuType_PARAM = 0, int ITERATIONS = 4>
inline void calculate_sfpu(uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0)
{
    if constexpr (operation == SfpuType::exponential) {
	constexpr bool zero_negative = true;
        calculate_exponential<APPROXIMATION_MODE, zero_negative, false, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::exp_with_base) {
	constexpr bool zero_negative = true;
        calculate_exponential<APPROXIMATION_MODE, zero_negative, true, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::tanh) {
        calculate_tanh<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::signbit) {
        calculate_signbit<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::hardtanh) {
        calculate_hardtanh<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::gelu) {
	//param0 = true -> approximate fast mode
	//         false -> high precision mode
	if ( param0 ) {
	  calculate_gelu<true, ITERATIONS>();
	} else {
	  calculate_gelu<false, ITERATIONS>();
	}
    }
    else if constexpr (operation == SfpuType::rsqrt) {
	//param0 = true -> approximate fast mode
	//         false -> high precision mode
    // The algorithm uses Newton's method based on no.of iteration better approximation can be calculated
	if ( param0 ) {
	  calculate_rsqrt<true, ITERATIONS, 10>();
	} else {
	  calculate_rsqrt<false, ITERATIONS, 25>();
	}
    }
    else if constexpr (operation == SfpuType::reciprocal) {
        calculate_reciprocal<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sigmoid) {
        calculate_sigmoid<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sigmoid_appx) {
        calculate_sigmoid_appx<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sqrt) {
        calculate_sqrt<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::tanh_derivative) {
        calculate_tanh_derivative<APPROXIMATION_MODE, SfpuType_PARAM, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::lrelu) {
        calculate_lrelu<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::dropout) {
        calculate_dropout<APPROXIMATION_MODE, ITERATIONS>(param0, param1);
    }
    else if constexpr (operation == SfpuType::power) {
	if ( param0 <= 64 ) {
	  calculate_power<APPROXIMATION_MODE, ITERATIONS>(param0);
	} else {
	  calculate_power_iterative<APPROXIMATION_MODE, ITERATIONS>(param0);
	}
    }
    else if constexpr (operation == SfpuType::square) {
        calculate_square<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::log) {
        calculate_log<APPROXIMATION_MODE, false, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::log_with_base) {
        calculate_log<APPROXIMATION_MODE, true, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::gelu_derivative) {
        calculate_gelu_derivative<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr ((operation == SfpuType::equal_zero) ||
                       (operation == SfpuType::not_equal_zero) ||
                       (operation == SfpuType::less_than_zero) ||
                       (operation == SfpuType::greater_than_equal_zero) ||
                       (operation == SfpuType::less_than_equal_zero) ||
                       (operation == SfpuType::greater_than_zero)) {
        calculate_comp<APPROXIMATION_MODE, operation, ITERATIONS>(8); //BFLOAT16 - exp
    }
    else if constexpr (operation == SfpuType::clamp) {
        calculate_clamp<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::abs) {
        calculate_abs<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sign) {
        calculate_sign<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::max) {
        calculate_max<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::min) {
        calculate_min<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sine) {
        calculate_sine<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::cosine) {
        calculate_cosine<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::relu_min) {
        relu_min<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::relu_max) {
        relu_max<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::elu) {
        calculate_elu<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::exp2) {
        calculate_exp2<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::heaviside) {
        calculate_heaviside<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::expm1) {
        calculate_expm1<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::asin) {
        calculate_asin<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::acos) {
        calculate_acos<APPROXIMATION_MODE, ITERATIONS>();
    }
}

} // namespace sfpu
} // namespace ckernel
