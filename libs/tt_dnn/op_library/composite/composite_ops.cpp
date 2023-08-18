#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"

#include "tt_numpy/functions.hpp"

namespace tt {

namespace tt_metal {

Tensor mk_zero_tensor_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    //Tensor zero_like = bcast(reference_tensor, , BcastOpMath::MUL, BcastOpDim::HW);
    static const Tensor zero = mk_tiled_scalar(0.0f);
    Tensor zero_like = bcast(reference_tensor, zero, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    return zero_like;
}

//TODO: enable zeroes(), ones() and eye() type functions on-device using this type of logic
template<typename T>
Tensor mk_filled_tensor_like(const Tensor& reference_tensor, T val, const MemoryConfig& output_mem_config) {
    Tensor k = mk_tiled_scalar(val);
    Tensor zero_like = mk_zero_tensor_like(reference_tensor, output_mem_config);
    Tensor result = bcast(zero_like, k, BcastOpMath::ADD,BcastOpDim::HW, output_mem_config);
    return result;
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor _softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t_a_plus_param = add_unary(a, param, output_mem_config);
    Tensor t1 = mul( ltz(t_a_plus_param, output_mem_config), t_a_plus_param, std::nullopt, output_mem_config);
    t_a_plus_param.deallocate();
    Tensor t_a_minus_param = sub_unary(a, param, output_mem_config);
    Tensor t2 = mul( gtz(t_a_minus_param, output_mem_config), t_a_minus_param, std::nullopt, output_mem_config);
    t_a_minus_param.deallocate();
    return add( t1, t2, std::nullopt, output_mem_config );
}
Tensor softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softshrink)(a, param, output_mem_config);
}

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor _hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t1 = mul( ltz(add_unary(a, param)), a, std::nullopt, output_mem_config );
    Tensor t2 = mul( gtz(sub_unary(a, param)), a, std::nullopt, output_mem_config );
    return add( t1, t2, std::nullopt, output_mem_config );
}
Tensor hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardshrink)(a, param, output_mem_config);
}

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor _softsign(const Tensor& a, const MemoryConfig& output_mem_config) {
    return mul(a, recip(add1(abs(a, output_mem_config), output_mem_config), output_mem_config), std::nullopt, output_mem_config);
}
Tensor softsign(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softsign)(a, output_mem_config);
}

// Function SILU (same as Swish)
// use activation Silu[x] = x*Sigmoid[x]
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
Tensor _silu(const Tensor& a, const MemoryConfig& output_mem_config) {
    //x / (1.0f + exp(-x))
    Tensor sigmoid_a = sigmoid(a, output_mem_config);
    Tensor silu_a = mul(a, sigmoid_a, std::nullopt, output_mem_config);
    return silu_a;
}
Tensor silu(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _silu)(a, output_mem_config);
}

Tensor _swish(const Tensor& a, const MemoryConfig& output_mem_config) {
    //x / (1.0f + exp(-x))
    return silu(a, output_mem_config);
}
Tensor swish(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _swish)(a, output_mem_config);
}

//log1p 1
//use transformation y = log(1.0 + x) by broadcast
Tensor _log1p(const Tensor& x, const MemoryConfig& output_mem_config) {
    Tensor x_1 = add1(x, output_mem_config);
    Tensor result_log1p = log(x_1, output_mem_config);
    return result_log1p;
}
Tensor log1p(const Tensor& x, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _log1p)(x, output_mem_config);
}

//softplus[x] = log[1 + exp[x]]
//use transformation y = log[1+exp[x]] by broadcast
Tensor _softplus(const Tensor& x, const MemoryConfig& output_mem_config) {
    Tensor exp_x = exp(x, output_mem_config);
    Tensor result_log1p = log1p(exp_x, output_mem_config);
    return result_log1p;
}
Tensor softplus(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softplus)(a, output_mem_config);
}

//tanhshrink(x) = x - tanh(x)
Tensor _tanhshrink(const Tensor& x, const MemoryConfig& output_mem_config) {
    Tensor tan_x = tanh(x, output_mem_config);
    Tensor result = sub(x, tan_x, std::nullopt, output_mem_config);
    return result;
}
Tensor tanhshrink(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _tanhshrink)(a, output_mem_config);
}

//mish[x] = x*tanh[softplus[x]]
//use transformation y = x*tanh[softplus[x]] by broadcast
//Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor _mish(const Tensor& x, const MemoryConfig& output_mem_config) {
    Tensor sp_x = softplus(x, output_mem_config);
    Tensor tanh_x = tanh(sp_x, output_mem_config);
    sp_x.deallocate();
    Tensor mish_x = mul(x, tanh_x, std::nullopt, output_mem_config);
    return mish_x;
}
Tensor mish(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mish)(a, output_mem_config);
}


// Theano defines this differently...
/**
 *
 *   alpha = 1.6732632423543772848170429916717
*    scale = 1.0507009873554804934193349852946
*    return scale * elu(x, alpha)
*
*/
// Function Selu - scaled exponential linear
//use transformation y = scale *(max(0,x)) + min(0,alpha * (exp(X)-1)) by broadcast
//Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor _selu(const Tensor& x, const float scale, const float alpha, const MemoryConfig& output_mem_config) {
    // term 2
    Tensor x_Exp = exp(x, output_mem_config);
    Tensor minus_one = mk_tiled_scalar(-1.0f);
    Tensor x_Exp_minus_1 = bcast(x_Exp, minus_one, BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
    x_Exp.deallocate();
    minus_one.deallocate();
    Tensor t_alpha = mk_tiled_scalar(alpha);
    Tensor result_t2_ = bcast(x_Exp_minus_1, t_alpha, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    x_Exp_minus_1.deallocate();
    t_alpha.deallocate();
    Tensor result_term2 = mul(gtz(result_t2_, output_mem_config), result_t2_, std::nullopt, output_mem_config);
    result_t2_.deallocate();

    // term 1
    Tensor t_scale = mk_tiled_scalar(scale);
    Tensor x_relu = relu(x, output_mem_config);
    Tensor result_term1 = bcast(x_relu, t_scale, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    t_scale.deallocate();
    x_relu.deallocate();
    Tensor result_selu = add(result_term1, result_term2, std::nullopt, output_mem_config);

    return result_selu;
}
Tensor selu(const Tensor& x,const float scale, const float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _selu)(x, scale, alpha, output_mem_config);
}

//ELU :
// Theano defins it as,
// return tensor.switch(x > 0, x, alpha * tensor.expm1(x))


// Function Clip
//use clip y = min( max( x, min_value), max_value) by broadcast
//Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor _clip(const Tensor& a,float low, float high, const MemoryConfig& output_mem_config) {
    const Tensor h_const = full_like(a, high);
    Tensor a_max = tt::tt_metal::min(a, h_const, output_mem_config);
    if ( low == 0.0f ) {
        return relu(a_max, output_mem_config);
    } else {
        const Tensor l_const = full_like(a, low);
        return tt::tt_metal::max(a_max, l_const, output_mem_config);
    }
}
Tensor clip(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _clip)(a, low, high, output_mem_config);
}

// Function Hard Sigmoid
//     Ref: https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
//
//     slope = tensor.constant(0.2, dtype=out_dtype)
//     shift = tensor.constant(0.5, dtype=out_dtype)
//
//     x1 = (x * slope) + shift
//     y = tensor.clip(x1, 0, 1)
//
// PyTorch version:
// hard sigmoid(x) = { x <= -3: 0, x >= +3: +3, x/6 + 0.5 otherwise}
Tensor _hardsigmoid(const Tensor& a, float scale, float shift, const MemoryConfig& output_mem_config) {
    Tensor a_mac = mac(a, scale, shift, output_mem_config); // multiply and add.
    Tensor a_clip = relu_max(a_mac, 1.0f, output_mem_config);
    return a_clip;
}
Tensor hardsigmoid(const Tensor& a, float scale, float shift, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardsigmoid)(a, scale, shift, output_mem_config);
}

// Function @hard_swish
//use transformation y = x * hardsigmoid( x ) by broadcast
//Ref: PyTorch
//hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor _hardswish(const Tensor& a, float scale, float shift, const MemoryConfig& output_mem_config) {
    Tensor a_sigmoid = hardsigmoid(a, scale, shift, output_mem_config);
    Tensor result_sq = mul(a_sigmoid, a, std::nullopt, output_mem_config);
    return result_sq;
}
Tensor hardswish(const Tensor& a, float scale, float shift, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardswish)(a, scale, shift, output_mem_config);
}

//compute polyval by Horner's rule
Tensor _polyval(const Tensor &input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    TT_ASSERT( coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
    if ( coeffs.size() == 1 ) {
        return  mk_filled_tensor_like( input_tensor, coeffs[0], output_mem_config );
    }

    Tensor result = bcast(input_tensor, mk_tiled_scalar(coeffs[0]), BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    for(int idx=1; idx < coeffs.size() - 1; idx++) {
        result = bcast(result, mk_tiled_scalar(coeffs[idx]), BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
        result = mul(input_tensor, result, std::nullopt, output_mem_config);
    }
    return bcast(result, mk_tiled_scalar(coeffs.back()), BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
}
Tensor polyval(const Tensor& input_tensor,std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polyval)(input_tensor, coeffs, output_mem_config);
}

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor & c, const MemoryConfig& output_mem_config) {
    bool a_is_scalar = a.volume() == 1;
    bool b_is_scalar = b.volume() == 1;
    bool c_is_scalar = c.volume() == 1;

    const auto dim = BcastOpDim::HW;
    if ( !a_is_scalar && !b_is_scalar && !c_is_scalar ) {
        //all tensors
        return add(mul(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if ( !a_is_scalar && !b_is_scalar && c_is_scalar ) {
        //a - tensor, b - tensor, c - is scalar
        return bcast(mul(a, b, std::nullopt, output_mem_config), c, BcastOpMath::ADD, dim, output_mem_config);
    } else if ( !a_is_scalar && b_is_scalar && !c_is_scalar ) {
        //a - tensor, b - scalar, c - is tensor
        return add(bcast(a, b, BcastOpMath::MUL, dim, output_mem_config), c, std::nullopt, output_mem_config);
    } else if ( !a_is_scalar && b_is_scalar && c_is_scalar ) {
        //a - tensor, b - scalar, c - is scalar
        return bcast(bcast(a, b, BcastOpMath::MUL, dim, output_mem_config), c, BcastOpMath::ADD, dim, output_mem_config);
    } else if ( a_is_scalar && !b_is_scalar && !c_is_scalar ) {
        //a - scalar, b - tensor, c - tensor
        return add(bcast(b, a, BcastOpMath::MUL, dim, output_mem_config), c, std::nullopt, output_mem_config);
    } else if ( a_is_scalar && !b_is_scalar && c_is_scalar ) {
        //a - scalar, b - tensor, c - is scalar
        return bcast(bcast(b, a, BcastOpMath::MUL, dim, output_mem_config), c, BcastOpMath::ADD, dim, output_mem_config);
    } else if ( a_is_scalar && b_is_scalar && !c_is_scalar ) {
        //a - scalar, b - scalar, c - is tensor
        return  bcast(c, mul(a, b, std::nullopt, output_mem_config), BcastOpMath::ADD, dim, output_mem_config);
    }

    // all scalars
    //a - scalar, b - scalar, c - is scalar
    TT_ASSERT( a_is_scalar && b_is_scalar && c_is_scalar);
    return add(mul(a, b), c);
}
Tensor mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config )
{
    return operation::decorate_as_composite(__func__, _mac)(a, b, c, output_mem_config);
}

Tensor _mac_overload(const Tensor& a, float b, float c, const MemoryConfig& output_mem_config) {
    Tensor t_b = mk_scalar(b);
    Tensor t_c = mk_scalar(c);
    return  mac(a, t_b, t_c, output_mem_config);
}
Tensor mac(const Tensor& input_a, float b, float c, const MemoryConfig& output_mem_config )
{
    return operation::decorate_as_composite(__func__, _mac_overload)(input_a, b, c, output_mem_config);
}

//min(a,b) = a - (a - b > 0 )*(a-b)
Tensor _min(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config)
{
    Tensor t_diff = sub(input_a, input_b, std::nullopt, output_mem_config);
    Tensor result = where(t_diff, input_b, input_a, output_mem_config);
    return result;
}
Tensor min(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _min)(input_a, input_b, output_mem_config);
}

//max(a,b) = a + (b - a > 0 )*(b-a)
Tensor _max(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config)
{
    Tensor t_diff = sub(input_b, input_a, std::nullopt, output_mem_config);
    Tensor result = where(t_diff, input_b, input_a, output_mem_config);
    return result;
}
Tensor max(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _max)(input_a, input_b, output_mem_config);
}

//sinh[x] = (exp[x] - exp[-x])/2
Tensor _sinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor e_pos_x = exp(input_a, output_mem_config);
    Tensor e_neg_x = exp(neg(input_a, output_mem_config), output_mem_config);
    Tensor nr_term = sub(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    return bcast(nr_term, mk_tiled_scalar(0.5f), BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
}
Tensor sinh(const Tensor &input_a, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sinh)(input_a, output_mem_config);
}

//cosh[x] = (exp[x] + exp[-x])/2
Tensor _cosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor e_pos_x = exp(input_a, output_mem_config);
    Tensor e_neg_x = exp(neg(input_a, output_mem_config), output_mem_config);
    Tensor nr_term = add(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    return bcast(nr_term, mk_tiled_scalar(0.5f), BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
}
Tensor cosh(const Tensor &input_a, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _cosh)(input_a, output_mem_config);
}

//atanh[x] = 0.5 * ln((1 + x) / (1 - x))
Tensor _atanh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor comp_result(input_a);
    {
        Tensor nr_term(input_a);
        {
        Tensor pos_x = add_unary(input_a, 1.0f, output_mem_config);
        Tensor neg_x = sub_unary(input_a, 1.0f, output_mem_config);
        nr_term  = log(mul(pos_x,recip(neg(neg_x, output_mem_config), output_mem_config), std::nullopt, output_mem_config), output_mem_config);
        }
        comp_result = mul_unary(nr_term, 0.5f, output_mem_config);
     }
    // Input is -1 > value > 1, output is nan
    // Input is -1 < value < 1, output is atanh(input)
    Tensor t_nan = mul_unary(comp_result, std::nanf(""), output_mem_config);
    Tensor abs_temp = sub_unary(abs(input_a, output_mem_config), 1.0f, output_mem_config);
    Tensor result = where(ltz(abs_temp, output_mem_config), comp_result, t_nan, output_mem_config);
    return result;
}
Tensor atanh(const Tensor &input_a, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _atanh)(input_a, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    Tensor t_value = mk_tiled_scalar(value);
    Tensor t_diff = sub(input_b, input_a, std::nullopt, output_mem_config);
    Tensor t_mul = bcast(t_diff, t_value, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    Tensor result =  add(input_a, t_mul, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lerp)(input_a, input_b, value, output_mem_config);
}

Tensor _atan2(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config)
{
    Tensor result(input_a);
    {
    Tensor atan_input = mul(abs(input_b, output_mem_config), recip(abs(input_a, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = atan(atan_input, output_mem_config);
    }
    Tensor res(result);
    {
    Tensor ib_gtz = gtz(input_b, output_mem_config);
    Tensor t_zero   = zeros_like(input_a, output_mem_config);
    Tensor ib_gt = gt(input_b, t_zero, std::nullopt, output_mem_config);
    Tensor ib_lt = lt(input_b, t_zero, std::nullopt, output_mem_config);
    Tensor pi_2 = add_unary(t_zero, M_PI_2, output_mem_config);
    Tensor neg_result = neg(result, output_mem_config);

    res = where(gt(input_a, t_zero, std::nullopt, output_mem_config),
    where(ib_gtz, result, neg_result, output_mem_config),
    where(lt(input_a, t_zero, std::nullopt, output_mem_config),
    where(ib_gt, add_unary(neg_result, M_PI, output_mem_config),
    where(ib_lt, sub_unary(result, M_PI, output_mem_config), add_unary(t_zero, M_PI, output_mem_config), output_mem_config), output_mem_config),
    where(ib_gt, pi_2, where(ib_lt, neg(pi_2, output_mem_config), t_zero, output_mem_config), output_mem_config), output_mem_config));
    }
    return res;
}
Tensor atan2(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _atan2)(input_a, input_b, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp_overload(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    Tensor t_diff = mul(sub(input_b, input_a, std::nullopt, output_mem_config), input_c, std::nullopt, output_mem_config);
    Tensor result = add(input_a, t_diff, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lerp_overload)(input_a, input_b, input_c, output_mem_config);
}

//ldexp(input,other)=input * (2^other)
Tensor _ldexp(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor result = mul(input_a, exp2(input_b, output_mem_config), std::nullopt, output_mem_config);
    return result;
}
Tensor ldexp(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _ldexp)(input_a, input_b, output_mem_config);
}

//subalpha(input,other,alpha)=input-alpha*other
Tensor _subalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const MemoryConfig& output_mem_config) {
    Tensor result = mac(input_b, neg(full_like(input_b, alpha), output_mem_config), input_a, output_mem_config);
    return result;
}
Tensor subalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _subalpha)(input_a, input_b, alpha, output_mem_config);
}

//addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, float value, const MemoryConfig& output_mem_config) {
    Tensor t_value = mk_tiled_scalar(value);
	Tensor t_mul = mul(input_b, input_c, std::nullopt, output_mem_config);
    Tensor t_factor = bcast(t_mul, t_value, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    t_mul.deallocate();
    t_value.deallocate();
    Tensor result = add(input_a, t_factor, std::nullopt, output_mem_config);
    return result;
}
Tensor addcmul(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcmul)(input_a, input_b,input_c, value, output_mem_config);
}

//addcdiv(input,tensor1,tensor2,value)=input+value×tensor1/tensor2
Tensor _addcdiv(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, float value, const MemoryConfig& output_mem_config) {
    Tensor t_value = mk_tiled_scalar(value);
	Tensor t_div = mul(input_b, recip(input_c, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_factor = bcast(t_div, t_value, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    t_div.deallocate();
    t_value.deallocate();
    Tensor result = add(input_a, t_factor, std::nullopt, output_mem_config);
    return result;
}
Tensor addcdiv(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcdiv)(input_a, input_b, input_c, value, output_mem_config);
}

//these ops need more polish - TBD
#if 0
//Function std
//compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
// Ref: torch.std
Tensor std(const Tensor& y);

// Function mean
//use transformation y = (y - mean(y))/std(y) by broadcast
// Ref: torch.mean
Tensor mean(const Tensor& y);

// Function normalize
//use transformation y = (y - mean(y))/std(y) by broadcast
Tensor normalize(const Tensor& a);


Tensor mean(const Tensor& y) {
    Tensor sum_y = sum(y);
    const float val = 1.0f/(float)y.volume();
    Tensor recip_size = mk_scalar(val);
    Tensor mean_y = bcast(sum_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
    return mean_y;
}


// Function normalize
//use transformation y = (y - mean(y))/std(y) by broadcast
Tensor normalize(const Tensor& y) {
    Tensor mean_y = mean(y);
    Tensor y_minus_mean_y = bcast(y,mean_y,BcastOpMath::SUB, BcastOpDim::HW);
    Tensor sqr_y_minus_mean_y = square(y_minus_mean_y);
    float scale = 1.0f/(float)y.volume();
    Tensor recip_size = mk_scalar(scale);
    Tensor var_y = bcast(sqr_y_minus_mean_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
    Tensor std_y = sqrt(var_y);
    Tensor recip_std_y = recip(std_y);
    Tensor z = bcast(y_minus_mean_y,recip_std_y,BcastOpMath::MUL, BcastOpDim::HW);
    return z;
}

Tensor std(const Tensor& y) {
    Tensor mean_y = mean(y);
    Tensor y_minus_mean_y = bcast(y,mean_y,BcastOpMath::SUB, BcastOpDim::HW);
    Tensor sqr_y_minus_mean_y = square(y_minus_mean_y);
    float scale = 1.0f/(float)y.volume();
    Tensor recip_size = mk_scalar(scale);
    Tensor var_y = bcast(sqr_y_minus_mean_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
    Tensor std_y = sqrt(var_y);
    return std_y;
}
#endif

//hypot(a,b) = sqrt[ a^2 + b^2 ]
Tensor _hypot(const Tensor &input_a, const Tensor &input_b, const MemoryConfig& output_mem_config) {
    Tensor a_sq = square(input_a, output_mem_config);
    Tensor b_sq = square(input_b, output_mem_config);
    Tensor c_sq = add(a_sq, b_sq, std::nullopt, output_mem_config);
    a_sq.deallocate();
    b_sq.deallocate();
    return  sqrt( c_sq, output_mem_config );
}
Tensor hypot(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _hypot)(input_a, input_b, output_mem_config);
}


//threshold(a,t,v) = (a <= t)*v + (a > t)*a
Tensor _threshold(const Tensor &input_a, float threshold, float value, const MemoryConfig& output_mem_config) {
    Tensor t_threshold = mk_tiled_scalar(threshold);
    Tensor t0 = bcast(input_a, t_threshold, BcastOpMath::SUB, BcastOpDim::HW, output_mem_config);
    t_threshold.deallocate();
    Tensor t_value = mk_tiled_scalar(value);
    Tensor t1 = bcast(lez(t0), t_value, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    t_value.deallocate();
    Tensor t2 = mul(gtz(t0, output_mem_config), input_a, std::nullopt, output_mem_config);
    return add(t1, t2, std::nullopt, output_mem_config);
}
Tensor threshold(const Tensor& input_a, float threshold, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _threshold)(input_a, threshold, value, output_mem_config);
}

//cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
//        = exp[ (1/3)*log[a] ]
Tensor _cbrt(const Tensor &input_a, const MemoryConfig& output_mem_config) {
    constexpr float scale = (float)(1.0/3.0);
    Tensor t_scale = mk_tiled_scalar(scale);
    Tensor t_ln_input = log(abs(input_a, output_mem_config), output_mem_config); //negative log is not useful here
    Tensor t1 = bcast(t_ln_input, t_scale, BcastOpMath::MUL, BcastOpDim::HW, output_mem_config);
    t_scale.deallocate();
    t_ln_input.deallocate();
    Tensor t2 = exp(t1, output_mem_config);
    t1.deallocate();
    Tensor t3 = mul(t2, sign(input_a, output_mem_config), std::nullopt, output_mem_config);
    return t3;
}
Tensor cbrt(const Tensor& input_a, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _cbrt)(input_a, output_mem_config);
}

//where - ternary operator y = (predicate) ? value_true : value_false; elementwise
//           y = (predicate >= 0)*value_true + (predicate < 0)*value_false
Tensor _where(const Tensor& predicate, const Tensor& value_true, const Tensor& value_false, const MemoryConfig& output_mem_config) {
    Tensor t2 = mul(gtz(predicate, output_mem_config), value_true, std::nullopt, output_mem_config);
    Tensor t1 = mul(lez(predicate, output_mem_config), value_false, std::nullopt, output_mem_config);
    return add(t2, t1, std::nullopt, output_mem_config);
}
Tensor where(const Tensor& predicate, const Tensor& value_true, const Tensor& value_false, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _where)(predicate, value_true, value_false, output_mem_config);
}

//on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    return mk_zero_tensor_like(reference_tensor, output_mem_config);
}

//on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    return mk_filled_tensor_like(reference_tensor, 1.0f, output_mem_config);
}

//on-device tensor creation with value like @reference_tensor
Tensor full_like(const Tensor& reference_tensor, float value, const MemoryConfig& output_mem_config) {
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config);
}

//hardtanh
Tensor _hardtanh(const Tensor& a,float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return  clip(a, low, high, output_mem_config);
}
Tensor hardtanh(const Tensor& a,float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardtanh)(a, low, high, output_mem_config);
}

//clamp
Tensor _clamp(const Tensor& a,float low, float high, const MemoryConfig& output_mem_config) {
    return  clip(a, low, high, output_mem_config);
}
Tensor clamp(const Tensor& a,float low, float high, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _clamp)(a, low, high, output_mem_config);
}

//on-device tensor creation 0s with shape
Tensor zeros(const Shape shape, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
    return tt::numpy::zeros(shape, DataType::BFLOAT16, layout, device, output_mem_config);
}

//on-device tensor creation 1s with shape
Tensor ones(const Shape shape, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
    return tt::numpy::ones(shape, DataType::BFLOAT16, layout, device, output_mem_config);
}

//on-device tensor creation with shape and filled with value
Tensor full(const Shape shape, float value, Layout layout, Device * device, const MemoryConfig& output_mem_config) {
    return tt::numpy::full(shape, value, DataType::BFLOAT16, layout, device, output_mem_config);
}

//on-device with increment
Tensor arange(int32_t start, int32_t end, int32_t step, Device * device, const MemoryConfig& output_mem_config) {
    return tt::numpy::arange<bfloat16>(start, end, step, Layout::ROW_MAJOR, device, output_mem_config);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    const Shape s_a = a.shape();
    const Shape s_b = b.shape();

    auto num_ones = [](const Shape& s) -> uint32_t {
        uint32_t num1s = 0;
        for(uint32_t idx = 0 ; idx < 4; idx++)
            num1s += (uint32_t)(s[idx] == 1);
        return num1s;
    };

    //check if 3 dimensions are 1
    TT_ASSERT( !(num_ones(s_a) < 3) , "3 dimensions are required to be 1 for use with outer product");
    TT_ASSERT( !(num_ones(s_b) < 3) , "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1 );
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1 );

    Tensor a_slim = a;
    Tensor b_slim = b;

    if (!skip_reshape_a) {
        a_slim = reshape (a, 1, 1, a.volume(), 1, output_mem_config);
    }
    if (!skip_reshape_b) {
        b_slim  = reshape (b, 1, 1, 1, b.volume(), output_mem_config);
    }

    return matmul(a_slim, b_slim, output_mem_config);
}
Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _outer)(a, b, output_mem_config);
}


}//namespace tt_metal

}//namespace tt
