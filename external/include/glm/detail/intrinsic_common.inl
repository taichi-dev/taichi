///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2015 G-Truc Creation (www.g-truc.net)
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @ref core
/// @file glm/detail/intrinsic_common.inl
/// @date 2009-05-08 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

#if(GLM_COMPILER & GLM_COMPILER_VC)
#pragma warning(push)
#pragma warning(disable : 4510 4512 4610)
#endif

	union ieee754_QNAN
	{
		const float f;
		struct i
		{
			const unsigned int mantissa:23, exp:8, sign:1;
		};

		ieee754_QNAN() : f(0.0)/*, mantissa(0x7FFFFF), exp(0xFF), sign(0x0)*/ {}
	};

#if(GLM_COMPILER & GLM_COMPILER_VC)
#pragma warning(pop)
#endif

	static const __m128 GLM_VAR_USED zero = _mm_setzero_ps();
	static const __m128 GLM_VAR_USED one = _mm_set_ps1(1.0f);
	static const __m128 GLM_VAR_USED minus_one = _mm_set_ps1(-1.0f);
	static const __m128 GLM_VAR_USED two = _mm_set_ps1(2.0f);
	static const __m128 GLM_VAR_USED three = _mm_set_ps1(3.0f);
	static const __m128 GLM_VAR_USED pi = _mm_set_ps1(3.1415926535897932384626433832795f);
	static const __m128 GLM_VAR_USED hundred_eighty = _mm_set_ps1(180.f);
	static const __m128 GLM_VAR_USED pi_over_hundred_eighty = _mm_set_ps1(0.017453292519943295769236907684886f);
	static const __m128 GLM_VAR_USED hundred_eighty_over_pi = _mm_set_ps1(57.295779513082320876798154814105f);

	static const ieee754_QNAN absMask;
	static const __m128 GLM_VAR_USED abs4Mask = _mm_set_ps1(absMask.f);

	static const __m128 GLM_VAR_USED _epi32_sign_mask = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000)));
	//static const __m128 GLM_VAR_USED _epi32_inv_sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
	//static const __m128 GLM_VAR_USED _epi32_mant_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7F800000));
	//static const __m128 GLM_VAR_USED _epi32_inv_mant_mask = _mm_castsi128_ps(_mm_set1_epi32(0x807FFFFF));
	//static const __m128 GLM_VAR_USED _epi32_min_norm_pos = _mm_castsi128_ps(_mm_set1_epi32(0x00800000));
	static const __m128 GLM_VAR_USED _epi32_0 = _mm_set_ps1(0);
	static const __m128 GLM_VAR_USED _epi32_1 = _mm_set_ps1(1);
	static const __m128 GLM_VAR_USED _epi32_2 = _mm_set_ps1(2);
	static const __m128 GLM_VAR_USED _epi32_3 = _mm_set_ps1(3);
	static const __m128 GLM_VAR_USED _epi32_4 = _mm_set_ps1(4);
	static const __m128 GLM_VAR_USED _epi32_5 = _mm_set_ps1(5);
	static const __m128 GLM_VAR_USED _epi32_6 = _mm_set_ps1(6);
	static const __m128 GLM_VAR_USED _epi32_7 = _mm_set_ps1(7);
	static const __m128 GLM_VAR_USED _epi32_8 = _mm_set_ps1(8);
	static const __m128 GLM_VAR_USED _epi32_9 = _mm_set_ps1(9);
	static const __m128 GLM_VAR_USED _epi32_127 = _mm_set_ps1(127);
	//static const __m128 GLM_VAR_USED _epi32_ninf = _mm_castsi128_ps(_mm_set1_epi32(0xFF800000));
	//static const __m128 GLM_VAR_USED _epi32_pinf = _mm_castsi128_ps(_mm_set1_epi32(0x7F800000));

	static const __m128 GLM_VAR_USED _ps_1_3 = _mm_set_ps1(0.33333333333333333333333333333333f);
	static const __m128 GLM_VAR_USED _ps_0p5 = _mm_set_ps1(0.5f);
	static const __m128 GLM_VAR_USED _ps_1 = _mm_set_ps1(1.0f);
	static const __m128 GLM_VAR_USED _ps_m1 = _mm_set_ps1(-1.0f);
	static const __m128 GLM_VAR_USED _ps_2 = _mm_set_ps1(2.0f);
	static const __m128 GLM_VAR_USED _ps_3 = _mm_set_ps1(3.0f);
	static const __m128 GLM_VAR_USED _ps_127 = _mm_set_ps1(127.0f);
	static const __m128 GLM_VAR_USED _ps_255 = _mm_set_ps1(255.0f);
	static const __m128 GLM_VAR_USED _ps_2pow23 = _mm_set_ps1(8388608.0f);

	static const __m128 GLM_VAR_USED _ps_1_0_0_0 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
	static const __m128 GLM_VAR_USED _ps_0_1_0_0 = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
	static const __m128 GLM_VAR_USED _ps_0_0_1_0 = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
	static const __m128 GLM_VAR_USED _ps_0_0_0_1 = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);

	static const __m128 GLM_VAR_USED _ps_pi = _mm_set_ps1(3.1415926535897932384626433832795f);
	static const __m128 GLM_VAR_USED _ps_pi2 = _mm_set_ps1(6.283185307179586476925286766560f);
	static const __m128 GLM_VAR_USED _ps_2_pi = _mm_set_ps1(0.63661977236758134307553505349006f);
	static const __m128 GLM_VAR_USED _ps_pi_2 = _mm_set_ps1(1.5707963267948966192313216916398f);
	static const __m128 GLM_VAR_USED _ps_4_pi = _mm_set_ps1(1.2732395447351626861510701069801f);
	static const __m128 GLM_VAR_USED _ps_pi_4 = _mm_set_ps1(0.78539816339744830961566084581988f);

	static const __m128 GLM_VAR_USED _ps_sincos_p0 = _mm_set_ps1(0.15707963267948963959e1f);
	static const __m128 GLM_VAR_USED _ps_sincos_p1 = _mm_set_ps1(-0.64596409750621907082e0f);
	static const __m128 GLM_VAR_USED _ps_sincos_p2 = _mm_set_ps1(0.7969262624561800806e-1f);
	static const __m128 GLM_VAR_USED _ps_sincos_p3 = _mm_set_ps1(-0.468175413106023168e-2f);
	static const __m128 GLM_VAR_USED _ps_tan_p0 = _mm_set_ps1(-1.79565251976484877988e7f);
	static const __m128 GLM_VAR_USED _ps_tan_p1 = _mm_set_ps1(1.15351664838587416140e6f);
	static const __m128 GLM_VAR_USED _ps_tan_p2 = _mm_set_ps1(-1.30936939181383777646e4f);
	static const __m128 GLM_VAR_USED _ps_tan_q0 = _mm_set_ps1(-5.38695755929454629881e7f);
	static const __m128 GLM_VAR_USED _ps_tan_q1 = _mm_set_ps1(2.50083801823357915839e7f);
	static const __m128 GLM_VAR_USED _ps_tan_q2 = _mm_set_ps1(-1.32089234440210967447e6f);
	static const __m128 GLM_VAR_USED _ps_tan_q3 = _mm_set_ps1(1.36812963470692954678e4f);
	static const __m128 GLM_VAR_USED _ps_tan_poleval = _mm_set_ps1(3.68935e19f);
	static const __m128 GLM_VAR_USED _ps_atan_t0 = _mm_set_ps1(-0.91646118527267623468e-1f);
	static const __m128 GLM_VAR_USED _ps_atan_t1 = _mm_set_ps1(-0.13956945682312098640e1f);
	static const __m128 GLM_VAR_USED _ps_atan_t2 = _mm_set_ps1(-0.94393926122725531747e2f);
	static const __m128 GLM_VAR_USED _ps_atan_t3 = _mm_set_ps1(0.12888383034157279340e2f);
	static const __m128 GLM_VAR_USED _ps_atan_s0 = _mm_set_ps1(0.12797564625607904396e1f);
	static const __m128 GLM_VAR_USED _ps_atan_s1 = _mm_set_ps1(0.21972168858277355914e1f);
	static const __m128 GLM_VAR_USED _ps_atan_s2 = _mm_set_ps1(0.68193064729268275701e1f);
	static const __m128 GLM_VAR_USED _ps_atan_s3 = _mm_set_ps1(0.28205206687035841409e2f);

	static const __m128 GLM_VAR_USED _ps_exp_hi = _mm_set_ps1(88.3762626647949f);
	static const __m128 GLM_VAR_USED _ps_exp_lo = _mm_set_ps1(-88.3762626647949f);
	static const __m128 GLM_VAR_USED _ps_exp_rln2 = _mm_set_ps1(1.4426950408889634073599f);
	static const __m128 GLM_VAR_USED _ps_exp_p0 = _mm_set_ps1(1.26177193074810590878e-4f);
	static const __m128 GLM_VAR_USED _ps_exp_p1 = _mm_set_ps1(3.02994407707441961300e-2f);
	static const __m128 GLM_VAR_USED _ps_exp_q0 = _mm_set_ps1(3.00198505138664455042e-6f);
	static const __m128 GLM_VAR_USED _ps_exp_q1 = _mm_set_ps1(2.52448340349684104192e-3f);
	static const __m128 GLM_VAR_USED _ps_exp_q2 = _mm_set_ps1(2.27265548208155028766e-1f);
	static const __m128 GLM_VAR_USED _ps_exp_q3 = _mm_set_ps1(2.00000000000000000009e0f);
	static const __m128 GLM_VAR_USED _ps_exp_c1 = _mm_set_ps1(6.93145751953125e-1f);
	static const __m128 GLM_VAR_USED _ps_exp_c2 = _mm_set_ps1(1.42860682030941723212e-6f);
	static const __m128 GLM_VAR_USED _ps_exp2_hi = _mm_set_ps1(127.4999961853f);
	static const __m128 GLM_VAR_USED _ps_exp2_lo = _mm_set_ps1(-127.4999961853f);
	static const __m128 GLM_VAR_USED _ps_exp2_p0 = _mm_set_ps1(2.30933477057345225087e-2f);
	static const __m128 GLM_VAR_USED _ps_exp2_p1 = _mm_set_ps1(2.02020656693165307700e1f);
	static const __m128 GLM_VAR_USED _ps_exp2_p2 = _mm_set_ps1(1.51390680115615096133e3f);
	static const __m128 GLM_VAR_USED _ps_exp2_q0 = _mm_set_ps1(2.33184211722314911771e2f);
	static const __m128 GLM_VAR_USED _ps_exp2_q1 = _mm_set_ps1(4.36821166879210612817e3f);
	static const __m128 GLM_VAR_USED _ps_log_p0 = _mm_set_ps1(-7.89580278884799154124e-1f);
	static const __m128 GLM_VAR_USED _ps_log_p1 = _mm_set_ps1(1.63866645699558079767e1f);
	static const __m128 GLM_VAR_USED _ps_log_p2 = _mm_set_ps1(-6.41409952958715622951e1f);
	static const __m128 GLM_VAR_USED _ps_log_q0 = _mm_set_ps1(-3.56722798256324312549e1f);
	static const __m128 GLM_VAR_USED _ps_log_q1 = _mm_set_ps1(3.12093766372244180303e2f);
	static const __m128 GLM_VAR_USED _ps_log_q2 = _mm_set_ps1(-7.69691943550460008604e2f);
	static const __m128 GLM_VAR_USED _ps_log_c0 = _mm_set_ps1(0.693147180559945f);
	static const __m128 GLM_VAR_USED _ps_log2_c0 = _mm_set_ps1(1.44269504088896340735992f);

GLM_FUNC_QUALIFIER __m128 sse_abs_ps(__m128 x)
{
	return _mm_and_ps(glm::detail::abs4Mask, x);
} 

GLM_FUNC_QUALIFIER __m128 sse_sgn_ps(__m128 x)
{
	__m128 Neg = _mm_set1_ps(-1.0f);
	__m128 Pos = _mm_set1_ps(1.0f);

	__m128 Cmp0 = _mm_cmplt_ps(x, zero);
	__m128 Cmp1 = _mm_cmpgt_ps(x, zero);

	__m128 And0 = _mm_and_ps(Cmp0, Neg);
	__m128 And1 = _mm_and_ps(Cmp1, Pos);

	return _mm_or_ps(And0, And1);
}

//floor
GLM_FUNC_QUALIFIER __m128 sse_flr_ps(__m128 x)
{
	__m128 rnd0 = sse_rnd_ps(x);
	__m128 cmp0 = _mm_cmplt_ps(x, rnd0);
	__m128 and0 = _mm_and_ps(cmp0, glm::detail::_ps_1);
	__m128 sub0 = _mm_sub_ps(rnd0, and0);
	return sub0;
}

//trunc
/*
GLM_FUNC_QUALIFIER __m128 _mm_trc_ps(__m128 v)
{
	return __m128();
}
*/
//round
GLM_FUNC_QUALIFIER __m128 sse_rnd_ps(__m128 x)
{
	__m128 and0 = _mm_and_ps(glm::detail::_epi32_sign_mask, x);
	__m128 or0 = _mm_or_ps(and0, glm::detail::_ps_2pow23);
	__m128 add0 = _mm_add_ps(x, or0);
	__m128 sub0 = _mm_sub_ps(add0, or0);
	return sub0;
}

//roundEven
GLM_FUNC_QUALIFIER __m128 sse_rde_ps(__m128 x)
{
	__m128 and0 = _mm_and_ps(glm::detail::_epi32_sign_mask, x);
	__m128 or0 = _mm_or_ps(and0, glm::detail::_ps_2pow23);
	__m128 add0 = _mm_add_ps(x, or0);
	__m128 sub0 = _mm_sub_ps(add0, or0);
	return sub0;
}

GLM_FUNC_QUALIFIER __m128 sse_ceil_ps(__m128 x)
{
	__m128 rnd0 = sse_rnd_ps(x);
	__m128 cmp0 = _mm_cmpgt_ps(x, rnd0);
	__m128 and0 = _mm_and_ps(cmp0, glm::detail::_ps_1);
	__m128 add0 = _mm_add_ps(rnd0, and0);
	return add0;
}

GLM_FUNC_QUALIFIER __m128 sse_frc_ps(__m128 x)
{
	__m128 flr0 = sse_flr_ps(x);
	__m128 sub0 = _mm_sub_ps(x, flr0);
	return sub0;
}

GLM_FUNC_QUALIFIER __m128 sse_mod_ps(__m128 x, __m128 y)
{
	__m128 div0 = _mm_div_ps(x, y);
	__m128 flr0 = sse_flr_ps(div0);
	__m128 mul0 = _mm_mul_ps(y, flr0);
	__m128 sub0 = _mm_sub_ps(x, mul0);
	return sub0;
}

/// TODO
/*
GLM_FUNC_QUALIFIER __m128 sse_modf_ps(__m128 x, __m128i & i)
{
	__m128 empty;
	return empty;
}
*/

//GLM_FUNC_QUALIFIER __m128 _mm_min_ps(__m128 x, __m128 y)

//GLM_FUNC_QUALIFIER __m128 _mm_max_ps(__m128 x, __m128 y)

GLM_FUNC_QUALIFIER __m128 sse_clp_ps(__m128 v, __m128 minVal, __m128 maxVal)
{
	__m128 min0 = _mm_min_ps(v, maxVal);
	__m128 max0 = _mm_max_ps(min0, minVal);
	return max0;
}

GLM_FUNC_QUALIFIER __m128 sse_mix_ps(__m128 v1, __m128 v2, __m128 a)
{
	__m128 sub0 = _mm_sub_ps(glm::detail::one, a);
	__m128 mul0 = _mm_mul_ps(v1, sub0);
	__m128 mul1 = _mm_mul_ps(v2, a);
	__m128 add0 = _mm_add_ps(mul0, mul1);
	return add0;
}

GLM_FUNC_QUALIFIER __m128 sse_stp_ps(__m128 edge, __m128 x)
{
	__m128 cmp = _mm_cmple_ps(x, edge);
	if(_mm_movemask_ps(cmp) == 0)
		return glm::detail::one;
	else
		return glm::detail::zero;
}

GLM_FUNC_QUALIFIER __m128 sse_ssp_ps(__m128 edge0, __m128 edge1, __m128 x)
{
	__m128 sub0 = _mm_sub_ps(x, edge0);
	__m128 sub1 = _mm_sub_ps(edge1, edge0);
	__m128 div0 = _mm_sub_ps(sub0, sub1);
	__m128 clp0 = sse_clp_ps(div0, glm::detail::zero, glm::detail::one);
	__m128 mul0 = _mm_mul_ps(glm::detail::two, clp0);
	__m128 sub2 = _mm_sub_ps(glm::detail::three, mul0);
	__m128 mul1 = _mm_mul_ps(clp0, clp0);
	__m128 mul2 = _mm_mul_ps(mul1, sub2);
	return mul2;
}

/// \todo
//GLM_FUNC_QUALIFIER __m128 sse_nan_ps(__m128 x)
//{
//	__m128 empty;
//	return empty;
//}

/// \todo
//GLM_FUNC_QUALIFIER __m128 sse_inf_ps(__m128 x)
//{
//	__m128 empty;
//	return empty;
//}

// SSE scalar reciprocal sqrt using rsqrt op, plus one Newton-Rhaphson iteration
// By Elan Ruskin, http://assemblyrequired.crashworks.org/
GLM_FUNC_QUALIFIER __m128 sse_sqrt_wip_ss(__m128 const & x)
{
	__m128 const recip = _mm_rsqrt_ss(x);  // "estimate" opcode
	__m128 const half = _mm_set_ps1(0.5f);
	__m128 const halfrecip = _mm_mul_ss(half, recip);
	__m128 const threeminus_xrr = _mm_sub_ss(three, _mm_mul_ss(x, _mm_mul_ss (recip, recip)));
	return _mm_mul_ss(halfrecip, threeminus_xrr);
}

}//namespace detail
}//namespace glms

