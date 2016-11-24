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
/// @file glm/detail/intrinsic_common.hpp
/// @date 2009-05-11 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "setup.hpp"

#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else

namespace glm{
namespace detail
{
	__m128 sse_abs_ps(__m128 x);

	__m128 sse_sgn_ps(__m128 x);

	//floor
	__m128 sse_flr_ps(__m128 v);

	//trunc
	__m128 sse_trc_ps(__m128 v);

	//round
	__m128 sse_nd_ps(__m128 v);

	//roundEven
	__m128 sse_rde_ps(__m128 v);

	__m128 sse_rnd_ps(__m128 x);

	__m128 sse_ceil_ps(__m128 v);

	__m128 sse_frc_ps(__m128 x);

	__m128 sse_mod_ps(__m128 x, __m128 y);

	__m128 sse_modf_ps(__m128 x, __m128i & i);

	//GLM_FUNC_QUALIFIER __m128 sse_min_ps(__m128 x, __m128 y)

	//GLM_FUNC_QUALIFIER __m128 sse_max_ps(__m128 x, __m128 y)

	__m128 sse_clp_ps(__m128 v, __m128 minVal, __m128 maxVal);

	__m128 sse_mix_ps(__m128 v1, __m128 v2, __m128 a);

	__m128 sse_stp_ps(__m128 edge, __m128 x);

	__m128 sse_ssp_ps(__m128 edge0, __m128 edge1, __m128 x);

	__m128 sse_nan_ps(__m128 x);

	__m128 sse_inf_ps(__m128 x);

}//namespace detail
}//namespace glm

#include "intrinsic_common.inl"

#endif//GLM_ARCH
