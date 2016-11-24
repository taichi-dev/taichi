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
/// @date 2009-06-05 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "setup.hpp"

#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else

#include "intrinsic_geometric.hpp"

namespace glm{
namespace detail
{
	void sse_add_ps(__m128 in1[4], __m128 in2[4], __m128 out[4]);

	void sse_sub_ps(__m128 in1[4], __m128 in2[4], __m128 out[4]);

	__m128 sse_mul_ps(__m128 m[4], __m128 v);

	__m128 sse_mul_ps(__m128 v, __m128 m[4]);

	void sse_mul_ps(__m128 const in1[4], __m128 const in2[4], __m128 out[4]);

	void sse_transpose_ps(__m128 const in[4], __m128 out[4]);

	void sse_inverse_ps(__m128 const in[4], __m128 out[4]);

	void sse_rotate_ps(__m128 const in[4], float Angle, float const v[3], __m128 out[4]);

	__m128 sse_det_ps(__m128 const m[4]);

	__m128 sse_slow_det_ps(__m128 const m[4]);

}//namespace detail
}//namespace glm

#include "intrinsic_matrix.inl"

#endif//GLM_ARCH
