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
/// @file glm/detail/intrinsic_geometric.hpp
/// @date 2009-05-08 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "setup.hpp"

#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else

#include "intrinsic_common.hpp"

namespace glm{
namespace detail
{
	//length
	__m128 sse_len_ps(__m128 x);

	//distance
	__m128 sse_dst_ps(__m128 p0, __m128 p1);

	//dot
	__m128 sse_dot_ps(__m128 v1, __m128 v2);

	// SSE1
	__m128 sse_dot_ss(__m128 v1, __m128 v2);

	//cross
	__m128 sse_xpd_ps(__m128 v1, __m128 v2);

	//normalize
	__m128 sse_nrm_ps(__m128 v);

	//faceforward
	__m128 sse_ffd_ps(__m128 N, __m128 I, __m128 Nref);

	//reflect
	__m128 sse_rfe_ps(__m128 I, __m128 N);

	//refract
	__m128 sse_rfa_ps(__m128 I, __m128 N, __m128 eta);

}//namespace detail
}//namespace glm

#include "intrinsic_geometric.inl"

#endif//GLM_ARCH
