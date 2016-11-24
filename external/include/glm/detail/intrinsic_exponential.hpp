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
/// @file glm/detail/intrinsic_exponential.hpp
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
/*
GLM_FUNC_QUALIFIER __m128 sse_rsqrt_nr_ss(__m128 const x)
{
	__m128 recip = _mm_rsqrt_ss( x );  // "estimate" opcode
	const static __m128 three = { 3, 3, 3, 3 }; // aligned consts for fast load
	const static __m128 half = { 0.5,0.5,0.5,0.5 };
	__m128 halfrecip = _mm_mul_ss( half, recip );
	__m128 threeminus_xrr = _mm_sub_ss( three, _mm_mul_ss( x, _mm_mul_ss ( recip, recip ) ) );
	return _mm_mul_ss( halfrecip, threeminus_xrr );
}
 
GLM_FUNC_QUALIFIER __m128 sse_normalize_fast_ps(  float * RESTRICT vOut, float * RESTRICT vIn )
{
        __m128 x = _mm_load_ss(&vIn[0]);
        __m128 y = _mm_load_ss(&vIn[1]);
        __m128 z = _mm_load_ss(&vIn[2]);
 
        const __m128 l =  // compute x*x + y*y + z*z
                _mm_add_ss(
                 _mm_add_ss( _mm_mul_ss(x,x),
                             _mm_mul_ss(y,y)
                            ),
                 _mm_mul_ss( z, z )
                );
 
 
        const __m128 rsqt = _mm_rsqrt_nr_ss( l );
        _mm_store_ss( &vOut[0] , _mm_mul_ss( rsqt, x ) );
        _mm_store_ss( &vOut[1] , _mm_mul_ss( rsqt, y ) );
        _mm_store_ss( &vOut[2] , _mm_mul_ss( rsqt, z ) );
 
        return _mm_mul_ss( l , rsqt );
}
*/
}//namespace detail
}//namespace glm

#endif//GLM_ARCH
