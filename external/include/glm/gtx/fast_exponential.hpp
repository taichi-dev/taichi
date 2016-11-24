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
/// Restrictions:
///		By making use of the Software for military purposes, you choose to make
///		a Bunny unhappy.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @ref gtx_fast_exponential
/// @file glm/gtx/fast_exponential.hpp
/// @date 2006-01-09 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_half_float (dependence)
///
/// @defgroup gtx_fast_exponential GLM_GTX_fast_exponential
/// @ingroup gtx
/// 
/// @brief Fast but less accurate implementations of exponential based functions.
/// 
/// <glm/gtx/fast_exponential.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_fast_exponential extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_fast_exponential
	/// @{

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template <typename genType>
	GLM_FUNC_DECL genType fastPow(genType x, genType y);

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastPow(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template <typename genTypeT, typename genTypeU>
	GLM_FUNC_DECL genTypeT fastPow(genTypeT x, genTypeU y);

	/// Faster than the common pow function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastPow(vecType<T, P> const & x);

	/// Faster than the common exp function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T>
	GLM_FUNC_DECL T fastExp(T x);

	/// Faster than the common exp function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastExp(vecType<T, P> const & x);

	/// Faster than the common log function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T>
	GLM_FUNC_DECL T fastLog(T x);

	/// Faster than the common exp2 function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastLog(vecType<T, P> const & x);

	/// Faster than the common exp2 function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T>
	GLM_FUNC_DECL T fastExp2(T x);

	/// Faster than the common exp2 function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastExp2(vecType<T, P> const & x);

	/// Faster than the common log2 function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T>
	GLM_FUNC_DECL T fastLog2(T x);

	/// Faster than the common log2 function but less accurate.
	/// @see gtx_fast_exponential
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastLog2(vecType<T, P> const & x);

	/// @}
}//namespace glm

#include "fast_exponential.inl"
