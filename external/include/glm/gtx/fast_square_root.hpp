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
/// @ref gtx_fast_square_root
/// @file glm/gtx/fast_square_root.hpp
/// @date 2006-01-04 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_fast_square_root GLM_GTX_fast_square_root
/// @ingroup gtx
/// 
/// @brief Fast but less accurate implementations of square root based functions.
/// - Sqrt optimisation based on Newton's method, 
/// www.gamedev.net/community/forums/topic.asp?topic id=139956
/// 
/// <glm/gtx/fast_square_root.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../common.hpp"
#include "../exponential.hpp"
#include "../geometric.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_fast_square_root extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_fast_square_root
	/// @{

	/// Faster than the common sqrt function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename genType> 
	GLM_FUNC_DECL genType fastSqrt(genType x);

	/// Faster than the common sqrt function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastSqrt(vecType<T, P> const & x);

	/// Faster than the common inversesqrt function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename genType> 
	GLM_FUNC_DECL genType fastInverseSqrt(genType x);

	/// Faster than the common inversesqrt function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fastInverseSqrt(vecType<T, P> const & x);

	/// Faster than the common length function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename genType>
	GLM_FUNC_DECL genType fastLength(genType x);

	/// Faster than the common length function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL T fastLength(vecType<T, P> const & x);

	/// Faster than the common distance function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename genType>
	GLM_FUNC_DECL genType fastDistance(genType x, genType y);

	/// Faster than the common distance function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL T fastDistance(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Faster than the common normalize function but less accurate.
	///
	/// @see gtx_fast_square_root extension.
	template <typename genType> 
	GLM_FUNC_DECL genType fastNormalize(genType const & x);

	/// @}
}// namespace glm

#include "fast_square_root.inl"
