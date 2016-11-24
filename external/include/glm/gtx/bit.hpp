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
/// @ref gtx_bit
/// @file glm/gtx/bit.hpp
/// @date 2007-03-14 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_half_float (dependence)
///
/// @defgroup gtx_bit GLM_GTX_bit
/// @ingroup gtx
/// 
/// @brief Allow to perform bit operations on integer values
/// 
/// <glm/gtx/bit.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../gtc/bitfield.hpp"

#if(defined(GLM_MESSAGES))
#	pragma message("GLM: GLM_GTX_bit extension is deprecated, include GLM_GTC_bitfield and GLM_GTC_integer instead")
#endif

namespace glm
{
	/// @addtogroup gtx_bit
	/// @{

	/// @see gtx_bit
	template <typename genIUType>
	GLM_FUNC_DECL genIUType highestBitValue(genIUType Value);

	/// Find the highest bit set to 1 in a integer variable and return its value.
	///
	/// @see gtx_bit
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> highestBitValue(vecType<T, P> const & value);

	/// Return the power of two number which value is just higher the input value.
	/// Deprecated, use ceilPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template <typename genIUType>
	GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoAbove(genIUType Value);

	/// Return the power of two number which value is just higher the input value.
	/// Deprecated, use ceilPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> powerOfTwoAbove(vecType<T, P> const & value);

	/// Return the power of two number which value is just lower the input value.
	/// Deprecated, use floorPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template <typename genIUType>
	GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoBelow(genIUType Value);

	/// Return the power of two number which value is just lower the input value.
	/// Deprecated, use floorPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> powerOfTwoBelow(vecType<T, P> const & value);

	/// Return the power of two number which value is the closet to the input value.
	/// Deprecated, use roundPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template <typename genIUType>
	GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoNearest(genIUType Value);

	/// Return the power of two number which value is the closet to the input value.
	/// Deprecated, use roundPowerOfTwo from GTC_round instead
	///
	/// @see gtc_round
	/// @see gtx_bit
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> powerOfTwoNearest(vecType<T, P> const & value);

	/// @}
} //namespace glm


#include "bit.inl"

