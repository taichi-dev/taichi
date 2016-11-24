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
/// @ref gtx_multiple
/// @file glm/gtx/multiple.hpp
/// @date 2009-10-26 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_multiple GLM_GTX_multiple
/// @ingroup gtx
/// 
/// @brief Find the closest number of a number multiple of other number.
/// 
/// <glm/gtx/multiple.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../gtc/round.hpp"

#pragma message("GLM: GLM_GTX_multiple extension is deprecated, use GLM_GTC_round instead.")

namespace glm
{
	/// @addtogroup gtx_multiple
	/// @{

	/// Higher multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	/// @param Source 
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtx_multiple
	template <typename genType>
	GLM_DEPRECATED GLM_FUNC_DECL genType higherMultiple(
		genType Source,
		genType Multiple);

	/// Higher multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	/// @param Source 
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtx_multiple
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> higherMultiple(
		vecType<T, P> const & Source,
		vecType<T, P> const & Multiple);

	/// Lower multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	/// @param Source 
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtx_multiple
	template <typename genType>
	GLM_DEPRECATED GLM_FUNC_DECL genType lowerMultiple(
		genType Source,
		genType Multiple);

	/// Lower multiple number of Source.
	///
	/// @tparam genType Floating-point or integer scalar or vector types.
	/// @param Source 
	/// @param Multiple Must be a null or positive value
	///
	/// @see gtx_multiple
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> lowerMultiple(
		vecType<T, P> const & Source,
		vecType<T, P> const & Multiple);

	/// @}
}//namespace glm

#include "multiple.inl"
