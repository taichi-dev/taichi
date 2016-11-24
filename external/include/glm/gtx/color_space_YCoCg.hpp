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
/// @ref gtx_color_space_YCoCg
/// @file glm/gtx/color_space_YCoCg.hpp
/// @date 2008-10-28 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_color_space_YCoCg GLM_GTX_color_space_YCoCg
/// @ingroup gtx
///
/// @brief RGB to YCoCg conversions and operations
/// 
/// <glm/gtx/color_space_YCoCg.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_color_space_YCoCg extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_color_space_YCoCg
	/// @{

	/// Convert a color from RGB color space to YCoCg color space.
	/// @see gtx_color_space_YCoCg
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rgb2YCoCg(
		tvec3<T, P> const & rgbColor);

	/// Convert a color from YCoCg color space to RGB color space.
	/// @see gtx_color_space_YCoCg
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> YCoCg2rgb(
		tvec3<T, P> const & YCoCgColor);

	/// Convert a color from RGB color space to YCoCgR color space.
	/// @see "YCoCg-R: A Color Space with RGB Reversibility and Low Dynamic Range"
	/// @see gtx_color_space_YCoCg
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rgb2YCoCgR(
		tvec3<T, P> const & rgbColor);

	/// Convert a color from YCoCgR color space to RGB color space.
	/// @see "YCoCg-R: A Color Space with RGB Reversibility and Low Dynamic Range"
	/// @see gtx_color_space_YCoCg
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> YCoCgR2rgb(
		tvec3<T, P> const & YCoCgColor);

	/// @}
}//namespace glm

#include "color_space_YCoCg.inl"
