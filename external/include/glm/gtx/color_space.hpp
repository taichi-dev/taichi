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
/// @ref gtx_color_space
/// @file glm/gtx/color_space.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_color_space GLM_GTX_color_space
/// @ingroup gtx
/// 
/// @brief Related to RGB to HSV conversions and operations.
/// 
/// <glm/gtx/color_space.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_color_space extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_color_space
	/// @{

	/// Converts a color from HSV color space to its color in RGB color space.
	/// @see gtx_color_space
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rgbColor(
		tvec3<T, P> const & hsvValue);

	/// Converts a color from RGB color space to its color in HSV color space.
	/// @see gtx_color_space
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> hsvColor(
		tvec3<T, P> const & rgbValue);
		
	/// Build a saturation matrix.
	/// @see gtx_color_space
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> saturation(
		T const s);

	/// Modify the saturation of a color.
	/// @see gtx_color_space
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> saturation(
		T const s,
		tvec3<T, P> const & color);
		
	/// Modify the saturation of a color.
	/// @see gtx_color_space
	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> saturation(
		T const s,
		tvec4<T, P> const & color);
		
	/// Compute color luminosity associating ratios (0.33, 0.59, 0.11) to RGB canals.
	/// @see gtx_color_space
	template <typename T, precision P>
	GLM_FUNC_DECL T luminosity(
		tvec3<T, P> const & color);

	/// @}
}//namespace glm

#include "color_space.inl"
