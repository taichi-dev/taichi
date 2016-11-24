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
/// @ref gtx_polar_coordinates
/// @file glm/gtx/polar_coordinates.hpp
/// @date 2007-03-06 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_polar_coordinates GLM_GTX_polar_coordinates
/// @ingroup gtx
/// 
/// @brief Conversion from Euclidean space to polar space and revert.
/// 
/// <glm/gtx/polar_coordinates.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_polar_coordinates extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_polar_coordinates
	/// @{

	/// Convert Euclidean to Polar coordinates, x is the xz distance, y, the latitude and z the longitude.
	///
	/// @see gtx_polar_coordinates
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> polar(
		tvec3<T, P> const & euclidean);

	/// Convert Polar to Euclidean coordinates.
	///
	/// @see gtx_polar_coordinates
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> euclidean(
		tvec2<T, P> const & polar);

	/// @}
}//namespace glm

#include "polar_coordinates.inl"
