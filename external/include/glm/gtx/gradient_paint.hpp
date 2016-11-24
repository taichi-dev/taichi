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
/// @ref gtx_gradient_paint
/// @file glm/gtx/gradient_paint.hpp
/// @date 2009-03-06 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_optimum_pow (dependence)
///
/// @defgroup gtx_gradient_paint GLM_GTX_gradient_paint
/// @ingroup gtx
/// 
/// @brief Functions that return the color of procedural gradient for specific coordinates.
/// <glm/gtx/gradient_paint.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_gradient_paint extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_gradient_paint
	/// @{

	/// Return a color from a radial gradient.
	/// @see - gtx_gradient_paint
	template <typename T, precision P>
	GLM_FUNC_DECL T radialGradient(
		tvec2<T, P> const & Center,
		T const & Radius,
		tvec2<T, P> const & Focal,
		tvec2<T, P> const & Position);

	/// Return a color from a linear gradient.
	/// @see - gtx_gradient_paint
	template <typename T, precision P>
	GLM_FUNC_DECL T linearGradient(
		tvec2<T, P> const & Point0,
		tvec2<T, P> const & Point1,
		tvec2<T, P> const & Position);

	/// @}
}// namespace glm

#include "gradient_paint.inl"
