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
/// @ref gtx_closest_point
/// @file glm/gtx/closest_point.hpp
/// @date 2005-12-30 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_closest_point GLM_GTX_closest_point
/// @ingroup gtx
///
/// @brief Find the point on a straight line which is the closet of a point.
/// 
/// <glm/gtx/closest_point.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_closest_point extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_closest_point
	/// @{

	/// Find the point on a straight line which is the closet of a point. 
	/// @see gtx_closest_point
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> closestPointOnLine(
		tvec3<T, P> const & point,
		tvec3<T, P> const & a, 
		tvec3<T, P> const & b);
	
	/// 2d lines work as well	
	template <typename T, precision P>
	GLM_FUNC_DECL tvec2<T, P> closestPointOnLine(
		tvec2<T, P> const & point,
		tvec2<T, P> const & a, 
		tvec2<T, P> const & b);	

	/// @}
}// namespace glm

#include "closest_point.inl"
