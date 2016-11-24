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
/// @ref gtx_transform
/// @file glm/gtx/transform.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_matrix_transform (dependence)
/// @see gtx_transform
/// @see gtx_transform2
///
/// @defgroup gtx_transform GLM_GTX_transform
/// @ingroup gtx
///
/// @brief Add transformation matrices
/// 
/// <glm/gtx/transform.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/matrix_transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_transform extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_transform
	/// @{

	/// Transforms a matrix with a translation 4 * 4 matrix created from 3 scalars.
	/// @see gtc_matrix_transform
	/// @see gtx_transform
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> translate(
		tvec3<T, P> const & v);

	/// Builds a rotation 4 * 4 matrix created from an axis of 3 scalars and an angle expressed in degrees. 
	/// @see gtc_matrix_transform
	/// @see gtx_transform
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> rotate(
		T angle, 
		tvec3<T, P> const & v);

	/// Transforms a matrix with a scale 4 * 4 matrix created from a vector of 3 components.
	/// @see gtc_matrix_transform
	/// @see gtx_transform
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> scale(
		tvec3<T, P> const & v);

	/// @}
}// namespace glm

#include "transform.inl"
