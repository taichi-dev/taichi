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
/// @ref gtx_matrix_interpolation
/// @file glm/gtx/matrix_interpolation.hpp
/// @date 2011-03-05 / 2011-06-07
/// @author Ghenadii Ursachi (the.asteroth@gmail.com)
///
/// @see core (dependence)
///
/// @defgroup gtx_matrix_interpolation GLM_GTX_matrix_interpolation
/// @ingroup gtx
/// 
/// @brief Allows to directly interpolate two exiciting matrices.
/// 
/// <glm/gtx/matrix_interpolation.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_interpolation extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_interpolation
	/// @{

	/// Get the axis and angle of the rotation from a matrix.
	/// From GLM_GTX_matrix_interpolation extension.
	template <typename T, precision P>
	GLM_FUNC_DECL void axisAngle(
		tmat4x4<T, P> const & mat,
		tvec3<T, P> & axis,
		T & angle);

	/// Build a matrix from axis and angle.
	/// From GLM_GTX_matrix_interpolation extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> axisAngleMatrix(
		tvec3<T, P> const & axis,
		T const angle);

	/// Extracts the rotation part of a matrix.
	/// From GLM_GTX_matrix_interpolation extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> extractMatrixRotation(
		tmat4x4<T, P> const & mat);

	/// Build a interpolation of 4 * 4 matrixes.
	/// From GLM_GTX_matrix_interpolation extension.
	/// Warning! works only with rotation and/or translation matrixes, scale will generate unexpected results.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> interpolate(
		tmat4x4<T, P> const & m1,
		tmat4x4<T, P> const & m2,
		T const delta);

	/// @}
}//namespace glm

#include "matrix_interpolation.inl"
