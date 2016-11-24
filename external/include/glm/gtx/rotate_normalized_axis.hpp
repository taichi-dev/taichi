///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2012 G-Truc Creation (www.g-truc.net)
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
/// @ref gtx_rotate_normalized_axis
/// @file glm/gtx/rotate_normalized_axis.hpp
/// @date 2012-12-13 / 2012-12-13
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_matrix_transform
/// @see gtc_quaternion
/// 
/// @defgroup gtx_rotate_normalized_axis GLM_GTX_rotate_normalized_axis
/// @ingroup gtx
/// 
/// @brief Quaternions and matrices rotations around normalized axis.
/// 
/// <glm/gtx/rotate_normalized_axis.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/epsilon.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_rotate_normalized_axis extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_rotate_normalized_axis
	/// @{

	/// Builds a rotation 4 * 4 matrix created from a normalized axis and an angle. 
	/// 
	/// @param m Input matrix multiplied by this rotation matrix.
	/// @param angle Rotation angle expressed in radians if GLM_FORCE_RADIANS is define or degrees otherwise.
	/// @param axis Rotation axis, must be normalized.
	/// @tparam T Value type used to build the matrix. Currently supported: half (not recommanded), float or double.
	/// 
	/// @see gtx_rotate_normalized_axis
	/// @see - rotate(T angle, T x, T y, T z) 
	/// @see - rotate(tmat4x4<T, P> const & m, T angle, T x, T y, T z) 
	/// @see - rotate(T angle, tvec3<T, P> const & v) 
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> rotateNormalizedAxis(
		tmat4x4<T, P> const & m,
		T const & angle,
		tvec3<T, P> const & axis);

	/// Rotates a quaternion from a vector of 3 components normalized axis and an angle.
	/// 
	/// @param q Source orientation
	/// @param angle Angle expressed in radians if GLM_FORCE_RADIANS is define or degrees otherwise.
	/// @param axis Normalized axis of the rotation, must be normalized.
	/// 
	/// @see gtx_rotate_normalized_axis
	template <typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> rotateNormalizedAxis(
		tquat<T, P> const & q,
		T const & angle,
		tvec3<T, P> const & axis);

	/// @}
}//namespace glm

#include "rotate_normalized_axis.inl"
