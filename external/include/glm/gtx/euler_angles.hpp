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
/// @ref gtx_euler_angles
/// @file glm/gtx/euler_angles.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_half_float (dependence)
///
/// @defgroup gtx_euler_angles GLM_GTX_euler_angles
/// @ingroup gtx
/// 
/// @brief Build matrices from Euler angles.
/// 
/// <glm/gtx/euler_angles.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_euler_angles extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_euler_angles
	/// @{

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from an euler angle X.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleX(
		T const & angleX);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from an euler angle Y.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleY(
		T const & angleY);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from an euler angle Z.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleZ(
		T const & angleZ);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Y).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleXY(
		T const & angleX,
		T const & angleY);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleYX(
		T const & angleY,
		T const & angleX);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Z).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleXZ(
		T const & angleX,
		T const & angleZ);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * X).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleZX(
		T const & angle,
		T const & angleX);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * Z).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleYZ(
		T const & angleY,
		T const & angleZ);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * Y).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleZY(
		T const & angleZ,
		T const & angleY);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleYXZ(
		T const & yaw,
		T const & pitch,
		T const & roll);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat4x4<T, defaultp> yawPitchRoll(
		T const & yaw,
		T const & pitch,
		T const & roll);

	/// Creates a 2D 2 * 2 rotation matrix from an euler angle.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat2x2<T, defaultp> orientate2(T const & angle);

	/// Creates a 2D 4 * 4 homogeneous rotation matrix from an euler angle.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL tmat3x3<T, defaultp> orientate3(T const & angle);

	/// Creates a 3D 3 * 3 rotation matrix from euler angles (Y * X * Z). 
	/// @see gtx_euler_angles
	template <typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> orientate3(tvec3<T, P> const & angles);
		
	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> orientate4(tvec3<T, P> const & angles);

	/// @}
}//namespace glm

#include "euler_angles.inl"
