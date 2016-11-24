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
/// @ref gtx_quaternion
/// @file glm/gtx/quaternion.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_quaternion GLM_GTX_quaternion
/// @ingroup gtx
/// 
/// @brief Extented quaternion types and functions
/// 
/// <glm/gtx/quaternion.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtx/norm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_quaternion extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_quaternion
	/// @{

	//! Compute a cross product between a quaternion and a vector.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> cross(
		tquat<T, P> const & q,
		tvec3<T, P> const & v);

	//! Compute a cross product between a vector and a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> cross(
		tvec3<T, P> const & v,
		tquat<T, P> const & q);

	//! Compute a point on a path according squad equation. 
	//! q1 and q2 are control points; s1 and s2 are intermediate control points.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> squad(
		tquat<T, P> const & q1,
		tquat<T, P> const & q2,
		tquat<T, P> const & s1,
		tquat<T, P> const & s2,
		T const & h);

	//! Returns an intermediate control point for squad interpolation.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> intermediate(
		tquat<T, P> const & prev,
		tquat<T, P> const & curr,
		tquat<T, P> const & next);

	//! Returns a exp of a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> exp(
		tquat<T, P> const & q);

	//! Returns a log of a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> log(
		tquat<T, P> const & q);

	/// Returns x raised to the y power.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> pow(
		tquat<T, P> const & x,
		T const & y);

	//! Returns quarternion square root.
	///
	/// @see gtx_quaternion
	//template<typename T, precision P>
	//tquat<T, P> sqrt(
	//	tquat<T, P> const & q);

	//! Rotates a 3 components vector by a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rotate(
		tquat<T, P> const & q,
		tvec3<T, P> const & v);

	/// Rotates a 4 components vector by a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> rotate(
		tquat<T, P> const & q,
		tvec4<T, P> const & v);

	/// Extract the real component of a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL T extractRealComponent(
		tquat<T, P> const & q);

	/// Converts a quaternion to a 3 * 3 matrix.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> toMat3(
		tquat<T, P> const & x){return mat3_cast(x);}

	/// Converts a quaternion to a 4 * 4 matrix.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> toMat4(
		tquat<T, P> const & x){return mat4_cast(x);}

	/// Converts a 3 * 3 matrix to a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> toQuat(
		tmat3x3<T, P> const & x){return quat_cast(x);}

	/// Converts a 4 * 4 matrix to a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> toQuat(
		tmat4x4<T, P> const & x){return quat_cast(x);}

	/// Quaternion interpolation using the rotation short path.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> shortMix(
		tquat<T, P> const & x,
		tquat<T, P> const & y,
		T const & a);

	/// Quaternion normalized linear interpolation.
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> fastMix(
		tquat<T, P> const & x,
		tquat<T, P> const & y,
		T const & a);

	/// Compute the rotation between two vectors.
	/// param orig vector, needs to be normalized
	/// param dest vector, needs to be normalized
	///
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL tquat<T, P> rotation(
		tvec3<T, P> const & orig, 
		tvec3<T, P> const & dest);

	/// Returns the squared length of x.
	/// 
	/// @see gtx_quaternion
	template<typename T, precision P>
	GLM_FUNC_DECL T length2(tquat<T, P> const & q);

	/// @}
}//namespace glm

#include "quaternion.inl"
