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
/// @ref gtx_rotate_vector
/// @file glm/gtx/rotate_vector.hpp
/// @date 2006-11-02 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_transform (dependence)
///
/// @defgroup gtx_rotate_vector GLM_GTX_rotate_vector
/// @ingroup gtx
/// 
/// @brief Function to directly rotate a vector
/// 
/// <glm/gtx/rotate_vector.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_rotate_vector extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_rotate_vector
	/// @{

	/// Returns Spherical interpolation between two vectors
	/// 
	/// @param x A first vector
	/// @param y A second vector
	/// @param a Interpolation factor. The interpolation is defined beyond the range [0, 1].
	/// 
	/// @see gtx_rotate_vector
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> slerp(
		tvec3<T, P> const & x,
		tvec3<T, P> const & y,
		T const & a);

	//! Rotate a two dimensional vector.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec2<T, P> rotate(
		tvec2<T, P> const & v,
		T const & angle);
		
	//! Rotate a three dimensional vector around an axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rotate(
		tvec3<T, P> const & v,
		T const & angle,
		tvec3<T, P> const & normal);
		
	//! Rotate a four dimensional vector around an axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> rotate(
		tvec4<T, P> const & v,
		T const & angle,
		tvec3<T, P> const & normal);
		
	//! Rotate a three dimensional vector around the X axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rotateX(
		tvec3<T, P> const & v,
		T const & angle);

	//! Rotate a three dimensional vector around the Y axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rotateY(
		tvec3<T, P> const & v,
		T const & angle);
		
	//! Rotate a three dimensional vector around the Z axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> rotateZ(
		tvec3<T, P> const & v,
		T const & angle);
		
	//! Rotate a four dimentionnals vector around the X axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> rotateX(
		tvec4<T, P> const & v,
		T const & angle);
		
	//! Rotate a four dimensional vector around the X axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> rotateY(
		tvec4<T, P> const & v,
		T const & angle);
		
	//! Rotate a four dimensional vector around the X axis.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> rotateZ(
		tvec4<T, P> const & v,
		T const & angle);
		
	//! Build a rotation matrix from a normal and a up vector.
	//! From GLM_GTX_rotate_vector extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> orientation(
		tvec3<T, P> const & Normal,
		tvec3<T, P> const & Up);

	/// @}
}//namespace glm

#include "rotate_vector.inl"
