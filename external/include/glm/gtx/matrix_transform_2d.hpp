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
/// @ref gtx_matrix_transform_2d
/// @file glm/gtx/matrix_transform_2d.hpp
/// @date 2014-02-20
/// @author Miguel Ángel Pérez Martínez
///
/// @see core (dependence)
///
/// @defgroup gtx_matrix_transform_2d GLM_GTX_matrix_transform_2d
/// @ingroup gtx
/// 
/// @brief Defines functions that generate common 2d transformation matrices.
/// 
/// <glm/gtx/matrix_transform_2d.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../mat3x3.hpp"
#include "../vec2.hpp"


#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_transform_2d extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_transform_2d
	/// @{
	
	/// Builds a translation 3 * 3 matrix created from a vector of 2 components.
	///
	/// @param m Input matrix multiplied by this translation matrix.
	/// @param v Coordinates of a translation vector.		
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x3<T, P> translate(
		tmat3x3<T, P> const & m,
		tvec2<T, P> const & v);

	/// Builds a rotation 3 * 3 matrix created from an angle. 
	///
	/// @param m Input matrix multiplied by this translation matrix.
	/// @param angle Rotation angle expressed in radians if GLM_FORCE_RADIANS is defined or degrees otherwise.
	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> rotate(
		tmat3x3<T, P> const & m,
		T angle);

	/// Builds a scale 3 * 3 matrix created from a vector of 2 components.
	///
	/// @param m Input matrix multiplied by this translation matrix.
	/// @param v Coordinates of a scale vector.		
	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> scale(
		tmat3x3<T, P> const & m,
		tvec2<T, P> const & v);

	/// Builds an horizontal (parallel to the x axis) shear 3 * 3 matrix. 
	///
	/// @param m Input matrix multiplied by this translation matrix.
	/// @param y Shear factor.
	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> shearX(
		tmat3x3<T, P> const & m,
		T y);

	/// Builds a vertical (parallel to the y axis) shear 3 * 3 matrix. 
	///
	/// @param m Input matrix multiplied by this translation matrix.
	/// @param x Shear factor.
	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> shearY(
		tmat3x3<T, P> const & m,
		T x);

	/// @}
}//namespace glm

#include "matrix_transform_2d.inl"
