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
/// @ref gtx_matrix_major_storage
/// @file glm/gtx/matrix_major_storage.hpp
/// @date 2006-04-19 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_matrix_major_storage GLM_GTX_matrix_major_storage
/// @ingroup gtx
/// 
/// @brief Build matrices with specific matrix order, row or column
/// 
/// <glm/gtx/matrix_major_storage.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_major_storage extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_major_storage
	/// @{

	//! Build a row major matrix from row vectors.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat2x2<T, P> rowMajor2(
		tvec2<T, P> const & v1, 
		tvec2<T, P> const & v2);
		
	//! Build a row major matrix from other matrix.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat2x2<T, P> rowMajor2(
		tmat2x2<T, P> const & m);

	//! Build a row major matrix from row vectors.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> rowMajor3(
		tvec3<T, P> const & v1, 
		tvec3<T, P> const & v2, 
		tvec3<T, P> const & v3);

	//! Build a row major matrix from other matrix.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> rowMajor3(
		tmat3x3<T, P> const & m);

	//! Build a row major matrix from row vectors.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> rowMajor4(
		tvec4<T, P> const & v1, 
		tvec4<T, P> const & v2,
		tvec4<T, P> const & v3, 
		tvec4<T, P> const & v4);

	//! Build a row major matrix from other matrix.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> rowMajor4(
		tmat4x4<T, P> const & m);

	//! Build a column major matrix from column vectors.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat2x2<T, P> colMajor2(
		tvec2<T, P> const & v1, 
		tvec2<T, P> const & v2);
		
	//! Build a column major matrix from other matrix.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat2x2<T, P> colMajor2(
		tmat2x2<T, P> const & m);

	//! Build a column major matrix from column vectors.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> colMajor3(
		tvec3<T, P> const & v1, 
		tvec3<T, P> const & v2, 
		tvec3<T, P> const & v3);
		
	//! Build a column major matrix from other matrix.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat3x3<T, P> colMajor3(
		tmat3x3<T, P> const & m);
		
	//! Build a column major matrix from column vectors.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P>
	GLM_FUNC_DECL tmat4x4<T, P> colMajor4(
		tvec4<T, P> const & v1, 
		tvec4<T, P> const & v2, 
		tvec4<T, P> const & v3, 
		tvec4<T, P> const & v4);
				
	//! Build a column major matrix from other matrix.
	//! From GLM_GTX_matrix_major_storage extension.
	template <typename T, precision P> 
	GLM_FUNC_DECL tmat4x4<T, P> colMajor4(
		tmat4x4<T, P> const & m);

	/// @}
}//namespace glm

#include "matrix_major_storage.inl"
