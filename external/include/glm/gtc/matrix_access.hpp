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
/// @ref gtc_matrix_access
/// @file glm/gtc/matrix_access.hpp
/// @date 2005-12-27 / 2011-05-16
/// @author Christophe Riccio
/// 
/// @see core (dependence)
/// 
/// @defgroup gtc_matrix_access GLM_GTC_matrix_access
/// @ingroup gtc
/// 
/// Defines functions to access rows or columns of a matrix easily.
/// <glm/gtc/matrix_access.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../detail/setup.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_matrix_access extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_matrix_access
	/// @{

	/// Get a specific row of a matrix.
	/// @see gtc_matrix_access
	template <typename genType>
	GLM_FUNC_DECL typename genType::row_type row(
		genType const & m,
		length_t index);

	/// Set a specific row to a matrix.
	/// @see gtc_matrix_access
	template <typename genType>
	GLM_FUNC_DECL genType row(
		genType const & m,
		length_t index,
		typename genType::row_type const & x);

	/// Get a specific column of a matrix.
	/// @see gtc_matrix_access
	template <typename genType>
	GLM_FUNC_DECL typename genType::col_type column(
		genType const & m,
		length_t index);

	/// Set a specific column to a matrix.
	/// @see gtc_matrix_access
	template <typename genType>
	GLM_FUNC_DECL genType column(
		genType const & m,
		length_t index,
		typename genType::col_type const & x);

	/// @}
}//namespace glm

#include "matrix_access.inl"
