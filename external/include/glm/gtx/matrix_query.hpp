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
/// @ref gtx_matrix_query
/// @file glm/gtx/matrix_query.hpp
/// @date 2007-03-05 / 2011-08-28
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_vector_query (dependence)
///
/// @defgroup gtx_matrix_query GLM_GTX_matrix_query
/// @ingroup gtx
/// 
/// @brief Query to evaluate matrix properties
/// 
/// <glm/gtx/matrix_query.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/vector_query.hpp"
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_query extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_query
	/// @{

	/// Return whether a matrix a null matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P>
	GLM_FUNC_DECL bool isNull(tmat2x2<T, P> const & m, T const & epsilon);
		
	/// Return whether a matrix a null matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P>
	GLM_FUNC_DECL bool isNull(tmat3x3<T, P> const & m, T const & epsilon);
		
	/// Return whether a matrix is a null matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P>
	GLM_FUNC_DECL bool isNull(tmat4x4<T, P> const & m, T const & epsilon);
			
	/// Return whether a matrix is an identity matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_DECL bool isIdentity(matType<T, P> const & m, T const & epsilon);

	/// Return whether a matrix is a normalized matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P>
	GLM_FUNC_DECL bool isNormalized(tmat2x2<T, P> const & m, T const & epsilon);

	/// Return whether a matrix is a normalized matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P>
	GLM_FUNC_DECL bool isNormalized(tmat3x3<T, P> const & m, T const & epsilon);

	/// Return whether a matrix is a normalized matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P>
	GLM_FUNC_DECL bool isNormalized(tmat4x4<T, P> const & m, T const & epsilon);

	/// Return whether a matrix is an orthonormalized matrix.
	/// From GLM_GTX_matrix_query extension.
	template<typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_DECL bool isOrthogonal(matType<T, P> const & m, T const & epsilon);

	/// @}
}//namespace glm

#include "matrix_query.inl"
