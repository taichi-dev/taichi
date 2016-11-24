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
/// @ref gtx_vector_query
/// @file glm/gtx/vector_query.hpp
/// @date 2008-03-10 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_vector_query GLM_GTX_vector_query
/// @ingroup gtx
/// 
/// @brief Query informations of vector types
///
/// <glm/gtx/vector_query.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include <cfloat>
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_vector_query extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_vector_query
	/// @{

	//! Check whether two vectors are collinears.
	/// @see gtx_vector_query extensions.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool areCollinear(vecType<T, P> const & v0, vecType<T, P> const & v1, T const & epsilon);
		
	//! Check whether two vectors are orthogonals.
	/// @see gtx_vector_query extensions.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool areOrthogonal(vecType<T, P> const & v0, vecType<T, P> const & v1, T const & epsilon);

	//! Check whether a vector is normalized.
	/// @see gtx_vector_query extensions.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool isNormalized(vecType<T, P> const & v, T const & epsilon);
		
	//! Check whether a vector is null.
	/// @see gtx_vector_query extensions.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool isNull(vecType<T, P> const & v, T const & epsilon);

	//! Check whether a each component of a vector is null.
	/// @see gtx_vector_query extensions.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> isCompNull(vecType<T, P> const & v, T const & epsilon);

	//! Check whether two vectors are orthonormal.
	/// @see gtx_vector_query extensions.
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool areOrthonormal(vecType<T, P> const & v0, vecType<T, P> const & v1, T const & epsilon);

	/// @}
}// namespace glm

#include "vector_query.inl"
