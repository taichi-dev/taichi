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
/// @ref gtx_norm
/// @file glm/gtx/norm.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_quaternion (dependence)
///
/// @defgroup gtx_norm GLM_GTX_norm
/// @ingroup gtx
/// 
/// @brief Various ways to compute vector norms.
/// 
/// <glm/gtx/norm.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_norm extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_norm
	/// @{

	//! Returns the squared length of x.
	//! From GLM_GTX_norm extension.
	template <typename T>
	GLM_FUNC_DECL T length2(
		T const & x);

	//! Returns the squared length of x.
	//! From GLM_GTX_norm extension.
	template <typename genType>
	GLM_FUNC_DECL typename genType::value_type length2(
		genType const & x);
		
	//! Returns the squared distance between p0 and p1, i.e., length(p0 - p1).
	//! From GLM_GTX_norm extension.
	template <typename T>
	GLM_FUNC_DECL T distance2(
		T const & p0,
		T const & p1);
		
	//! Returns the squared distance between p0 and p1, i.e., length(p0 - p1).
	//! From GLM_GTX_norm extension.
	template <typename genType>
	GLM_FUNC_DECL typename genType::value_type distance2(
		genType const & p0,
		genType const & p1);

	//! Returns the L1 norm between x and y.
	//! From GLM_GTX_norm extension.
	template <typename T, precision P>
	GLM_FUNC_DECL T l1Norm(
		tvec3<T, P> const & x,
		tvec3<T, P> const & y);
		
	//! Returns the L1 norm of v.
	//! From GLM_GTX_norm extension.
	template <typename T, precision P>
	GLM_FUNC_DECL T l1Norm(
		tvec3<T, P> const & v);
		
	//! Returns the L2 norm between x and y.
	//! From GLM_GTX_norm extension.
	template <typename T, precision P>
	GLM_FUNC_DECL T l2Norm(
		tvec3<T, P> const & x,
		tvec3<T, P> const & y);
		
	//! Returns the L2 norm of v.
	//! From GLM_GTX_norm extension.
	template <typename T, precision P>
	GLM_FUNC_DECL T l2Norm(
		tvec3<T, P> const & x);
		
	//! Returns the L norm between x and y.
	//! From GLM_GTX_norm extension.
	template <typename T, precision P>
	GLM_FUNC_DECL T lxNorm(
		tvec3<T, P> const & x,
		tvec3<T, P> const & y,
		unsigned int Depth);

	//! Returns the L norm of v.
	//! From GLM_GTX_norm extension.
	template <typename T, precision P>
	GLM_FUNC_DECL T lxNorm(
		tvec3<T, P> const & x,
		unsigned int Depth);

	/// @}
}//namespace glm

#include "norm.inl"
