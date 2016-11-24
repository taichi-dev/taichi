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
/// @ref gtx_spline
/// @file glm/gtx/spline.hpp
/// @date 2007-01-25 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_spline GLM_GTX_spline
/// @ingroup gtx
/// 
/// @brief Spline functions
/// 
/// <glm/gtx/spline.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_spline extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_spline
	/// @{

	/// Return a point from a catmull rom curve.
	/// @see gtx_spline extension.
	template <typename genType> 
	GLM_FUNC_DECL genType catmullRom(
		genType const & v1, 
		genType const & v2, 
		genType const & v3, 
		genType const & v4, 
		typename genType::value_type const & s);
		
	/// Return a point from a hermite curve.
	/// @see gtx_spline extension.
	template <typename genType> 
	GLM_FUNC_DECL genType hermite(
		genType const & v1, 
		genType const & t1, 
		genType const & v2, 
		genType const & t2, 
		typename genType::value_type const & s);
		
	/// Return a point from a cubic curve. 
	/// @see gtx_spline extension.
	template <typename genType> 
	GLM_FUNC_DECL genType cubic(
		genType const & v1, 
		genType const & v2, 
		genType const & v3, 
		genType const & v4, 
		typename genType::value_type const & s);

	/// @}
}//namespace glm

#include "spline.inl"
