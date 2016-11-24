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
/// @ref gtx_optimum_pow
/// @file glm/gtx/optimum_pow.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_optimum_pow GLM_GTX_optimum_pow
/// @ingroup gtx
/// 
/// @brief Integer exponentiation of power functions.
/// 
/// <glm/gtx/optimum_pow.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_optimum_pow extension included")
#endif

namespace glm{
namespace gtx
{
	/// @addtogroup gtx_optimum_pow
	/// @{

	/// Returns x raised to the power of 2.
	///
	/// @see gtx_optimum_pow
	template <typename genType>
	GLM_FUNC_DECL genType pow2(genType const & x);

	/// Returns x raised to the power of 3.
	///
	/// @see gtx_optimum_pow
	template <typename genType>
	GLM_FUNC_DECL genType pow3(genType const & x);

	/// Returns x raised to the power of 4.
	///
	/// @see gtx_optimum_pow
	template <typename genType>
	GLM_FUNC_DECL genType pow4(genType const & x);

	/// @}
}//namespace gtx
}//namespace glm

#include "optimum_pow.inl"
