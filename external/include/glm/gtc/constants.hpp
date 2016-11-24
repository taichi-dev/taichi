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
/// @ref gtc_constants
/// @file glm/gtc/constants.hpp
/// @date 2011-09-30 / 2012-01-25
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_half_float (dependence)
///
/// @defgroup gtc_constants GLM_GTC_constants
/// @ingroup gtc
/// 
/// @brief Provide a list of constants and precomputed useful values.
/// 
/// <glm/gtc/constants.hpp> need to be included to use these features.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/setup.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_constants extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_constants
	/// @{

	/// Return the epsilon constant for floating point types.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType epsilon();

	/// Return 0.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType zero();

	/// Return 1.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType one();

	/// Return the pi constant.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType pi();

	/// Return pi * 2.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType two_pi();

	/// Return square root of pi.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_pi();

	/// Return pi / 2.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType half_pi();

	/// Return pi / 2 * 3.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType three_over_two_pi();

	/// Return pi / 4.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType quarter_pi();

	/// Return 1 / pi.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType one_over_pi();

	/// Return 1 / (pi * 2).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType one_over_two_pi();

	/// Return 2 / pi.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType two_over_pi();

	/// Return 4 / pi.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType four_over_pi();

	/// Return 2 / sqrt(pi).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType two_over_root_pi();

	/// Return 1 / sqrt(2).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType one_over_root_two();

	/// Return sqrt(pi / 2).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_half_pi();

	/// Return sqrt(2 * pi).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_two_pi();

	/// Return sqrt(ln(4)).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_ln_four();

	/// Return e constant.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType e();

	/// Return Euler's constant.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType euler();

	/// Return sqrt(2).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_two();

	/// Return sqrt(3).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_three();

	/// Return sqrt(5).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType root_five();

	/// Return ln(2).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType ln_two();

	/// Return ln(10).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType ln_ten();

	/// Return ln(ln(2)).
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType ln_ln_two();

	/// Return 1 / 3.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType third();

	/// Return 2 / 3.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType two_thirds();

	/// Return the golden ratio constant.
	/// @see gtc_constants
	template <typename genType>
	GLM_FUNC_DECL genType golden_ratio();

	/// @}
} //namespace glm

#include "constants.inl"
