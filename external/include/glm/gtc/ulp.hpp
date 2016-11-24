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
/// @ref gtc_ulp
/// @file glm/gtc/ulp.hpp
/// @date 2011-02-21 / 2011-12-12
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtc_ulp GLM_GTC_ulp
/// @ingroup gtc
/// 
/// @brief Allow the measurement of the accuracy of a function against a reference 
/// implementation. This extension works on floating-point data and provide results 
/// in ULP.
/// <glm/gtc/ulp.hpp> need to be included to use these features.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#include "../detail/type_int.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_ulp extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_ulp
	/// @{

	/// Return the next ULP value(s) after the input value(s).
	/// @see gtc_ulp
	template <typename genType>
	GLM_FUNC_DECL genType next_float(genType const & x);

	/// Return the previous ULP value(s) before the input value(s).
	/// @see gtc_ulp
	template <typename genType>
	GLM_FUNC_DECL genType prev_float(genType const & x);

	/// Return the value(s) ULP distance after the input value(s).
	/// @see gtc_ulp
	template <typename genType>
	GLM_FUNC_DECL genType next_float(genType const & x, uint const & Distance);

	/// Return the value(s) ULP distance before the input value(s).
	/// @see gtc_ulp
	template <typename genType>
	GLM_FUNC_DECL genType prev_float(genType const & x, uint const & Distance);
	
	/// Return the distance in the number of ULP between 2 scalars.
	/// @see gtc_ulp
	template <typename T>
	GLM_FUNC_DECL uint float_distance(T const & x, T const & y);

	/// Return the distance in the number of ULP between 2 vectors.
	/// @see gtc_ulp
	template<typename T, template<typename> class vecType>
	GLM_FUNC_DECL vecType<uint> float_distance(vecType<T> const & x, vecType<T> const & y);
	
	/// @}
}// namespace glm

#include "ulp.inl"
