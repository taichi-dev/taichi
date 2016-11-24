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
/// @ref gtc_matrix_inverse
/// @file glm/gtc/matrix_inverse.hpp
/// @date 2005-12-21 / 2011-06-05
/// @author Christophe Riccio
///
/// @see core (dependence)
/// 
/// @defgroup gtc_matrix_inverse GLM_GTC_matrix_inverse
/// @ingroup gtc
/// 
/// Defines additional matrix inverting functions.
/// <glm/gtc/matrix_inverse.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../matrix.hpp"
#include "../mat2x2.hpp"
#include "../mat3x3.hpp"
#include "../mat4x4.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_matrix_inverse extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_matrix_inverse
	/// @{

	/// Fast matrix inverse for affine matrix.
	/// 
	/// @param m Input matrix to invert.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-precision floating point value is highly innacurate.
	/// @see gtc_matrix_inverse
	template <typename genType> 
	GLM_FUNC_DECL genType affineInverse(genType const & m);

	/// Compute the inverse transpose of a matrix.
	/// 
	/// @param m Input matrix to invert transpose.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-precision floating point value is highly innacurate.
	/// @see gtc_matrix_inverse
	template <typename genType>
	GLM_FUNC_DECL genType inverseTranspose(genType const & m);

	/// @}
}//namespace glm

#include "matrix_inverse.inl"
