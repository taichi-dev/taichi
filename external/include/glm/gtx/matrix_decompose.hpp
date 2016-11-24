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
/// @ref gtx_matrix_decompose
/// @file glm/gtx/matrix_decompose.hpp
/// @date 2014-08-29 / 2014-08-29
/// @author Christophe Riccio
/// 
/// @see core (dependence)
///
/// @defgroup gtx_matrix_decompose GLM_GTX_matrix_decompose
/// @ingroup gtx
/// 
/// @brief Decomposes a model matrix to translations, rotation and scale components
/// 
/// <glm/gtx/decomposition.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../mat4x4.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtc/matrix_transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_decompose extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_decompose
	/// @{

	/// Decomposes a model matrix to translations, rotation and scale components 
	/// @see gtx_matrix_decompose
	template <typename T, precision P>
	GLM_FUNC_DECL bool decompose(
		tmat4x4<T, P> const & modelMatrix,
		tvec3<T, P> & scale, tquat<T, P> & orientation, tvec3<T, P> & translation, tvec3<T, P> & skew, tvec4<T, P> & perspective);

	/// @}
}//namespace glm

#include "matrix_decompose.inl"
