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
/// @ref gtx_component_wise
/// @file glm/gtx/component_wise.hpp
/// @date 2007-05-21 / 2011-06-07
/// @author Christophe Riccio
/// 
/// @see core (dependence)
///
/// @defgroup gtx_component_wise GLM_GTX_component_wise
/// @ingroup gtx
/// 
/// @brief Operations between components of a type
/// 
/// <glm/gtx/component_wise.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_component_wise extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_component_wise
	/// @{

	/// Add all vector components together. 
	/// @see gtx_component_wise
	template <typename genType> 
	GLM_FUNC_DECL typename genType::value_type compAdd(
		genType const & v);

	/// Multiply all vector components together. 
	/// @see gtx_component_wise
	template <typename genType> 
	GLM_FUNC_DECL typename genType::value_type compMul(
		genType const & v);

	/// Find the minimum value between single vector components.
	/// @see gtx_component_wise
	template <typename genType> 
	GLM_FUNC_DECL typename genType::value_type compMin(
		genType const & v);

	/// Find the maximum value between single vector components.
	/// @see gtx_component_wise
	template <typename genType> 
	GLM_FUNC_DECL typename genType::value_type compMax(
		genType const & v);

	/// @}
}//namespace glm

#include "component_wise.inl"
