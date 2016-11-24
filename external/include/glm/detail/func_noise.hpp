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
/// @ref core
/// @file glm/detail/func_noise.hpp
/// @date 2008-08-01 / 2011-06-18
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.13 Noise Functions</a>
/// 
/// @defgroup core_func_noise Noise functions
/// @ingroup core
/// 
/// Noise functions are stochastic functions that can be used to increase visual 
/// complexity. Values returned by the following noise functions give the 
/// appearance of randomness, but are not truly random.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "type_vec1.hpp"
#include "type_vec2.hpp"
#include "type_vec3.hpp"
#include "setup.hpp"

namespace glm
{
	/// @addtogroup core_func_noise
	/// @{

	/// Returns a 1D noise value based on the input value x.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/noise1.xml">GLSL noise1 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.13 Noise Functions</a>
	template <typename genType>
	GLM_FUNC_DECL typename genType::value_type noise1(genType const & x);

	/// Returns a 2D noise value based on the input value x.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/noise2.xml">GLSL noise2 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.13 Noise Functions</a>
	template <typename genType>
	GLM_FUNC_DECL tvec2<typename genType::value_type, defaultp> noise2(genType const & x);

	/// Returns a 3D noise value based on the input value x.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/noise3.xml">GLSL noise3 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.13 Noise Functions</a>
	template <typename genType>
	GLM_FUNC_DECL tvec3<typename genType::value_type, defaultp> noise3(genType const & x);

	/// Returns a 4D noise value based on the input value x.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/noise4.xml">GLSL noise4 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.13 Noise Functions</a>
	template <typename genType>
	GLM_FUNC_DECL tvec4<typename genType::value_type, defaultp> noise4(genType const & x);

	/// @}
}//namespace glm

#include "func_noise.inl"
