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
/// @ref gtc_integer
/// @file glm/gtc/integer.hpp
/// @date 2014-11-17 / 2014-11-17
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_integer (dependence)
///
/// @defgroup gtc_integer GLM_GTC_integer
/// @ingroup gtc
/// 
/// @brief Allow to perform bit operations on integer values
/// 
/// <glm/gtc/integer.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#include "../detail/func_common.hpp"
#include "../detail/func_integer.hpp"
#include "../detail/func_exponential.hpp"
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_integer extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_integer
	/// @{

	/// Returns the log2 of x for integer values. Can be reliably using to compute mipmap count from the texture size.
	/// @see gtc_integer
	template <typename genIUType>
	GLM_FUNC_DECL genIUType log2(genIUType x);

	/// Modulus. Returns x % y
	/// for each component in x using the floating point value y.
	///
	/// @tparam genIUType Integer-point scalar or vector types.
	///
	/// @see gtc_integer
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/mod.xml">GLSL mod man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genIUType>
	GLM_FUNC_DECL genIUType mod(genIUType x, genIUType y);

	/// Modulus. Returns x % y
	/// for each component in x using the floating point value y.
	///
	/// @tparam T Integer scalar types.
	/// @tparam vecType vector types.
	///
	/// @see gtc_integer
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/mod.xml">GLSL mod man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> mod(vecType<T, P> const & x, T y);

	/// Modulus. Returns x % y
	/// for each component in x using the floating point value y.
	///
	/// @tparam T Integer scalar types.
	/// @tparam vecType vector types.
	///
	/// @see gtc_integer
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/mod.xml">GLSL mod man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> mod(vecType<T, P> const & x, vecType<T, P> const & y);

	/// @}
} //namespace glm

#include "integer.inl"
