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
/// @file glm/detail/func_vector_relational.hpp
/// @date 2008-08-03 / 2011-06-15
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
/// 
/// @defgroup core_func_vector_relational Vector Relational Functions
/// @ingroup core
/// 
/// Relational and equality operators (<, <=, >, >=, ==, !=) are defined to 
/// operate on scalars and produce scalar Boolean results. For vector results, 
/// use the following built-in functions. 
/// 
/// In all cases, the sizes of all the input and return vectors for any particular 
/// call must match.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "precision.hpp"
#include "setup.hpp"

namespace glm
{
	/// @addtogroup core_func_vector_relational
	/// @{

	/// Returns the component-wise comparison result of x < y.
	/// 
	/// @tparam vecType Floating-point or integer vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/lessThan.xml">GLSL lessThan man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> lessThan(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns the component-wise comparison of result x <= y.
	///
	/// @tparam vecType Floating-point or integer vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/lessThanEqual.xml">GLSL lessThanEqual man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> lessThanEqual(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns the component-wise comparison of result x > y.
	///
	/// @tparam vecType Floating-point or integer vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/greaterThan.xml">GLSL greaterThan man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> greaterThan(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns the component-wise comparison of result x >= y.
	///
	/// @tparam vecType Floating-point or integer vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/greaterThanEqual.xml">GLSL greaterThanEqual man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> greaterThanEqual(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns the component-wise comparison of result x == y.
	///
	/// @tparam vecType Floating-point, integer or boolean vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/equal.xml">GLSL equal man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> equal(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns the component-wise comparison of result x != y.
	/// 
	/// @tparam vecType Floating-point, integer or boolean vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/notEqual.xml">GLSL notEqual man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> notEqual(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns true if any component of x is true.
	///
	/// @tparam vecType Boolean vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/any.xml">GLSL any man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool any(vecType<bool, P> const & v);

	/// Returns true if all components of x are true.
	///
	/// @tparam vecType Boolean vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/all.xml">GLSL all man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL bool all(vecType<bool, P> const & v);

	/// Returns the component-wise logical complement of x.
	/// /!\ Because of language incompatibilities between C++ and GLSL, GLM defines the function not but not_ instead.
	///
	/// @tparam vecType Boolean vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/not.xml">GLSL not man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.7 Vector Relational Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> not_(vecType<bool, P> const & v);

	/// @}
}//namespace glm

#include "func_vector_relational.inl"
