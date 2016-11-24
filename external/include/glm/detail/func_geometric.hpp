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
/// @file glm/detail/func_geometric.hpp
/// @date 2008-08-03 / 2011-06-14
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
/// 
/// @defgroup core_func_geometric Geometric functions
/// @ingroup core
/// 
/// These operate on vectors as vectors, not component-wise.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "type_vec3.hpp"

namespace glm
{
	/// @addtogroup core_func_geometric
	/// @{

	/// Returns the length of x, i.e., sqrt(x * x).
	/// 
	/// @tparam genType Floating-point vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/length.xml">GLSL length man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL T length(
		vecType<T, P> const & x);

	/// Returns the distance betwwen p0 and p1, i.e., length(p0 - p1).
	///
	/// @tparam genType Floating-point vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/distance.xml">GLSL distance man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL T distance(
		vecType<T, P> const & p0,
		vecType<T, P> const & p1);

	/// Returns the dot product of x and y, i.e., result = x * y.
	///
	/// @tparam genType Floating-point vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/dot.xml">GLSL dot man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL T dot(
		vecType<T, P> const & x,
		vecType<T, P> const & y);

	/// Returns the cross product of x and y.
	///
	/// @tparam valType Floating-point scalar types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/cross.xml">GLSL cross man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P>
	GLM_FUNC_DECL tvec3<T, P> cross(
		tvec3<T, P> const & x,
		tvec3<T, P> const & y);

	/// Returns a vector in the same direction as x but with length of 1.
	/// According to issue 10 GLSL 1.10 specification, if length(x) == 0 then result is undefined and generate an error.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/normalize.xml">GLSL normalize man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> normalize(
		vecType<T, P> const & x);

	/// If dot(Nref, I) < 0.0, return N, otherwise, return -N.
	///
	/// @tparam genType Floating-point vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/faceforward.xml">GLSL faceforward man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> faceforward(
		vecType<T, P> const & N,
		vecType<T, P> const & I,
		vecType<T, P> const & Nref);

	/// For the incident vector I and surface orientation N, 
	/// returns the reflection direction : result = I - 2.0 * dot(N, I) * N.
	///
	/// @tparam genType Floating-point vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/reflect.xml">GLSL reflect man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType reflect(
		genType const & I,
		genType const & N);

	/// For the incident vector I and surface normal N, 
	/// and the ratio of indices of refraction eta, 
	/// return the refraction vector.
	///
	/// @tparam genType Floating-point vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/refract.xml">GLSL refract man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.5 Geometric Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> refract(
		vecType<T, P> const & I,
		vecType<T, P> const & N,
		T eta);

	/// @}
}//namespace glm

#include "func_geometric.inl"
