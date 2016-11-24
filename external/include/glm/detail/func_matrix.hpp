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
/// @file glm/detail/func_matrix.hpp
/// @date 2008-08-03 / 2011-06-15
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.6 Matrix Functions</a>
/// 
/// @defgroup core_func_matrix Matrix functions
/// @ingroup core
/// 
/// For each of the following built-in matrix functions, there is both a 
/// single-precision floating point version, where all arguments and return values 
/// are single precision, and a double-precision floating version, where all 
/// arguments and return values are double precision. Only the single-precision 
/// floating point version is shown.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/precision.hpp"
#include "../detail/setup.hpp"
#include "../detail/type_mat.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"

namespace glm{
namespace detail
{
	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec2, tvec2>
	{
		typedef tmat2x2<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec2, tvec3>
	{
		typedef tmat2x3<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec2, tvec4>
	{
		typedef tmat2x4<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec3, tvec2>
	{
		typedef tmat3x2<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec3, tvec3>
	{
		typedef tmat3x3<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec3, tvec4>
	{
		typedef tmat3x4<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec4, tvec2>
	{
		typedef tmat4x2<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec4, tvec3>
	{
		typedef tmat4x3<T, P> type;
	};

	template <typename T, precision P>
	struct outerProduct_trait<T, P, tvec4, tvec4>
	{
		typedef tmat4x4<T, P> type;
	};

}//namespace detail

	/// @addtogroup core_func_matrix
	/// @{

	/// Multiply matrix x by matrix y component-wise, i.e., 
	/// result[i][j] is the scalar product of x[i][j] and y[i][j].
	/// 
	/// @tparam matType Floating-point matrix types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/matrixCompMult.xml">GLSL matrixCompMult man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.6 Matrix Functions</a>
	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_DECL matType<T, P> matrixCompMult(matType<T, P> const & x, matType<T, P> const & y);

	/// Treats the first parameter c as a column vector
	/// and the second parameter r as a row vector
	/// and does a linear algebraic matrix multiply c * r.
	/// 
	/// @tparam matType Floating-point matrix types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/outerProduct.xml">GLSL outerProduct man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.6 Matrix Functions</a>
	template <typename T, precision P, template <typename, precision> class vecTypeA, template <typename, precision> class vecTypeB>
	GLM_FUNC_DECL typename detail::outerProduct_trait<T, P, vecTypeA, vecTypeB>::type outerProduct(vecTypeA<T, P> const & c, vecTypeB<T, P> const & r);

	/// Returns the transposed matrix of x
	/// 
	/// @tparam matType Floating-point matrix types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/transpose.xml">GLSL transpose man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.6 Matrix Functions</a>
#	if((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC2012))
		template <typename T, precision P, template <typename, precision> class matType>
		GLM_FUNC_DECL typename matType<T, P>::transpose_type transpose(matType<T, P> const & x);
#	endif
	
	/// Return the determinant of a squared matrix.
	/// 
	/// @tparam valType Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/determinant.xml">GLSL determinant man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.6 Matrix Functions</a>	
	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_DECL T determinant(matType<T, P> const & m);

	/// Return the inverse of a squared matrix.
	/// 
	/// @tparam valType Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/inverse.xml">GLSL inverse man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.6 Matrix Functions</a>	 
	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_DECL matType<T, P> inverse(matType<T, P> const & m);

	/// @}
}//namespace glm

#include "func_matrix.inl"
