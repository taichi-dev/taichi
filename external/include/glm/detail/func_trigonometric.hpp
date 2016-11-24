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
/// @file glm/detail/func_trigonometric.hpp
/// @date 2008-08-01 / 2011-06-15
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
/// 
/// @defgroup core_func_trigonometric Angle and Trigonometry Functions
/// @ingroup core
/// 
/// Function parameters specified as angle are assumed to be in units of radians. 
/// In no case will any of these functions result in a divide by zero error. If 
/// the divisor of a ratio is 0, then results will be undefined.
/// 
/// These all operate component-wise. The description is per component.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "setup.hpp"
#include "precision.hpp"

namespace glm
{
	/// @addtogroup core_func_trigonometric
	/// @{

	/// Converts degrees to radians and returns the result.
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/radians.xml">GLSL radians man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> radians(vecType<T, P> const & degrees);

	/// Converts radians to degrees and returns the result.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/degrees.xml">GLSL degrees man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> degrees(vecType<T, P> const & radians);

	/// The standard trigonometric sine function. 
	/// The values returned by this function will range from [-1, 1].
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/sin.xml">GLSL sin man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> sin(vecType<T, P> const & angle);

	/// The standard trigonometric cosine function. 
	/// The values returned by this function will range from [-1, 1].
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/cos.xml">GLSL cos man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> cos(vecType<T, P> const & angle);

	/// The standard trigonometric tangent function.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/tan.xml">GLSL tan man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> tan(vecType<T, P> const & angle); 

	/// Arc sine. Returns an angle whose sine is x. 
	/// The range of values returned by this function is [-PI/2, PI/2]. 
	/// Results are undefined if |x| > 1.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/asin.xml">GLSL asin man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> asin(vecType<T, P> const & x);

	/// Arc cosine. Returns an angle whose sine is x. 
	/// The range of values returned by this function is [0, PI]. 
	/// Results are undefined if |x| > 1.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/acos.xml">GLSL acos man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> acos(vecType<T, P> const & x);

	/// Arc tangent. Returns an angle whose tangent is y/x. 
	/// The signs of x and y are used to determine what 
	/// quadrant the angle is in. The range of values returned 
	/// by this function is [-PI, PI]. Results are undefined 
	/// if x and y are both 0. 
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/atan.xml">GLSL atan man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> atan(vecType<T, P> const & y, vecType<T, P> const & x);

	/// Arc tangent. Returns an angle whose tangent is y_over_x. 
	/// The range of values returned by this function is [-PI/2, PI/2].
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/atan.xml">GLSL atan man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> atan(vecType<T, P> const & y_over_x);

	/// Returns the hyperbolic sine function, (exp(x) - exp(-x)) / 2
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/sinh.xml">GLSL sinh man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> sinh(vecType<T, P> const & angle);

	/// Returns the hyperbolic cosine function, (exp(x) + exp(-x)) / 2
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/cosh.xml">GLSL cosh man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> cosh(vecType<T, P> const & angle);

	/// Returns the hyperbolic tangent function, sinh(angle) / cosh(angle)
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/tanh.xml">GLSL tanh man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> tanh(vecType<T, P> const & angle);

	/// Arc hyperbolic sine; returns the inverse of sinh.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/asinh.xml">GLSL asinh man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> asinh(vecType<T, P> const & x);
	
	/// Arc hyperbolic cosine; returns the non-negative inverse
	/// of cosh. Results are undefined if x < 1.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/acosh.xml">GLSL acosh man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> acosh(vecType<T, P> const & x);

	/// Arc hyperbolic tangent; returns the inverse of tanh.
	/// Results are undefined if abs(x) >= 1.
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/atanh.xml">GLSL atanh man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.1 Angle and Trigonometry Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> atanh(vecType<T, P> const & x);

	/// @}
}//namespace glm

#include "func_trigonometric.inl"
