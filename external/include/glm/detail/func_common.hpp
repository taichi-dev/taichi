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
/// @file glm/detail/func_common.hpp
/// @date 2008-03-08 / 2010-01-26
/// @author Christophe Riccio
/// 
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
///
/// @defgroup core_func_common Common functions
/// @ingroup core
/// 
/// These all operate component-wise. The description is per component.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "setup.hpp"
#include "precision.hpp"
#include "type_int.hpp"
#include "_fixes.hpp"

namespace glm
{
	/// @addtogroup core_func_common
	/// @{

	/// Returns x if x >= 0; otherwise, it returns -x.
	/// 
	/// @tparam genType floating-point or signed integer; scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/abs.xml">GLSL abs man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType abs(genType x);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> abs(vecType<T, P> const & x);

	/// Returns 1.0 if x > 0, 0.0 if x == 0, or -1.0 if x < 0. 
	/// 
	/// @tparam genType Floating-point or signed integer; scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/sign.xml">GLSL sign man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> sign(vecType<T, P> const & x);

	/// Returns a value equal to the nearest integer that is less then or equal to x. 
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/floor.xml">GLSL floor man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> floor(vecType<T, P> const & x);

	/// Returns a value equal to the nearest integer to x
	/// whose absolute value is not larger than the absolute value of x.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/trunc.xml">GLSL trunc man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> trunc(vecType<T, P> const & x);

	/// Returns a value equal to the nearest integer to x.
	/// The fraction 0.5 will round in a direction chosen by the
	/// implementation, presumably the direction that is fastest.
	/// This includes the possibility that round(x) returns the
	/// same value as roundEven(x) for all values of x.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/round.xml">GLSL round man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> round(vecType<T, P> const & x);

	/// Returns a value equal to the nearest integer to x.
	/// A fractional part of 0.5 will round toward the nearest even
	/// integer. (Both 3.5 and 4.5 for x will return 4.0.)
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/roundEven.xml">GLSL roundEven man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	/// @see <a href="http://developer.amd.com/documentation/articles/pages/New-Round-to-Even-Technique.aspx">New round to even technique</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> roundEven(vecType<T, P> const & x);

	/// Returns a value equal to the nearest integer
	/// that is greater than or equal to x.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/ceil.xml">GLSL ceil man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> ceil(vecType<T, P> const & x);

	/// Return x - floor(x).
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/fract.xml">GLSL fract man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType fract(genType x);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> fract(vecType<T, P> const & x);

	/// Modulus. Returns x - y * floor(x / y)
	/// for each component in x using the floating point value y.
	///
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/mod.xml">GLSL mod man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType mod(genType x, genType y);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> mod(vecType<T, P> const & x, T y);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> mod(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns the fractional part of x and sets i to the integer
	/// part (as a whole number floating point value). Both the
	/// return value and the output parameter will have the same
	/// sign as x.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/modf.xml">GLSL modf man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType modf(genType x, genType & i);

	/// Returns y if y < x; otherwise, it returns x.
	///
	/// @tparam genType Floating-point or integer; scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/min.xml">GLSL min man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType min(genType x, genType y);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> min(vecType<T, P> const & x, T y);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> min(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns y if x < y; otherwise, it returns x.
	/// 
	/// @tparam genType Floating-point or integer; scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/max.xml">GLSL max man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType max(genType x, genType y);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> max(vecType<T, P> const & x, T y);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> max(vecType<T, P> const & x, vecType<T, P> const & y);

	/// Returns min(max(x, minVal), maxVal) for each component in x 
	/// using the floating-point values minVal and maxVal.
	///
	/// @tparam genType Floating-point or integer; scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/clamp.xml">GLSL clamp man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType clamp(genType x, genType minVal, genType maxVal);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> clamp(vecType<T, P> const & x, T minVal, T maxVal);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> clamp(vecType<T, P> const & x, vecType<T, P> const & minVal, vecType<T, P> const & maxVal);

	/// If genTypeU is a floating scalar or vector:
	/// Returns x * (1.0 - a) + y * a, i.e., the linear blend of
	/// x and y using the floating-point value a.
	/// The value for a is not restricted to the range [0, 1].
	/// 
	/// If genTypeU is a boolean scalar or vector:
	/// Selects which vector each returned component comes
	/// from. For a component of <a> that is false, the
	/// corresponding component of x is returned. For a
	/// component of a that is true, the corresponding
	/// component of y is returned. Components of x and y that
	/// are not selected are allowed to be invalid floating point
	/// values and will have no effect on the results. Thus, this
	/// provides different functionality than
	/// genType mix(genType x, genType y, genType(a))
	/// where a is a Boolean vector.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/mix.xml">GLSL mix man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	/// 
	/// @param[in]  x Value to interpolate.
	/// @param[in]  y Value to interpolate.
	/// @param[in]  a Interpolant.
	/// 
	/// @tparam	genTypeT Floating point scalar or vector.
	/// @tparam genTypeU Floating point or boolean scalar or vector. It can't be a vector if it is the length of genTypeT.
	/// 
	/// @code
	/// #include <glm/glm.hpp>
	/// ...
	/// float a;
	/// bool b;
	/// glm::dvec3 e;
	/// glm::dvec3 f;
	/// glm::vec4 g;
	/// glm::vec4 h;
	/// ...
	/// glm::vec4 r = glm::mix(g, h, a); // Interpolate with a floating-point scalar two vectors. 
	/// glm::vec4 s = glm::mix(g, h, b); // Teturns g or h;
	/// glm::dvec3 t = glm::mix(e, f, a); // Types of the third parameter is not required to match with the first and the second.
	/// glm::vec4 u = glm::mix(g, h, r); // Interpolations can be perform per component with a vector for the last parameter.
	/// @endcode
	template <typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> mix(vecType<T, P> const & x, vecType<T, P> const & y, vecType<U, P> const & a);

	template <typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> mix(vecType<T, P> const & x, vecType<T, P> const & y, U a);

	template <typename genTypeT, typename genTypeU>
	GLM_FUNC_DECL genTypeT mix(genTypeT x, genTypeT y, genTypeU a);

	/// Returns 0.0 if x < edge, otherwise it returns 1.0 for each component of a genType.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/step.xml">GLSL step man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType step(genType edge, genType x);

	/// Returns 0.0 if x < edge, otherwise it returns 1.0.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/step.xml">GLSL step man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <template <typename, precision> class vecType, typename T, precision P>
	GLM_FUNC_DECL vecType<T, P> step(T edge, vecType<T, P> const & x);

	/// Returns 0.0 if x < edge, otherwise it returns 1.0.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/step.xml">GLSL step man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <template <typename, precision> class vecType, typename T, precision P>
	GLM_FUNC_DECL vecType<T, P> step(vecType<T, P> const & edge, vecType<T, P> const & x);

	/// Returns 0.0 if x <= edge0 and 1.0 if x >= edge1 and
	/// performs smooth Hermite interpolation between 0 and 1
	/// when edge0 < x < edge1. This is useful in cases where
	/// you would want a threshold function with a smooth
	/// transition. This is equivalent to:
	/// genType t;
	/// t = clamp ((x - edge0) / (edge1 - edge0), 0, 1);
	/// return t * t * (3 - 2 * t);
	/// Results are undefined if edge0 >= edge1.
	///
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/smoothstep.xml">GLSL smoothstep man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType smoothstep(genType edge0, genType edge1, genType x);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> smoothstep(T edge0, T edge1, vecType<T, P> const & x);

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> smoothstep(vecType<T, P> const & edge0, vecType<T, P> const & edge1, vecType<T, P> const & x);

	/// Returns true if x holds a NaN (not a number)
	/// representation in the underlying implementation's set of
	/// floating point representations. Returns false otherwise,
	/// including for implementations with no NaN
	/// representations.
	/// 
	/// /!\ When using compiler fast math, this function may fail.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/isnan.xml">GLSL isnan man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> isnan(vecType<T, P> const & x);

	/// Returns true if x holds a positive infinity or negative
	/// infinity representation in the underlying implementation's
	/// set of floating point representations. Returns false
	/// otherwise, including for implementations with no infinity
	/// representations.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/isinf.xml">GLSL isinf man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<bool, P> isinf(vecType<T, P> const & x);

	/// Returns a signed integer value representing
	/// the encoding of a floating-point value. The floating-point
	/// value's bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/floatBitsToInt.xml">GLSL floatBitsToInt man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	GLM_FUNC_DECL int floatBitsToInt(float const & v);

	/// Returns a signed integer value representing
	/// the encoding of a floating-point value. The floatingpoint
	/// value's bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/floatBitsToInt.xml">GLSL floatBitsToInt man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_DECL vecType<int, P> floatBitsToInt(vecType<float, P> const & v);

	/// Returns a unsigned integer value representing
	/// the encoding of a floating-point value. The floatingpoint
	/// value's bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/floatBitsToUint.xml">GLSL floatBitsToUint man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	GLM_FUNC_DECL uint floatBitsToUint(float const & v);

	/// Returns a unsigned integer value representing
	/// the encoding of a floating-point value. The floatingpoint
	/// value's bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/floatBitsToUint.xml">GLSL floatBitsToUint man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_DECL vecType<uint, P> floatBitsToUint(vecType<float, P> const & v);

	/// Returns a floating-point value corresponding to a signed
	/// integer encoding of a floating-point value.
	/// If an inf or NaN is passed in, it will not signal, and the
	/// resulting floating point value is unspecified. Otherwise,
	/// the bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/intBitsToFloat.xml">GLSL intBitsToFloat man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	GLM_FUNC_DECL float intBitsToFloat(int const & v);

	/// Returns a floating-point value corresponding to a signed
	/// integer encoding of a floating-point value.
	/// If an inf or NaN is passed in, it will not signal, and the
	/// resulting floating point value is unspecified. Otherwise,
	/// the bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/intBitsToFloat.xml">GLSL intBitsToFloat man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_DECL vecType<float, P> intBitsToFloat(vecType<int, P> const & v);

	/// Returns a floating-point value corresponding to a
	/// unsigned integer encoding of a floating-point value.
	/// If an inf or NaN is passed in, it will not signal, and the
	/// resulting floating point value is unspecified. Otherwise,
	/// the bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/uintBitsToFloat.xml">GLSL uintBitsToFloat man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	GLM_FUNC_DECL float uintBitsToFloat(uint const & v);

	/// Returns a floating-point value corresponding to a
	/// unsigned integer encoding of a floating-point value.
	/// If an inf or NaN is passed in, it will not signal, and the
	/// resulting floating point value is unspecified. Otherwise,
	/// the bit-level representation is preserved.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/uintBitsToFloat.xml">GLSL uintBitsToFloat man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_DECL vecType<float, P> uintBitsToFloat(vecType<uint, P> const & v);

	/// Computes and returns a * b + c.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/fma.xml">GLSL fma man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType>
	GLM_FUNC_DECL genType fma(genType const & a, genType const & b, genType const & c);

	/// Splits x into a floating-point significand in the range
	/// [0.5, 1.0) and an integral exponent of two, such that:
	/// x = significand * exp(2, exponent)
	/// 
	/// The significand is returned by the function and the
	/// exponent is returned in the parameter exp. For a
	/// floating-point value of zero, the significant and exponent
	/// are both zero. For a floating-point value that is an
	/// infinity or is not a number, the results are undefined.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/frexp.xml">GLSL frexp man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType, typename genIType>
	GLM_FUNC_DECL genType frexp(genType const & x, genIType & exp);

	/// Builds a floating-point number from x and the
	/// corresponding integral exponent of two in exp, returning:
	/// significand * exp(2, exponent)
	/// 
	/// If this product is too large to be represented in the
	/// floating-point type, the result is undefined.
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	///  
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/ldexp.xml">GLSL ldexp man page</a>; 
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.3 Common Functions</a>
	template <typename genType, typename genIType>
	GLM_FUNC_DECL genType ldexp(genType const & x, genIType const & exp);

	/// @}
}//namespace glm

#include "func_common.inl"

