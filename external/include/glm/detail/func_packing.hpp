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
/// @file glm/detail/func_packing.hpp
/// @date 2010-03-17 / 2011-06-15
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
/// @see gtc_packing
/// 
/// @defgroup core_func_packing Floating-Point Pack and Unpack Functions
/// @ingroup core
/// 
/// These functions do not operate component-wise, rather as described in each case.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "type_vec2.hpp"
#include "type_vec4.hpp"

namespace glm
{
	/// @addtogroup core_func_packing
	/// @{

	/// First, converts each component of the normalized floating-point value v into 8- or 16-bit integer values. 
	/// Then, the results are packed into the returned 32-bit unsigned integer.
	/// 
	/// The conversion for component c of v to fixed point is done as follows:
	/// packUnorm2x16: round(clamp(c, 0, +1) * 65535.0) 
	/// 
	/// The first component of the vector will be written to the least significant bits of the output; 
	/// the last component will be written to the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/packUnorm2x16.xml">GLSL packUnorm2x16 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL uint packUnorm2x16(vec2 const & v);

	/// First, converts each component of the normalized floating-point value v into 8- or 16-bit integer values. 
	/// Then, the results are packed into the returned 32-bit unsigned integer.
	/// 
	/// The conversion for component c of v to fixed point is done as follows:
	/// packSnorm2x16: round(clamp(v, -1, +1) * 32767.0)
	/// 
	/// The first component of the vector will be written to the least significant bits of the output; 
	/// the last component will be written to the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/packSnorm2x16.xml">GLSL packSnorm2x16 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL uint packSnorm2x16(vec2 const & v);

	/// First, converts each component of the normalized floating-point value v into 8- or 16-bit integer values. 
	/// Then, the results are packed into the returned 32-bit unsigned integer.
	/// 
	/// The conversion for component c of v to fixed point is done as follows:
	/// packUnorm4x8:	round(clamp(c, 0, +1) * 255.0)
	/// 
	/// The first component of the vector will be written to the least significant bits of the output; 
	/// the last component will be written to the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/packUnorm4x8.xml">GLSL packUnorm4x8 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL uint packUnorm4x8(vec4 const & v);

	/// First, converts each component of the normalized floating-point value v into 8- or 16-bit integer values. 
	/// Then, the results are packed into the returned 32-bit unsigned integer.
	/// 
	/// The conversion for component c of v to fixed point is done as follows:
	/// packSnorm4x8:	round(clamp(c, -1, +1) * 127.0) 
	/// 
	/// The first component of the vector will be written to the least significant bits of the output; 
	/// the last component will be written to the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/packSnorm4x8.xml">GLSL packSnorm4x8 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL uint packSnorm4x8(vec4 const & v);

	/// First, unpacks a single 32-bit unsigned integer p into a pair of 16-bit unsigned integers, four 8-bit unsigned integers, or four 8-bit signed integers. 
	/// Then, each component is converted to a normalized floating-point value to generate the returned two- or four-component vector.
	/// 
	/// The conversion for unpacked fixed-point value f to floating point is done as follows:
	/// unpackUnorm2x16: f / 65535.0 
	/// 
	/// The first component of the returned vector will be extracted from the least significant bits of the input; 
	/// the last component will be extracted from the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/unpackUnorm2x16.xml">GLSL unpackUnorm2x16 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL vec2 unpackUnorm2x16(uint p);

	/// First, unpacks a single 32-bit unsigned integer p into a pair of 16-bit unsigned integers, four 8-bit unsigned integers, or four 8-bit signed integers. 
	/// Then, each component is converted to a normalized floating-point value to generate the returned two- or four-component vector.
	/// 
	/// The conversion for unpacked fixed-point value f to floating point is done as follows:
	/// unpackSnorm2x16: clamp(f / 32767.0, -1, +1)
	/// 
	/// The first component of the returned vector will be extracted from the least significant bits of the input; 
	/// the last component will be extracted from the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/unpackSnorm2x16.xml">GLSL unpackSnorm2x16 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL vec2 unpackSnorm2x16(uint p);

	/// First, unpacks a single 32-bit unsigned integer p into a pair of 16-bit unsigned integers, four 8-bit unsigned integers, or four 8-bit signed integers. 
	/// Then, each component is converted to a normalized floating-point value to generate the returned two- or four-component vector.
	/// 
	/// The conversion for unpacked fixed-point value f to floating point is done as follows:
	/// unpackUnorm4x8: f / 255.0
	/// 
	/// The first component of the returned vector will be extracted from the least significant bits of the input; 
	/// the last component will be extracted from the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/unpackUnorm4x8.xml">GLSL unpackUnorm4x8 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL vec4 unpackUnorm4x8(uint p);

	/// First, unpacks a single 32-bit unsigned integer p into a pair of 16-bit unsigned integers, four 8-bit unsigned integers, or four 8-bit signed integers. 
	/// Then, each component is converted to a normalized floating-point value to generate the returned two- or four-component vector.
	/// 
	/// The conversion for unpacked fixed-point value f to floating point is done as follows:
	/// unpackSnorm4x8: clamp(f / 127.0, -1, +1)
	/// 
	/// The first component of the returned vector will be extracted from the least significant bits of the input; 
	/// the last component will be extracted from the most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/unpackSnorm4x8.xml">GLSL unpackSnorm4x8 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL vec4 unpackSnorm4x8(uint p);

	/// Returns a double-precision value obtained by packing the components of v into a 64-bit value. 
	/// If an IEEE 754 Inf or NaN is created, it will not signal, and the resulting floating point value is unspecified. 
	/// Otherwise, the bit- level representation of v is preserved. 
	/// The first vector component specifies the 32 least significant bits; 
	/// the second component specifies the 32 most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/packDouble2x32.xml">GLSL packDouble2x32 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL double packDouble2x32(uvec2 const & v);

	/// Returns a two-component unsigned integer vector representation of v. 
	/// The bit-level representation of v is preserved. 
	/// The first component of the vector contains the 32 least significant bits of the double; 
	/// the second component consists the 32 most significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/unpackDouble2x32.xml">GLSL unpackDouble2x32 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL uvec2 unpackDouble2x32(double v);

	/// Returns an unsigned integer obtained by converting the components of a two-component floating-point vector 
	/// to the 16-bit floating-point representation found in the OpenGL Specification, 
	/// and then packing these two 16- bit integers into a 32-bit unsigned integer.
	/// The first vector component specifies the 16 least-significant bits of the result; 
	/// the second component specifies the 16 most-significant bits.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/packHalf2x16.xml">GLSL packHalf2x16 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL uint packHalf2x16(vec2 const & v);
	
	/// Returns a two-component floating-point vector with components obtained by unpacking a 32-bit unsigned integer into a pair of 16-bit values, 
	/// interpreting those values as 16-bit floating-point numbers according to the OpenGL Specification, 
	/// and converting them to 32-bit floating-point values.
	/// The first component of the vector is obtained from the 16 least-significant bits of v; 
	/// the second component is obtained from the 16 most-significant bits of v.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/unpackHalf2x16.xml">GLSL unpackHalf2x16 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.4 Floating-Point Pack and Unpack Functions</a>
	GLM_FUNC_DECL vec2 unpackHalf2x16(uint v);
	
	/// @}
}//namespace glm

#include "func_packing.inl"
