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
/// @file glm/detail/func_integer.hpp
/// @date 2010-03-17 / 2011-06-18
/// @author Christophe Riccio
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
/// 
/// @defgroup core_func_integer Integer functions
/// @ingroup core
/// 
/// These all operate component-wise. The description is per component. 
/// The notation [a, b] means the set of bits from bit-number a through bit-number 
/// b, inclusive. The lowest-order bit is bit 0.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "setup.hpp"
#include "precision.hpp"
#include "func_common.hpp"
#include "func_vector_relational.hpp"

namespace glm
{
	/// @addtogroup core_func_integer
	/// @{

	/// Adds 32-bit unsigned integer x and y, returning the sum
	/// modulo pow(2, 32). The value carry is set to 0 if the sum was
	/// less than pow(2, 32), or to 1 otherwise.
	///
	/// @tparam genUType Unsigned integer scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/uaddCarry.xml">GLSL uaddCarry man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<uint, P> uaddCarry(
		vecType<uint, P> const & x,
		vecType<uint, P> const & y,
		vecType<uint, P> & carry);

	/// Subtracts the 32-bit unsigned integer y from x, returning
	/// the difference if non-negative, or pow(2, 32) plus the difference
	/// otherwise. The value borrow is set to 0 if x >= y, or to 1 otherwise.
	///
	/// @tparam genUType Unsigned integer scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/usubBorrow.xml">GLSL usubBorrow man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<uint, P> usubBorrow(
		vecType<uint, P> const & x,
		vecType<uint, P> const & y,
		vecType<uint, P> & borrow);

	/// Multiplies 32-bit integers x and y, producing a 64-bit
	/// result. The 32 least-significant bits are returned in lsb.
	/// The 32 most-significant bits are returned in msb.
	///
	/// @tparam genUType Unsigned integer scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/umulExtended.xml">GLSL umulExtended man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL void umulExtended(
		vecType<uint, P> const & x,
		vecType<uint, P> const & y,
		vecType<uint, P> & msb,
		vecType<uint, P> & lsb);
		
	/// Multiplies 32-bit integers x and y, producing a 64-bit
	/// result. The 32 least-significant bits are returned in lsb.
	/// The 32 most-significant bits are returned in msb.
	/// 
	/// @tparam genIType Signed integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/imulExtended.xml">GLSL imulExtended man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL void imulExtended(
		vecType<int, P> const & x,
		vecType<int, P> const & y,
		vecType<int, P> & msb,
		vecType<int, P> & lsb);

	/// Extracts bits [offset, offset + bits - 1] from value,
	/// returning them in the least significant bits of the result.
	/// For unsigned data types, the most significant bits of the
	/// result will be set to zero. For signed data types, the
	/// most significant bits will be set to the value of bit offset + base - 1.
	///
	/// If bits is zero, the result will be zero. The result will be
	/// undefined if offset or bits is negative, or if the sum of
	/// offset and bits is greater than the number of bits used
	/// to store the operand.
	///
	/// @tparam T Signed or unsigned integer scalar or vector types.
	/// 
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitfieldExtract.xml">GLSL bitfieldExtract man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> bitfieldExtract(
		vecType<T, P> const & Value,
		int Offset,
		int Bits);

	/// Returns the insertion the bits least-significant bits of insert into base.
	///
	/// The result will have bits [offset, offset + bits - 1] taken
	/// from bits [0, bits - 1] of insert, and all other bits taken
	/// directly from the corresponding bits of base. If bits is
	/// zero, the result will simply be base. The result will be
	/// undefined if offset or bits is negative, or if the sum of
	/// offset and bits is greater than the number of bits used to
	/// store the operand.
	///
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitfieldInsert.xml">GLSL bitfieldInsert man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> bitfieldInsert(
		vecType<T, P> const & Base,
		vecType<T, P> const & Insert,
		int Offset,
		int Bits);

	/// Returns the reversal of the bits of value. 
	/// The bit numbered n of the result will be taken from bit (bits - 1) - n of value, 
	/// where bits is the total number of bits used to represent value.
	///
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitfieldReverse.xml">GLSL bitfieldReverse man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<T, P> bitfieldReverse(vecType<T, P> const & v);

	/// Returns the number of bits set to 1 in the binary representation of value.
	///
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitCount.xml">GLSL bitCount man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename genType>
	GLM_FUNC_DECL int bitCount(genType v);

	/// Returns the number of bits set to 1 in the binary representation of value.
	///
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitCount.xml">GLSL bitCount man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<int, P> bitCount(vecType<T, P> const & v);

	/// Returns the bit number of the least significant bit set to
	/// 1 in the binary representation of value. 
	/// If value is zero, -1 will be returned.
	///
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findLSB.xml">GLSL findLSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename genIUType>
	GLM_FUNC_DECL int findLSB(genIUType x);

	/// Returns the bit number of the least significant bit set to
	/// 1 in the binary representation of value. 
	/// If value is zero, -1 will be returned.
	///
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findLSB.xml">GLSL findLSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<int, P> findLSB(vecType<T, P> const & v);

	/// Returns the bit number of the most significant bit in the binary representation of value.
	/// For positive integers, the result will be the bit number of the most significant bit set to 1. 
	/// For negative integers, the result will be the bit number of the most significant
	/// bit set to 0. For a value of zero or negative one, -1 will be returned.
	///
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findMSB.xml">GLSL findMSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename genIUType>
	GLM_FUNC_DECL int findMSB(genIUType x);

	/// Returns the bit number of the most significant bit in the binary representation of value.
	/// For positive integers, the result will be the bit number of the most significant bit set to 1. 
	/// For negative integers, the result will be the bit number of the most significant
	/// bit set to 0. For a value of zero or negative one, -1 will be returned.
	///
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findMSB.xml">GLSL findMSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_DECL vecType<int, P> findMSB(vecType<T, P> const & v);

	/// @}
}//namespace glm

#include "func_integer.inl"
