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
/// @ref gtc_type_precision
/// @file glm/gtc/type_precision.hpp
/// @date 2009-06-04 / 2011-12-07
/// @author Christophe Riccio
/// 
/// @see core (dependence)
/// @see gtc_half_float (dependence)
/// @see gtc_quaternion (dependence)
/// 
/// @defgroup gtc_type_precision GLM_GTC_type_precision
/// @ingroup gtc
/// 
/// @brief Defines specific C++-based precision types.
/// 
/// @ref core_precision defines types based on GLSL's precision qualifiers. This
/// extension defines types based on explicitly-sized C++ data types.
/// 
/// <glm/gtc/type_precision.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../gtc/quaternion.hpp"
#include "../gtc/vec1.hpp"
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

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_type_precision extension included")
#endif

namespace glm
{
	///////////////////////////
	// Signed int vector types 

	/// @addtogroup gtc_type_precision
	/// @{

	/// Low precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 lowp_int8;
	
	/// Low precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 lowp_int16;

	/// Low precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 lowp_int32;

	/// Low precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 lowp_int64;

	/// Low precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 lowp_int8_t;
	
	/// Low precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 lowp_int16_t;

	/// Low precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 lowp_int32_t;

	/// Low precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 lowp_int64_t;

	/// Low precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 lowp_i8;
	
	/// Low precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 lowp_i16;

	/// Low precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 lowp_i32;

	/// Low precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 lowp_i64;

	/// Medium precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 mediump_int8;
	
	/// Medium precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 mediump_int16;

	/// Medium precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 mediump_int32;

	/// Medium precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 mediump_int64;

	/// Medium precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 mediump_int8_t;
	
	/// Medium precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 mediump_int16_t;

	/// Medium precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 mediump_int32_t;

	/// Medium precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 mediump_int64_t;

	/// Medium precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 mediump_i8;
	
	/// Medium precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 mediump_i16;

	/// Medium precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 mediump_i32;

	/// Medium precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 mediump_i64;

	/// High precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 highp_int8;
	
	/// High precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 highp_int16;

	/// High precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 highp_int32;

	/// High precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 highp_int64;

	/// High precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 highp_int8_t;
	
	/// High precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 highp_int16_t;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 highp_int32_t;

	/// High precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 highp_int64_t;

	/// High precision 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 highp_i8;
	
	/// High precision 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 highp_i16;

	/// High precision 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 highp_i32;

	/// High precision 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 highp_i64;
	

	/// 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 int8;
	
	/// 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 int16;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 int32;

	/// 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 int64;

#if GLM_HAS_EXTENDED_INTEGER_TYPE
	using std::int8_t;
	using std::int16_t;
	using std::int32_t;
	using std::int64_t;
#else
	/// 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 int8_t;
	
	/// 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 int16_t;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 int32_t;

	/// 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 int64_t;
#endif

	/// 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 i8;
	
	/// 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 i16;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 i32;

	/// 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 i64;


	/// 8 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<i8, defaultp> i8vec1;
	
	/// 8 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<i8, defaultp> i8vec2;

	/// 8 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<i8, defaultp> i8vec3;

	/// 8 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<i8, defaultp> i8vec4;


	/// 16 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<i16, defaultp> i16vec1;
	
	/// 16 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<i16, defaultp> i16vec2;

	/// 16 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<i16, defaultp> i16vec3;

	/// 16 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<i16, defaultp> i16vec4;


	/// 32 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<i32, defaultp> i32vec1;
	
	/// 32 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<i32, defaultp> i32vec2;

	/// 32 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<i32, defaultp> i32vec3;

	/// 32 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<i32, defaultp> i32vec4;


	/// 64 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<i64, defaultp> i64vec1;
	
	/// 64 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<i64, defaultp> i64vec2;

	/// 64 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<i64, defaultp> i64vec3;

	/// 64 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<i64, defaultp> i64vec4;


	/////////////////////////////
	// Unsigned int vector types

	/// Low precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 lowp_uint8;
	
	/// Low precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 lowp_uint16;

	/// Low precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 lowp_uint32;

	/// Low precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 lowp_uint64;

	/// Low precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 lowp_uint8_t;
	
	/// Low precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 lowp_uint16_t;

	/// Low precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 lowp_uint32_t;

	/// Low precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 lowp_uint64_t;

	/// Low precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 lowp_u8;
	
	/// Low precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 lowp_u16;

	/// Low precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 lowp_u32;

	/// Low precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 lowp_u64;
	
	/// Medium precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 mediump_uint8;
	
	/// Medium precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 mediump_uint16;

	/// Medium precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 mediump_uint32;

	/// Medium precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 mediump_uint64;

	/// Medium precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 mediump_uint8_t;
	
	/// Medium precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 mediump_uint16_t;

	/// Medium precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 mediump_uint32_t;

	/// Medium precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 mediump_uint64_t;

	/// Medium precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 mediump_u8;
	
	/// Medium precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 mediump_u16;

	/// Medium precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 mediump_u32;

	/// Medium precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 mediump_u64;
	
	/// High precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 highp_uint8;
	
	/// High precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 highp_uint16;

	/// High precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 highp_uint32;

	/// High precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 highp_uint64;

	/// High precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 highp_uint8_t;
	
	/// High precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 highp_uint16_t;

	/// High precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 highp_uint32_t;

	/// High precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 highp_uint64_t;

	/// High precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 highp_u8;
	
	/// High precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 highp_u16;

	/// High precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 highp_u32;

	/// High precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 highp_u64;

	/// Default precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 uint8;
	
	/// Default precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 uint16;

	/// Default precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 uint32;

	/// Default precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 uint64;

#if GLM_HAS_EXTENDED_INTEGER_TYPE
	using std::uint8_t;
	using std::uint16_t;
	using std::uint32_t;
	using std::uint64_t;
#else
	/// Default precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 uint8_t;
	
	/// Default precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 uint16_t;

	/// Default precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 uint32_t;

	/// Default precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 uint64_t;
#endif

	/// Default precision 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 u8;
	
	/// Default precision 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 u16;

	/// Default precision 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 u32;

	/// Default precision 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 u64;



	/// Default precision 8 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<u8, defaultp> u8vec1;
	
	/// Default precision 8 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<u8, defaultp> u8vec2;

	/// Default precision 8 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<u8, defaultp> u8vec3;

	/// Default precision 8 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<u8, defaultp> u8vec4;


	/// Default precision 16 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<u16, defaultp> u16vec1;
	
	/// Default precision 16 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<u16, defaultp> u16vec2;

	/// Default precision 16 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<u16, defaultp> u16vec3;

	/// Default precision 16 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<u16, defaultp> u16vec4;


	/// Default precision 32 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<u32, defaultp> u32vec1;
	
	/// Default precision 32 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<u32, defaultp> u32vec2;

	/// Default precision 32 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<u32, defaultp> u32vec3;

	/// Default precision 32 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<u32, defaultp> u32vec4;


	/// Default precision 64 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef tvec1<u64, defaultp> u64vec1;
	
	/// Default precision 64 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef tvec2<u64, defaultp> u64vec2;

	/// Default precision 64 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef tvec3<u64, defaultp> u64vec3;

	/// Default precision 64 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef tvec4<u64, defaultp> u64vec4;


	//////////////////////
	// Float vector types

	/// 32 bit single-precision floating-point scalar.
	/// @see gtc_type_precision
	typedef detail::float32 float32;

	/// 64 bit double-precision floating-point scalar.
	/// @see gtc_type_precision
	typedef detail::float64 float64;


	/// 32 bit single-precision floating-point scalar.
	/// @see gtc_type_precision
	typedef detail::float32 float32_t;

	/// 64 bit double-precision floating-point scalar.
	/// @see gtc_type_precision
	typedef detail::float64 float64_t;


	/// 32 bit single-precision floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 f32;

	/// 64 bit double-precision floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 f64;


	/// Single-precision floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef tvec1<float, defaultp> fvec1;

	/// Single-precision floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef tvec2<float, defaultp> fvec2;

	/// Single-precision floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef tvec3<float, defaultp> fvec3;

	/// Single-precision floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef tvec4<float, defaultp> fvec4;

	
	/// Single-precision floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef tvec1<f32, defaultp> f32vec1;

	/// Single-precision floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef tvec2<f32, defaultp> f32vec2;

	/// Single-precision floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef tvec3<f32, defaultp> f32vec3;

	/// Single-precision floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef tvec4<f32, defaultp> f32vec4;


	/// Double-precision floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef tvec1<f64, defaultp> f64vec1;

	/// Double-precision floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef tvec2<f64, defaultp> f64vec2;

	/// Double-precision floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef tvec3<f64, defaultp> f64vec3;

	/// Double-precision floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef tvec4<f64, defaultp> f64vec4;


	//////////////////////
	// Float matrix types 

	/// Single-precision floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32> fmat1;

	/// Single-precision floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef tmat2x2<f32, defaultp> fmat2;

	/// Single-precision floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef tmat3x3<f32, defaultp> fmat3;

	/// Single-precision floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef tmat4x4<f32, defaultp> fmat4;


	/// Single-precision floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 fmat1x1;

	/// Single-precision floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef tmat2x2<f32, defaultp> fmat2x2;

	/// Single-precision floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef tmat2x3<f32, defaultp> fmat2x3;

	/// Single-precision floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef tmat2x4<f32, defaultp> fmat2x4;

	/// Single-precision floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef tmat3x2<f32, defaultp> fmat3x2;

	/// Single-precision floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef tmat3x3<f32, defaultp> fmat3x3;

	/// Single-precision floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef tmat3x4<f32, defaultp> fmat3x4;

	/// Single-precision floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef tmat4x2<f32, defaultp> fmat4x2;

	/// Single-precision floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef tmat4x3<f32, defaultp> fmat4x3;

	/// Single-precision floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef tmat4x4<f32, defaultp> fmat4x4;


	/// Single-precision floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32, defaultp> f32mat1;

	/// Single-precision floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef tmat2x2<f32, defaultp> f32mat2;

	/// Single-precision floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef tmat3x3<f32, defaultp> f32mat3;

	/// Single-precision floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef tmat4x4<f32, defaultp> f32mat4;


	/// Single-precision floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 f32mat1x1;

	/// Single-precision floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef tmat2x2<f32, defaultp> f32mat2x2;

	/// Single-precision floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef tmat2x3<f32, defaultp> f32mat2x3;

	/// Single-precision floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef tmat2x4<f32, defaultp> f32mat2x4;

	/// Single-precision floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef tmat3x2<f32, defaultp> f32mat3x2;

	/// Single-precision floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef tmat3x3<f32, defaultp> f32mat3x3;

	/// Single-precision floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef tmat3x4<f32, defaultp> f32mat3x4;

	/// Single-precision floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef tmat4x2<f32, defaultp> f32mat4x2;

	/// Single-precision floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef tmat4x3<f32, defaultp> f32mat4x3;

	/// Single-precision floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef tmat4x4<f32, defaultp> f32mat4x4;


	/// Double-precision floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f64, defaultp> f64mat1;

	/// Double-precision floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef tmat2x2<f64, defaultp> f64mat2;

	/// Double-precision floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef tmat3x3<f64, defaultp> f64mat3;

	/// Double-precision floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef tmat4x4<f64, defaultp> f64mat4;


	/// Double-precision floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f64 f64mat1x1;

	/// Double-precision floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef tmat2x2<f64, defaultp> f64mat2x2;

	/// Double-precision floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef tmat2x3<f64, defaultp> f64mat2x3;

	/// Double-precision floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef tmat2x4<f64, defaultp> f64mat2x4;

	/// Double-precision floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef tmat3x2<f64, defaultp> f64mat3x2;

	/// Double-precision floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef tmat3x3<f64, defaultp> f64mat3x3;

	/// Double-precision floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef tmat3x4<f64, defaultp> f64mat3x4;

	/// Double-precision floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef tmat4x2<f64, defaultp> f64mat4x2;

	/// Double-precision floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef tmat4x3<f64, defaultp> f64mat4x3;

	/// Double-precision floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef tmat4x4<f64, defaultp> f64mat4x4;


	//////////////////////////
	// Quaternion types

	/// Single-precision floating-point quaternion.
	/// @see gtc_type_precision
	typedef tquat<f32, defaultp> f32quat;

	/// Double-precision floating-point quaternion.
	/// @see gtc_type_precision
	typedef tquat<f64, defaultp> f64quat;

	/// @}
}//namespace glm

#include "type_precision.inl"
