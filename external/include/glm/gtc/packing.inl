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
/// @ref gtc_packing
/// @file glm/gtc/packing.inl
/// @date 2013-08-08 / 2013-08-08
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "../common.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../detail/type_half.hpp"
#include <cstring>

namespace glm{
namespace detail
{
	GLM_FUNC_QUALIFIER glm::uint16 float2half(glm::uint32 f)
	{
		// 10 bits    =>                         EE EEEFFFFF
		// 11 bits    =>                        EEE EEFFFFFF
		// Half bits  =>                   SEEEEEFF FFFFFFFF
		// Float bits => SEEEEEEE EFFFFFFF FFFFFFFF FFFFFFFF

		// 0x00007c00 => 00000000 00000000 01111100 00000000
		// 0x000003ff => 00000000 00000000 00000011 11111111
		// 0x38000000 => 00111000 00000000 00000000 00000000
		// 0x7f800000 => 01111111 10000000 00000000 00000000
		// 0x00008000 => 00000000 00000000 10000000 00000000
		return
			((f >> 16) & 0x8000) | // sign
			((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | // exponential
			((f >> 13) & 0x03ff); // Mantissa
	}

	GLM_FUNC_QUALIFIER glm::uint32 float2packed11(glm::uint32 f)
	{
		// 10 bits    =>                         EE EEEFFFFF
		// 11 bits    =>                        EEE EEFFFFFF
		// Half bits  =>                   SEEEEEFF FFFFFFFF
		// Float bits => SEEEEEEE EFFFFFFF FFFFFFFF FFFFFFFF

		// 0x000007c0 => 00000000 00000000 00000111 11000000
		// 0x00007c00 => 00000000 00000000 01111100 00000000
		// 0x000003ff => 00000000 00000000 00000011 11111111
		// 0x38000000 => 00111000 00000000 00000000 00000000
		// 0x7f800000 => 01111111 10000000 00000000 00000000
		// 0x00008000 => 00000000 00000000 10000000 00000000
		return
			((((f & 0x7f800000) - 0x38000000) >> 17) & 0x07c0) | // exponential
			((f >> 17) & 0x003f); // Mantissa
	}

	GLM_FUNC_QUALIFIER glm::uint32 packed11ToFloat(glm::uint32 p)
	{
		// 10 bits    =>                         EE EEEFFFFF
		// 11 bits    =>                        EEE EEFFFFFF
		// Half bits  =>                   SEEEEEFF FFFFFFFF
		// Float bits => SEEEEEEE EFFFFFFF FFFFFFFF FFFFFFFF

		// 0x000007c0 => 00000000 00000000 00000111 11000000
		// 0x00007c00 => 00000000 00000000 01111100 00000000
		// 0x000003ff => 00000000 00000000 00000011 11111111
		// 0x38000000 => 00111000 00000000 00000000 00000000
		// 0x7f800000 => 01111111 10000000 00000000 00000000
		// 0x00008000 => 00000000 00000000 10000000 00000000
		return
			((((p & 0x07c0) << 17) + 0x38000000) & 0x7f800000) | // exponential
			((p & 0x003f) << 17); // Mantissa
	}

	GLM_FUNC_QUALIFIER glm::uint32 float2packed10(glm::uint32 f)
	{
		// 10 bits    =>                         EE EEEFFFFF
		// 11 bits    =>                        EEE EEFFFFFF
		// Half bits  =>                   SEEEEEFF FFFFFFFF
		// Float bits => SEEEEEEE EFFFFFFF FFFFFFFF FFFFFFFF

		// 0x0000001F => 00000000 00000000 00000000 00011111
		// 0x0000003F => 00000000 00000000 00000000 00111111
		// 0x000003E0 => 00000000 00000000 00000011 11100000
		// 0x000007C0 => 00000000 00000000 00000111 11000000
		// 0x00007C00 => 00000000 00000000 01111100 00000000
		// 0x000003FF => 00000000 00000000 00000011 11111111
		// 0x38000000 => 00111000 00000000 00000000 00000000
		// 0x7f800000 => 01111111 10000000 00000000 00000000
		// 0x00008000 => 00000000 00000000 10000000 00000000
		return
			((((f & 0x7f800000) - 0x38000000) >> 18) & 0x03E0) | // exponential
			((f >> 18) & 0x001f); // Mantissa
	}

	GLM_FUNC_QUALIFIER glm::uint32 packed10ToFloat(glm::uint32 p)
	{
		// 10 bits    =>                         EE EEEFFFFF
		// 11 bits    =>                        EEE EEFFFFFF
		// Half bits  =>                   SEEEEEFF FFFFFFFF
		// Float bits => SEEEEEEE EFFFFFFF FFFFFFFF FFFFFFFF

		// 0x0000001F => 00000000 00000000 00000000 00011111
		// 0x0000003F => 00000000 00000000 00000000 00111111
		// 0x000003E0 => 00000000 00000000 00000011 11100000
		// 0x000007C0 => 00000000 00000000 00000111 11000000
		// 0x00007C00 => 00000000 00000000 01111100 00000000
		// 0x000003FF => 00000000 00000000 00000011 11111111
		// 0x38000000 => 00111000 00000000 00000000 00000000
		// 0x7f800000 => 01111111 10000000 00000000 00000000
		// 0x00008000 => 00000000 00000000 10000000 00000000
		return
			((((p & 0x03E0) << 18) + 0x38000000) & 0x7f800000) | // exponential
			((p & 0x001f) << 18); // Mantissa
	}

	GLM_FUNC_QUALIFIER glm::uint half2float(glm::uint h)
	{
		return ((h & 0x8000) << 16) | ((( h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);
	}

	GLM_FUNC_QUALIFIER glm::uint floatTo11bit(float x)
	{
		if(x == 0.0f)
			return 0u;
		else if(glm::isnan(x))
			return ~0u;
		else if(glm::isinf(x))
			return 0x1Fu << 6u;

#		if(GLM_COMPILER & GLM_COMPILER_GCC || GLM_COMPILER & (GLM_COMPILER_APPLE_CLANG | GLM_COMPILER_LLVM))
			uint Pack = 0u;
			memcpy(&Pack, &x, sizeof(Pack));
#		else	
			uint Pack = reinterpret_cast<uint&>(x);
#		endif

		return float2packed11(Pack);
	}

	GLM_FUNC_QUALIFIER float packed11bitToFloat(glm::uint x)
	{
		if(x == 0)
			return 0.0f;
		else if(x == ((1 << 11) - 1))
			return ~0;//NaN
		else if(x == (0x1f << 6))
			return ~0;//Inf

		uint Result = packed11ToFloat(x);

#		if(GLM_COMPILER & GLM_COMPILER_GCC || GLM_COMPILER & (GLM_COMPILER_APPLE_CLANG | GLM_COMPILER_LLVM))
			float Temp = 0;
			memcpy(&Temp, &Result, sizeof(Temp));
			return Temp;
#		else	
			return reinterpret_cast<float&>(Result);
#		endif
	}

	GLM_FUNC_QUALIFIER glm::uint floatTo10bit(float x)
	{
		if(x == 0.0f)
			return 0u;
		else if(glm::isnan(x))
			return ~0u;
		else if(glm::isinf(x))
			return 0x1Fu << 5u;

#		if(GLM_COMPILER & GLM_COMPILER_GCC || GLM_COMPILER & (GLM_COMPILER_APPLE_CLANG | GLM_COMPILER_LLVM))
			uint Pack = 0;
			memcpy(&Pack, &x, sizeof(Pack));
#		else	
			uint Pack = reinterpret_cast<uint&>(x);
#		endif

		return float2packed10(Pack);
	}

	GLM_FUNC_QUALIFIER float packed10bitToFloat(glm::uint x)
	{
		if(x == 0)
			return 0.0f;
		else if(x == ((1 << 10) - 1))
			return ~0;//NaN
		else if(x == (0x1f << 5))
			return ~0;//Inf

		uint Result = packed10ToFloat(x);

#		if(GLM_COMPILER & GLM_COMPILER_GCC || GLM_COMPILER & (GLM_COMPILER_APPLE_CLANG | GLM_COMPILER_LLVM))
			float Temp = 0;
			memcpy(&Temp, &Result, sizeof(Temp));
			return Temp;
#		else	
			return reinterpret_cast<float&>(Result);
#		endif
	}

//	GLM_FUNC_QUALIFIER glm::uint f11_f11_f10(float x, float y, float z)
//	{
//		return ((floatTo11bit(x) & ((1 << 11) - 1)) << 0) |  ((floatTo11bit(y) & ((1 << 11) - 1)) << 11) | ((floatTo10bit(z) & ((1 << 10) - 1)) << 22);
//	}

	union u10u10u10u2
	{
		struct
		{
			uint x : 10;
			uint y : 10;
			uint z : 10;
			uint w : 2;
		} data;
		uint32 pack;
	};

	union i10i10i10i2
	{
		struct
		{
			int x : 10;
			int y : 10;
			int z : 10;
			int w : 2;
		} data;
		uint32 pack;
	};

}//namespace detail

	GLM_FUNC_QUALIFIER uint8 packUnorm1x8(float v)
	{
		return static_cast<uint8>(round(clamp(v, 0.0f, 1.0f) * 255.0f));
	}
	
	GLM_FUNC_QUALIFIER float unpackUnorm1x8(uint8 p)
	{
		float const Unpack(p);
		return Unpack * static_cast<float>(0.0039215686274509803921568627451); // 1 / 255
	}
	
	GLM_FUNC_QUALIFIER uint16 packUnorm2x8(vec2 const & v)
	{
		u8vec2 const Topack(round(clamp(v, 0.0f, 1.0f) * 255.0f));
		return reinterpret_cast<uint16 const &>(Topack);
	}
	
	GLM_FUNC_QUALIFIER vec2 unpackUnorm2x8(uint16 p)
	{
		vec2 const Unpack(reinterpret_cast<u8vec2 const &>(p));
		return Unpack * float(0.0039215686274509803921568627451); // 1 / 255
	}

	GLM_FUNC_QUALIFIER uint8 packSnorm1x8(float v)
	{
		int8 const Topack(static_cast<int8>(round(clamp(v ,-1.0f, 1.0f) * 127.0f)));
		return reinterpret_cast<uint8 const &>(Topack);
	}
	
	GLM_FUNC_QUALIFIER float unpackSnorm1x8(uint8 p)
	{
		float const Unpack(reinterpret_cast<int8 const &>(p));
		return clamp(
			Unpack * 0.00787401574803149606299212598425f, // 1.0f / 127.0f
			-1.0f, 1.0f);
	}
	
	GLM_FUNC_QUALIFIER uint16 packSnorm2x8(vec2 const & v)
	{
		i8vec2 const Topack(round(clamp(v, -1.0f, 1.0f) * 127.0f));
		return reinterpret_cast<uint16 const &>(Topack);
	}
	
	GLM_FUNC_QUALIFIER vec2 unpackSnorm2x8(uint16 p)
	{
		vec2 const Unpack(reinterpret_cast<i8vec2 const &>(p));
		return clamp(
			Unpack * 0.00787401574803149606299212598425f, // 1.0f / 127.0f
			-1.0f, 1.0f);
	}
	
	GLM_FUNC_QUALIFIER uint16 packUnorm1x16(float s)
	{
		return static_cast<uint16>(round(clamp(s, 0.0f, 1.0f) * 65535.0f));
	}

	GLM_FUNC_QUALIFIER float unpackUnorm1x16(uint16 p)
	{
		float const Unpack(p);
		return Unpack * 1.5259021896696421759365224689097e-5f; // 1.0 / 65535.0
	}

	GLM_FUNC_QUALIFIER uint64 packUnorm4x16(vec4 const & v)
	{
		u16vec4 const Topack(round(clamp(v , 0.0f, 1.0f) * 65535.0f));
		return reinterpret_cast<uint64 const &>(Topack);
	}

	GLM_FUNC_QUALIFIER vec4 unpackUnorm4x16(uint64 p)
	{
		vec4 const Unpack(reinterpret_cast<u16vec4 const &>(p));
		return Unpack * 1.5259021896696421759365224689097e-5f; // 1.0 / 65535.0
	}

	GLM_FUNC_QUALIFIER uint16 packSnorm1x16(float v)
	{
		int16 const Topack = static_cast<int16>(round(clamp(v ,-1.0f, 1.0f) * 32767.0f));
		return reinterpret_cast<uint16 const &>(Topack);
	}

	GLM_FUNC_QUALIFIER float unpackSnorm1x16(uint16 p)
	{
		float const Unpack(reinterpret_cast<int16 const &>(p));
		return clamp(
			Unpack * 3.0518509475997192297128208258309e-5f, //1.0f / 32767.0f, 
			-1.0f, 1.0f);
	}

	GLM_FUNC_QUALIFIER uint64 packSnorm4x16(vec4 const & v)
	{
		i16vec4 const Topack(round(clamp(v ,-1.0f, 1.0f) * 32767.0f));
		return reinterpret_cast<uint64 const &>(Topack);
	}

	GLM_FUNC_QUALIFIER vec4 unpackSnorm4x16(uint64 p)
	{
		vec4 const Unpack(reinterpret_cast<i16vec4 const &>(p));
		return clamp(
			Unpack * 3.0518509475997192297128208258309e-5f, //1.0f / 32767.0f,
			-1.0f, 1.0f);
	}

	GLM_FUNC_QUALIFIER uint16 packHalf1x16(float v)
	{
		int16 const Topack(detail::toFloat16(v));
		return reinterpret_cast<uint16 const &>(Topack);
	}

	GLM_FUNC_QUALIFIER float unpackHalf1x16(uint16 v)
	{
		return detail::toFloat32(reinterpret_cast<int16 const &>(v));
	}

	GLM_FUNC_QUALIFIER uint64 packHalf4x16(glm::vec4 const & v)
	{
		i16vec4 Unpack(
			detail::toFloat16(v.x),
			detail::toFloat16(v.y),
			detail::toFloat16(v.z),
			detail::toFloat16(v.w));

		return reinterpret_cast<uint64 const &>(Unpack);
	}

	GLM_FUNC_QUALIFIER glm::vec4 unpackHalf4x16(uint64 v)
	{
		i16vec4 Unpack(reinterpret_cast<i16vec4 const &>(v));
	
		return vec4(
			detail::toFloat32(Unpack.x),
			detail::toFloat32(Unpack.y),
			detail::toFloat32(Unpack.z),
			detail::toFloat32(Unpack.w));
	}

	GLM_FUNC_QUALIFIER uint32 packI3x10_1x2(ivec4 const & v)
	{
		detail::i10i10i10i2 Result;
		Result.data.x = v.x;
		Result.data.y = v.y;
		Result.data.z = v.z;
		Result.data.w = v.w;
		return Result.pack; 
	}

	GLM_FUNC_QUALIFIER ivec4 unpackI3x10_1x2(uint32 v)
	{
		detail::i10i10i10i2 Unpack;
		Unpack.pack = v;
		return ivec4(
			Unpack.data.x,
			Unpack.data.y,
			Unpack.data.z,
			Unpack.data.w);
	}

	GLM_FUNC_QUALIFIER uint32 packU3x10_1x2(uvec4 const & v)
	{
		detail::u10u10u10u2 Result;
		Result.data.x = v.x;
		Result.data.y = v.y;
		Result.data.z = v.z;
		Result.data.w = v.w;
		return Result.pack; 
	}

	GLM_FUNC_QUALIFIER uvec4 unpackU3x10_1x2(uint32 v)
	{
		detail::u10u10u10u2 Unpack;
		Unpack.pack = v;
		return uvec4(
			Unpack.data.x,
			Unpack.data.y,
			Unpack.data.z,
			Unpack.data.w);
	}

	GLM_FUNC_QUALIFIER uint32 packSnorm3x10_1x2(vec4 const & v)
	{
		detail::i10i10i10i2 Result;
		Result.data.x = int(round(clamp(v.x,-1.0f, 1.0f) * 511.f));
		Result.data.y = int(round(clamp(v.y,-1.0f, 1.0f) * 511.f));
		Result.data.z = int(round(clamp(v.z,-1.0f, 1.0f) * 511.f));
		Result.data.w = int(round(clamp(v.w,-1.0f, 1.0f) *   1.f));
		return Result.pack;
	}

	GLM_FUNC_QUALIFIER vec4 unpackSnorm3x10_1x2(uint32 v)
	{
		detail::i10i10i10i2 Unpack;
		Unpack.pack = v;
		vec4 Result;
		Result.x = clamp(float(Unpack.data.x) / 511.f, -1.0f, 1.0f);
		Result.y = clamp(float(Unpack.data.y) / 511.f, -1.0f, 1.0f);
		Result.z = clamp(float(Unpack.data.z) / 511.f, -1.0f, 1.0f);
		Result.w = clamp(float(Unpack.data.w) /   1.f, -1.0f, 1.0f);
		return Result;
	}

	GLM_FUNC_QUALIFIER uint32 packUnorm3x10_1x2(vec4 const & v)
	{
		detail::i10i10i10i2 Result;
		Result.data.x = int(round(clamp(v.x, 0.0f, 1.0f) * 1023.f));
		Result.data.y = int(round(clamp(v.y, 0.0f, 1.0f) * 1023.f));
		Result.data.z = int(round(clamp(v.z, 0.0f, 1.0f) * 1023.f));
		Result.data.w = int(round(clamp(v.w, 0.0f, 1.0f) *    3.f));
		return Result.pack;
	}

	GLM_FUNC_QUALIFIER vec4 unpackUnorm3x10_1x2(uint32 v)
	{
		detail::i10i10i10i2 Unpack;
		Unpack.pack = v;
		vec4 Result;
		Result.x = float(Unpack.data.x) / 1023.f;
		Result.y = float(Unpack.data.y) / 1023.f;
		Result.z = float(Unpack.data.z) / 1023.f;
		Result.w = float(Unpack.data.w) /   3.f;
		return Result;
	}

	GLM_FUNC_QUALIFIER uint32 packF2x11_1x10(vec3 const & v)
	{
		return
			((detail::floatTo11bit(v.x) & ((1 << 11) - 1)) <<  0) |
			((detail::floatTo11bit(v.y) & ((1 << 11) - 1)) << 11) |
			((detail::floatTo10bit(v.z) & ((1 << 10) - 1)) << 22);
	}

	GLM_FUNC_QUALIFIER vec3 unpackF2x11_1x10(uint32 v)
	{
		return vec3(
			detail::packed11bitToFloat(v >> 0),
			detail::packed11bitToFloat(v >> 11),
			detail::packed10bitToFloat(v >> 22));
	}

}//namespace glm
