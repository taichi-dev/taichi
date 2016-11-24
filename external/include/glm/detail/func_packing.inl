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
/// @file glm/detail/func_packing.inl
/// @date 2010-03-17 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "func_common.hpp"
#include "type_half.hpp"
#include "../fwd.hpp"

namespace glm
{
	GLM_FUNC_QUALIFIER uint packUnorm2x16(vec2 const & v)
	{
		u16vec2 const Topack(round(clamp(v, 0.0f, 1.0f) * 65535.0f));
		return reinterpret_cast<uint const &>(Topack);
	}

	GLM_FUNC_QUALIFIER vec2 unpackUnorm2x16(uint p)
	{
		vec2 Unpack(reinterpret_cast<u16vec2 const &>(p));
		return Unpack * float(1.5259021896696421759365224689097e-5); // 1.0 / 65535.0
	}

	GLM_FUNC_QUALIFIER uint packSnorm2x16(vec2 const & v)
	{
		i16vec2 const Topack(round(clamp(v ,-1.0f, 1.0f) * 32767.0f));
		return reinterpret_cast<uint const &>(Topack);
	}

	GLM_FUNC_QUALIFIER vec2 unpackSnorm2x16(uint p)
	{
		vec2 const Unpack(reinterpret_cast<i16vec2 const &>(p));
		return clamp(
			Unpack * 3.0518509475997192297128208258309e-5f, //1.0f / 32767.0f,
			-1.0f, 1.0f);
	}

	GLM_FUNC_QUALIFIER uint packUnorm4x8(vec4 const & v)
	{
		u8vec4 const Topack(round(clamp(v, 0.0f, 1.0f) * 255.0f));
		return reinterpret_cast<uint const &>(Topack);
	}

	GLM_FUNC_QUALIFIER vec4 unpackUnorm4x8(uint p)
	{
		vec4 const Unpack(reinterpret_cast<u8vec4 const&>(p));
		return Unpack * float(0.0039215686274509803921568627451); // 1 / 255
	}
	
	GLM_FUNC_QUALIFIER uint packSnorm4x8(vec4 const & v)
	{
		i8vec4 const Topack(round(clamp(v ,-1.0f, 1.0f) * 127.0f));
		return reinterpret_cast<uint const &>(Topack);
	}
	
	GLM_FUNC_QUALIFIER glm::vec4 unpackSnorm4x8(uint p)
	{
		vec4 const Unpack(reinterpret_cast<i8vec4 const &>(p));
		return clamp(
			Unpack * 0.0078740157480315f, // 1.0f / 127.0f
			-1.0f, 1.0f);
	}

	GLM_FUNC_QUALIFIER double packDouble2x32(uvec2 const & v)
	{
		return reinterpret_cast<double const &>(v);
	}

	GLM_FUNC_QUALIFIER uvec2 unpackDouble2x32(double v)
	{
		return reinterpret_cast<uvec2 const &>(v);
	}

	GLM_FUNC_QUALIFIER uint packHalf2x16(vec2 const & v)
	{
		i16vec2 const Unpack(
			detail::toFloat16(v.x),
			detail::toFloat16(v.y));

		return reinterpret_cast<uint const &>(Unpack);
	}

	GLM_FUNC_QUALIFIER vec2 unpackHalf2x16(uint v)
	{
		i16vec2 const Unpack(reinterpret_cast<i16vec2 const &>(v));
	
		return vec2(
			detail::toFloat32(Unpack.x), 
			detail::toFloat32(Unpack.y));
	}
}//namespace glm

