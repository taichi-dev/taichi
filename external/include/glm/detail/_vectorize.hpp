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
/// @file glm/detail/_vectorize.hpp
/// @date 2011-10-14 / 2011-10-14
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "type_vec1.hpp"
#include "type_vec2.hpp"
#include "type_vec3.hpp"
#include "type_vec4.hpp"

namespace glm{
namespace detail
{
	template <typename R, typename T, precision P, template <typename, precision> class vecType>
	struct functor1{};

	template <typename R, typename T, precision P>
	struct functor1<R, T, P, tvec1>
	{
		GLM_FUNC_QUALIFIER static tvec1<R, P> call(R (*Func) (T x), tvec1<T, P> const & v)
		{
			return tvec1<R, P>(Func(v.x));
		}
	};

	template <typename R, typename T, precision P>
	struct functor1<R, T, P, tvec2>
	{
		GLM_FUNC_QUALIFIER static tvec2<R, P> call(R (*Func) (T x), tvec2<T, P> const & v)
		{
			return tvec2<R, P>(Func(v.x), Func(v.y));
		}
	};

	template <typename R, typename T, precision P>
	struct functor1<R, T, P, tvec3>
	{
		GLM_FUNC_QUALIFIER static tvec3<R, P> call(R (*Func) (T x), tvec3<T, P> const & v)
		{
			return tvec3<R, P>(Func(v.x), Func(v.y), Func(v.z));
		}
	};

	template <typename R, typename T, precision P>
	struct functor1<R, T, P, tvec4>
	{
		GLM_FUNC_QUALIFIER static tvec4<R, P> call(R (*Func) (T x), tvec4<T, P> const & v)
		{
			return tvec4<R, P>(Func(v.x), Func(v.y), Func(v.z), Func(v.w));
		}
	};

	template <typename T, precision P, template <typename, precision> class vecType>
	struct functor2{};

	template <typename T, precision P>
	struct functor2<T, P, tvec1>
	{
		GLM_FUNC_QUALIFIER static tvec1<T, P> call(T (*Func) (T x, T y), tvec1<T, P> const & a, tvec1<T, P> const & b)
		{
			return tvec1<T, P>(Func(a.x, b.x));
		}
	};

	template <typename T, precision P>
	struct functor2<T, P, tvec2>
	{
		GLM_FUNC_QUALIFIER static tvec2<T, P> call(T (*Func) (T x, T y), tvec2<T, P> const & a, tvec2<T, P> const & b)
		{
			return tvec2<T, P>(Func(a.x, b.x), Func(a.y, b.y));
		}
	};

	template <typename T, precision P>
	struct functor2<T, P, tvec3>
	{
		GLM_FUNC_QUALIFIER static tvec3<T, P> call(T (*Func) (T x, T y), tvec3<T, P> const & a, tvec3<T, P> const & b)
		{
			return tvec3<T, P>(Func(a.x, b.x), Func(a.y, b.y), Func(a.z, b.z));
		}
	};

	template <typename T, precision P>
	struct functor2<T, P, tvec4>
	{
		GLM_FUNC_QUALIFIER static tvec4<T, P> call(T (*Func) (T x, T y), tvec4<T, P> const & a, tvec4<T, P> const & b)
		{
			return tvec4<T, P>(Func(a.x, b.x), Func(a.y, b.y), Func(a.z, b.z), Func(a.w, b.w));
		}
	};

	template <typename T, precision P, template <typename, precision> class vecType>
	struct functor2_vec_sca{};

	template <typename T, precision P>
	struct functor2_vec_sca<T, P, tvec1>
	{
		GLM_FUNC_QUALIFIER static tvec1<T, P> call(T (*Func) (T x, T y), tvec1<T, P> const & a, T b)
		{
			return tvec1<T, P>(Func(a.x, b));
		}
	};

	template <typename T, precision P>
	struct functor2_vec_sca<T, P, tvec2>
	{
		GLM_FUNC_QUALIFIER static tvec2<T, P> call(T (*Func) (T x, T y), tvec2<T, P> const & a, T b)
		{
			return tvec2<T, P>(Func(a.x, b), Func(a.y, b));
		}
	};

	template <typename T, precision P>
	struct functor2_vec_sca<T, P, tvec3>
	{
		GLM_FUNC_QUALIFIER static tvec3<T, P> call(T (*Func) (T x, T y), tvec3<T, P> const & a, T b)
		{
			return tvec3<T, P>(Func(a.x, b), Func(a.y, b), Func(a.z, b));
		}
	};

	template <typename T, precision P>
	struct functor2_vec_sca<T, P, tvec4>
	{
		GLM_FUNC_QUALIFIER static tvec4<T, P> call(T (*Func) (T x, T y), tvec4<T, P> const & a, T b)
		{
			return tvec4<T, P>(Func(a.x, b), Func(a.y, b), Func(a.z, b), Func(a.w, b));
		}
	};
}//namespace detail
}//namespace glm
