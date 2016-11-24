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
/// @file glm/detail/func_trigonometric.inl
/// @date 2008-08-03 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "_vectorize.hpp"
#include <cmath>
#include <limits>

namespace glm
{
	// radians
	template <typename genType>
	GLM_FUNC_QUALIFIER genType radians(genType degrees)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'radians' only accept floating-point input");

		return degrees * static_cast<genType>(0.01745329251994329576923690768489);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> radians(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(radians, v);
	}
	
	// degrees
	template <typename genType>
	GLM_FUNC_QUALIFIER genType degrees(genType radians)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'degrees' only accept floating-point input");

		return radians * static_cast<genType>(57.295779513082320876798154814105);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> degrees(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(degrees, v);
	}

	// sin
	using ::std::sin;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> sin(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(sin, v);
	}

	// cos
	using std::cos;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> cos(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(cos, v);
	}

	// tan
	using std::tan;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> tan(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(tan, v);
	}

	// asin
	using std::asin;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> asin(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(asin, v);
	}

	// acos
	using std::acos;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> acos(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(acos, v);
	}

	// atan
	template <typename genType>
	GLM_FUNC_QUALIFIER genType atan(genType const & y, genType const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'atan' only accept floating-point input");

		return ::std::atan2(y, x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> atan(vecType<T, P> const & a, vecType<T, P> const & b)
	{
		return detail::functor2<T, P, vecType>::call(atan2, a, b);
	}

	using std::atan;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> atan(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(atan, v);
	}

	// sinh
	using std::sinh;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> sinh(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(sinh, v);
	}

	// cosh
	using std::cosh;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> cosh(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(cosh, v);
	}

	// tanh
	using std::tanh;

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> tanh(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(tanh, v);
	}

	// asinh
#	if GLM_HAS_CXX11_STL
		using std::asinh;
#	else
		template <typename genType> 
		GLM_FUNC_QUALIFIER genType asinh(genType const & x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'asinh' only accept floating-point input");

			return (x < static_cast<genType>(0) ? static_cast<genType>(-1) : (x > static_cast<genType>(0) ? static_cast<genType>(1) : static_cast<genType>(0))) * log(abs(x) + sqrt(static_cast<genType>(1) + x * x));
		}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> asinh(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(asinh, v);
	}

	// acosh
#	if GLM_HAS_CXX11_STL
		using std::acosh;
#	else
		template <typename genType> 
		GLM_FUNC_QUALIFIER genType acosh(genType const & x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'acosh' only accept floating-point input");

			if(x < static_cast<genType>(1))
				return static_cast<genType>(0);
			return log(x + sqrt(x * x - static_cast<genType>(1)));
		}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> acosh(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(acosh, v);
	}

	// atanh
#	if GLM_HAS_CXX11_STL
		using std::atanh;
#	else
		template <typename genType>
		GLM_FUNC_QUALIFIER genType atanh(genType const & x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'atanh' only accept floating-point input");
		
			if(abs(x) >= static_cast<genType>(1))
				return 0;
			return static_cast<genType>(0.5) * log((static_cast<genType>(1) + x) / (static_cast<genType>(1) - x));
		}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> atanh(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(atanh, v);
	}
}//namespace glm
