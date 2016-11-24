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
/// @file glm/detail/func_exponential.inl
/// @date 2008-08-03 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "func_vector_relational.hpp"
#include "_vectorize.hpp"
#include <limits>
#include <cmath>
#include <cassert>

namespace glm{
namespace detail
{
#	if GLM_HAS_CXX11_STL
		using std::log2;
#	else
		template <typename genType>
		genType log2(genType Value)
		{
			return std::log(Value) * static_cast<genType>(1.4426950408889634073599246810019);
		}
#	endif

	template <typename T, precision P, template <class, precision> class vecType, bool isFloat = true>
	struct compute_log2
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & vec)
		{
			return detail::functor1<T, T, P, vecType>::call(log2, vec);
		}
	};

	template <template <class, precision> class vecType, typename T, precision P>
	struct compute_inversesqrt
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x)
		{
			return static_cast<T>(1) / sqrt(x);
		}
	};
		
	template <template <class, precision> class vecType>
	struct compute_inversesqrt<vecType, float, lowp>
	{
		GLM_FUNC_QUALIFIER static vecType<float, lowp> call(vecType<float, lowp> const & x)
		{
			vecType<float, lowp> tmp(x);
			vecType<float, lowp> xhalf(tmp * 0.5f);
			vecType<uint, lowp>* p = reinterpret_cast<vecType<uint, lowp>*>(const_cast<vecType<float, lowp>*>(&x));
			vecType<uint, lowp> i = vecType<uint, lowp>(0x5f375a86) - (*p >> vecType<uint, lowp>(1));
			vecType<float, lowp>* ptmp = reinterpret_cast<vecType<float, lowp>*>(&i);
			tmp = *ptmp;
			tmp = tmp * (1.5f - xhalf * tmp * tmp);
			return tmp;
		}
	};
}//namespace detail

	// pow
	using std::pow;
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> pow(vecType<T, P> const & base, vecType<T, P> const & exponent)
	{
		return detail::functor2<T, P, vecType>::call(pow, base, exponent);
	}

	// exp
	using std::exp;
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> exp(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(exp, x);
	}

	// log
	using std::log;
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> log(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(log, x);
	}

	//exp2, ln2 = 0.69314718055994530941723212145818f
	template <typename genType>
	GLM_FUNC_QUALIFIER genType exp2(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'exp2' only accept floating-point inputs");

		return std::exp(static_cast<genType>(0.69314718055994530941723212145818) * x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> exp2(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(exp2, x);
	}

	// log2, ln2 = 0.69314718055994530941723212145818f
	template <typename genType>
	GLM_FUNC_QUALIFIER genType log2(genType x)
	{
		return log2(tvec1<genType>(x)).x;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> log2(vecType<T, P> const & x)
	{
		return detail::compute_log2<T, P, vecType, std::numeric_limits<T>::is_iec559>::call(x);
	}

	// sqrt
	using std::sqrt;
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> sqrt(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'sqrt' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(sqrt, x);
	}

	// inversesqrt
	template <typename genType>
	GLM_FUNC_QUALIFIER genType inversesqrt(genType x)
	{
		return static_cast<genType>(1) / sqrt(x);
	}
	
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> inversesqrt(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'inversesqrt' only accept floating-point inputs");
		return detail::compute_inversesqrt<vecType, T, P>::call(x);
	}
}//namespace glm
