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
/// @file glm/detail/func_geometric.inl
/// @date 2008-08-03 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "func_exponential.hpp"
#include "func_common.hpp"
#include "type_vec2.hpp"
#include "type_vec4.hpp"
#include "type_float.hpp"

namespace glm{
namespace detail
{
	template <template <class, precision> class vecType, typename T, precision P>
	struct compute_dot{};

	template <typename T, precision P>
	struct compute_dot<tvec1, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tvec1<T, P> const & a, tvec1<T, P> const & b)
		{
			return a.x * b.x;
		}
	};

	template <typename T, precision P>
	struct compute_dot<tvec2, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tvec2<T, P> const & x, tvec2<T, P> const & y)
		{
			tvec2<T, P> tmp(x * y);
			return tmp.x + tmp.y;
		}
	};

	template <typename T, precision P>
	struct compute_dot<tvec3, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tvec3<T, P> const & x, tvec3<T, P> const & y)
		{
			tvec3<T, P> tmp(x * y);
			return tmp.x + tmp.y + tmp.z;
		}
	};

	template <typename T, precision P>
	struct compute_dot<tvec4, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tvec4<T, P> const & x, tvec4<T, P> const & y)
		{
			tvec4<T, P> tmp(x * y);
			return (tmp.x + tmp.y) + (tmp.z + tmp.w);
		}
	};
}//namespace detail

	// length
	template <typename genType>
	GLM_FUNC_QUALIFIER genType length(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'length' only accept floating-point inputs");

		return abs(x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER T length(vecType<T, P> const & v)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'length' only accept floating-point inputs");

		return sqrt(dot(v, v));
	}

	// distance
	template <typename genType>
	GLM_FUNC_QUALIFIER genType distance(genType const & p0, genType const & p1)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'distance' only accept floating-point inputs");

		return length(p1 - p0);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER T distance(vecType<T, P> const & p0, vecType<T, P> const & p1)
	{
		return length(p1 - p0);
	}

	// dot
	template <typename T>
	GLM_FUNC_QUALIFIER T dot(T x, T y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'dot' only accept floating-point inputs");
		return x * y;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER T dot(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'dot' only accept floating-point inputs");
		return detail::compute_dot<vecType, T, P>::call(x, y);
	}

	// cross
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> cross(tvec3<T, P> const & x, tvec3<T, P> const & y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'cross' only accept floating-point inputs");

		return tvec3<T, P>(
			x.y * y.z - y.y * x.z,
			x.z * y.x - y.z * x.x,
			x.x * y.y - y.x * x.y);
	}

	// normalize
	template <typename genType>
	GLM_FUNC_QUALIFIER genType normalize(genType const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'normalize' only accept floating-point inputs");

		return x < genType(0) ? genType(-1) : genType(1);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> normalize(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'normalize' only accept floating-point inputs");

		return x * inversesqrt(dot(x, x));
	}

	// faceforward
	template <typename genType>
	GLM_FUNC_QUALIFIER genType faceforward(genType const & N, genType const & I, genType const & Nref)
	{
		return dot(Nref, I) < static_cast<genType>(0) ? N : -N;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> faceforward(vecType<T, P> const & N, vecType<T, P> const & I, vecType<T, P> const & Nref)
	{
		return dot(Nref, I) < static_cast<T>(0) ? N : -N;
	}

	// reflect
	template <typename genType>
	GLM_FUNC_QUALIFIER genType reflect(genType const & I, genType const & N)
	{
		return I - N * dot(N, I) * static_cast<genType>(2);
	}

	// refract
	template <typename genType>
	GLM_FUNC_QUALIFIER genType refract(genType const & I, genType const & N, genType const & eta)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'refract' only accept floating-point inputs");

		genType const dotValue(dot(N, I));
		genType const k(static_cast<genType>(1) - eta * eta * (static_cast<genType>(1) - dotValue * dotValue));
		return (eta * I - (eta * dotValue + sqrt(k)) * N) * static_cast<genType>(k >= static_cast<genType>(0));
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> refract(vecType<T, P> const & I, vecType<T, P> const & N, T eta)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'refract' only accept floating-point inputs");

		T const dotValue(dot(N, I));
		T const k(static_cast<T>(1) - eta * eta * (static_cast<T>(1) - dotValue * dotValue));
		return (eta * I - (eta * dotValue + std::sqrt(k)) * N) * static_cast<T>(k >= static_cast<T>(0));
	}
}//namespace glm
