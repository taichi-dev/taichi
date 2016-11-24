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
/// @ref gtc_reciprocal
/// @file glm/gtc/reciprocal.inl
/// @date 2008-10-09 / 2012-04-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "../trigonometric.hpp"
#include <limits>

namespace glm
{
	// sec
	template <typename genType>
	GLM_FUNC_QUALIFIER genType sec(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'sec' only accept floating-point values");
		return genType(1) / glm::cos(angle);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> sec(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'sec' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(sec, x);
	}

	// csc
	template <typename genType>
	GLM_FUNC_QUALIFIER genType csc(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'csc' only accept floating-point values");
		return genType(1) / glm::sin(angle);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> csc(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'csc' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(csc, x);
	}

	// cot
	template <typename genType>
	GLM_FUNC_QUALIFIER genType cot(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'cot' only accept floating-point values");
	
		genType const pi_over_2 = genType(3.1415926535897932384626433832795 / 2.0);
		return glm::tan(pi_over_2 - angle);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> cot(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'cot' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(cot, x);
	}

	// asec
	template <typename genType>
	GLM_FUNC_QUALIFIER genType asec(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'asec' only accept floating-point values");
		return acos(genType(1) / x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> asec(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'asec' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(asec, x);
	}

	// acsc
	template <typename genType>
	GLM_FUNC_QUALIFIER genType acsc(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'acsc' only accept floating-point values");
		return asin(genType(1) / x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> acsc(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'acsc' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(acsc, x);
	}

	// acot
	template <typename genType>
	GLM_FUNC_QUALIFIER genType acot(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'acot' only accept floating-point values");

		genType const pi_over_2 = genType(3.1415926535897932384626433832795 / 2.0);
		return pi_over_2 - atan(x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> acot(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'acot' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(acot, x);
	}

	// sech
	template <typename genType>
	GLM_FUNC_QUALIFIER genType sech(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'sech' only accept floating-point values");
		return genType(1) / glm::cosh(angle);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> sech(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'sech' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(sech, x);
	}

	// csch
	template <typename genType>
	GLM_FUNC_QUALIFIER genType csch(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'csch' only accept floating-point values");
		return genType(1) / glm::sinh(angle);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> csch(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'csch' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(csch, x);
	}

	// coth
	template <typename genType>
	GLM_FUNC_QUALIFIER genType coth(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'coth' only accept floating-point values");
		return glm::cosh(angle) / glm::sinh(angle);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> coth(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'coth' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(coth, x);
	}

	// asech
	template <typename genType>
	GLM_FUNC_QUALIFIER genType asech(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'asech' only accept floating-point values");
		return acosh(genType(1) / x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> asech(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'asech' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(asech, x);
	}

	// acsch
	template <typename genType>
	GLM_FUNC_QUALIFIER genType acsch(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'acsch' only accept floating-point values");
		return acsch(genType(1) / x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> acsch(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'acsch' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(acsch, x);
	}

	// acoth
	template <typename genType>
	GLM_FUNC_QUALIFIER genType acoth(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'acoth' only accept floating-point values");
		return atanh(genType(1) / x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> acoth(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'acoth' only accept floating-point inputs");
		return detail::functor1<T, T, P, vecType>::call(acoth, x);
	}
}//namespace glm
