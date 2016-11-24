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
/// @ref gtc_epsilon
/// @file glm/gtc/epsilon.inl
/// @date 2012-04-07 / 2012-04-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

// Dependency:
#include "quaternion.hpp"
#include "../vector_relational.hpp"
#include "../common.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"

namespace glm
{
	template <>
	GLM_FUNC_QUALIFIER bool epsilonEqual
	(
		float const & x,
		float const & y,
		float const & epsilon
	)
	{
		return abs(x - y) < epsilon;
	}

	template <>
	GLM_FUNC_QUALIFIER bool epsilonEqual
	(
		double const & x,
		double const & y,
		double const & epsilon
	)
	{
		return abs(x - y) < epsilon;
	}

	template <>
	GLM_FUNC_QUALIFIER bool epsilonNotEqual
	(
		float const & x,
		float const & y,
		float const & epsilon
	)
	{
		return abs(x - y) >= epsilon;
	}

	template <>
	GLM_FUNC_QUALIFIER bool epsilonNotEqual
	(
		double const & x,
		double const & y,
		double const & epsilon
	)
	{
		return abs(x - y) >= epsilon;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> epsilonEqual
	(
		vecType<T, P> const & x,
		vecType<T, P> const & y,
		T const & epsilon
	)
	{
		return lessThan(abs(x - y), vecType<T, P>(epsilon));
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> epsilonEqual
	(
		vecType<T, P> const & x,
		vecType<T, P> const & y,
		vecType<T, P> const & epsilon
	)
	{
		return lessThan(abs(x - y), vecType<T, P>(epsilon));
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> epsilonNotEqual
	(
		vecType<T, P> const & x,
		vecType<T, P> const & y,
		T const & epsilon
	)
	{
		return greaterThanEqual(abs(x - y), vecType<T, P>(epsilon));
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> epsilonNotEqual
	(
		vecType<T, P> const & x,
		vecType<T, P> const & y,
		vecType<T, P> const & epsilon
	)
	{
		return greaterThanEqual(abs(x - y), vecType<T, P>(epsilon));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<bool, P> epsilonEqual
	(
		tquat<T, P> const & x,
		tquat<T, P> const & y,
		T const & epsilon
	)
	{
		tvec4<T, P> v(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
		return lessThan(abs(v), tvec4<T, P>(epsilon));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<bool, P> epsilonNotEqual
	(
		tquat<T, P> const & x,
		tquat<T, P> const & y,
		T const & epsilon
	)
	{
		tvec4<T, P> v(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
		return greaterThanEqual(abs(v), tvec4<T, P>(epsilon));
	}
}//namespace glm
