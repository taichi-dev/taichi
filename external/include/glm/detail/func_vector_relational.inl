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
/// @file glm/detail/func_vector_relational.inl
/// @date 2008-08-03 / 2011-09-09
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <limits>

namespace glm
{
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> lessThan(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		assert(detail::component_count(x) == detail::component_count(y));

		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = x[i] < y[i];

		return Result;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> lessThanEqual(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		assert(detail::component_count(x) == detail::component_count(y));

		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = x[i] <= y[i];
		return Result;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> greaterThan(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		assert(detail::component_count(x) == detail::component_count(y));

		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = x[i] > y[i];
		return Result;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> greaterThanEqual(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		assert(detail::component_count(x) == detail::component_count(y));

		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = x[i] >= y[i];
		return Result;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> equal(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		assert(detail::component_count(x) == detail::component_count(y));

		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = x[i] == y[i];
		return Result;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> notEqual(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		assert(detail::component_count(x) == detail::component_count(y));

		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = x[i] != y[i];
		return Result;
	}

	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER bool any(vecType<bool, P> const & v)
	{
		bool Result = false;
		for(detail::component_count_t i = 0; i < detail::component_count(v); ++i)
			Result = Result || v[i];
		return Result;
	}

	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER bool all(vecType<bool, P> const & v)
	{
		bool Result = true;
		for(detail::component_count_t i = 0; i < detail::component_count(v); ++i)
			Result = Result && v[i];
		return Result;
	}

	template <precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> not_(vecType<bool, P> const & v)
	{
		vecType<bool, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(v); ++i)
			Result[i] = !v[i];
		return Result;
	}
}//namespace glm

