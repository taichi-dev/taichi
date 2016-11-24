///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2012 G-Truc Creation (www.g-truc.net)
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
/// @ref gtx_scalar_relational
/// @file glm/gtx/scalar_relational.inl
/// @date 2013-02-04 / 2013-02-04
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T>
	GLM_FUNC_QUALIFIER bool lessThan
	(
		T const & x, 
		T const & y
	)
	{
		return x < y;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER bool lessThanEqual
	(
		T const & x, 
		T const & y
	)
	{
		return x <= y;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER bool greaterThan
	(
		T const & x, 
		T const & y
	)
	{
		return x > y;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER bool greaterThanEqual
	(
		T const & x, 
		T const & y
	)
	{
		return x >= y;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER bool equal
	(
		T const & x, 
		T const & y
	)
	{
		return x == y;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER bool notEqual
	(
		T const & x, 
		T const & y
	)
	{
		return x != y;
	}

	GLM_FUNC_QUALIFIER bool any
	(
		bool const & x
	)
	{
		return x;
	}

	GLM_FUNC_QUALIFIER bool all
	(
		bool const & x
	)
	{
		return x;
	}

	GLM_FUNC_QUALIFIER bool not_
	(
		bool const & x
	)
	{
		return !x;
	}
}//namespace glm
