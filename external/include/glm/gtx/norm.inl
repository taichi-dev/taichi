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
/// @ref gtx_norm
/// @file glm/gtx/norm.inl
/// @date 2005-12-21 / 2008-07-24
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T>
	GLM_FUNC_QUALIFIER T length2
	(
		T const & x
	)
	{
		return x * x;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T length2
	(
		tvec2<T, P> const & x
	)
	{
		return dot(x, x);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T length2
	(
		tvec3<T, P> const & x
	)
	{
		return dot(x, x);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T length2
	(
		tvec4<T, P> const & x
	)
	{
		return dot(x, x);
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T distance2
	(
		T const & p0,
		T const & p1
	)
	{
		return length2(p1 - p0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T distance2
	(
		tvec2<T, P> const & p0,
		tvec2<T, P> const & p1
	)
	{
		return length2(p1 - p0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T distance2
	(
		tvec3<T, P> const & p0,
		tvec3<T, P> const & p1
	)
	{
		return length2(p1 - p0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T distance2
	(
		tvec4<T, P> const & p0,
		tvec4<T, P> const & p1
	)
	{
		return length2(p1 - p0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T l1Norm
	(
		tvec3<T, P> const & a,
		tvec3<T, P> const & b
	)
	{
		return abs(b.x - a.x) + abs(b.y - a.y) + abs(b.z - a.z);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T l1Norm
	(
		tvec3<T, P> const & v
	)
	{
		return abs(v.x) + abs(v.y) + abs(v.z);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T l2Norm
	(
		tvec3<T, P> const & a,
		tvec3<T, P> const & b
	)
	{
		return length(b - a);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T l2Norm
	(
		tvec3<T, P> const & v
	)
	{
		return length(v);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T lxNorm
	(
		tvec3<T, P> const & x,
		tvec3<T, P> const & y,
		unsigned int Depth
	)
	{
		return pow(pow(y.x - x.x, T(Depth)) + pow(y.y - x.y, T(Depth)) + pow(y.z - x.z, T(Depth)), T(1) / T(Depth));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T lxNorm
	(
		tvec3<T, P> const & v,
		unsigned int Depth
	)
	{
		return pow(pow(v.x, T(Depth)) + pow(v.y, T(Depth)) + pow(v.z, T(Depth)), T(1) / T(Depth));
	}

}//namespace glm
