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
/// @ref gtx_matrix_transform_2d
/// @file glm/gtc/matrix_transform_2d.inl
/// @date 2014-02-20
/// @author Miguel Ángel Pérez Martínez
///////////////////////////////////////////////////////////////////////////////////

#include "../trigonometric.hpp"

namespace glm
{
	
	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> translate(
		tmat3x3<T, P> const & m,
		tvec2<T, P> const & v)
	{
		tmat3x3<T, P> Result(m);
		Result[2] = m[0] * v[0] + m[1] * v[1] + m[2];
		return Result;
	}


	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> rotate(
		tmat3x3<T, P> const & m,
		T angle)
	{
		T const a = angle;
		T const c = cos(a);
		T const s = sin(a);

		tmat3x3<T, P> Result(uninitialize);
		Result[0] = m[0] * c + m[1] * s;
		Result[1] = m[0] * -s + m[1] * c;
		Result[2] = m[2];
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> scale(
		tmat3x3<T, P> const & m,
		tvec2<T, P> const & v)
	{
		tmat3x3<T, P> Result(uninitialize);
		Result[0] = m[0] * v[0];
		Result[1] = m[1] * v[1];
		Result[2] = m[2];
		return Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> shearX(
		tmat3x3<T, P> const & m,
		T y)
	{
		tmat3x3<T, P> Result(1);
		Result[0][1] = y;
		return m * Result;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat3x3<T, P> shearY(
		tmat3x3<T, P> const & m,
		T x)
	{
		tmat3x3<T, P> Result(1);
		Result[1][0] = x;
		return m * Result;
	}

}//namespace glm
