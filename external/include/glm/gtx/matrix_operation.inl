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
/// @ref gtx_matrix_operation
/// @file glm/gtx/matrix_operation.inl
/// @date 2009-08-29 / 2009-08-29
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> diagonal2x2
	(
		tvec2<T, P> const & v
	)
	{
		tmat2x2<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x3<T, P> diagonal2x3
	(
		tvec2<T, P> const & v
	)
	{
		tmat2x3<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> diagonal2x4
	(
		tvec2<T, P> const & v
	)
	{
		tmat2x4<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x2<T, P> diagonal3x2
	(
		tvec2<T, P> const & v
	)
	{
		tmat3x2<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x3<T, P> diagonal3x3
	(
		tvec3<T, P> const & v
	)
	{
		tmat3x3<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		Result[2][2] = v[2];
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x4<T, P> diagonal3x4
	(
		tvec3<T, P> const & v
	)
	{
		tmat3x4<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		Result[2][2] = v[2];
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x4<T, P> diagonal4x4
	(
		tvec4<T, P> const & v
	)
	{
		tmat4x4<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		Result[2][2] = v[2];
		Result[3][3] = v[3];
		return Result;		
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> diagonal4x3
	(
		tvec3<T, P> const & v
	)
	{
		tmat4x3<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		Result[2][2] = v[2];
		return Result;		
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x2<T, P> diagonal4x2
	(
		tvec2<T, P> const & v
	)
	{
		tmat4x2<T, P> Result(static_cast<T>(1));
		Result[0][0] = v[0];
		Result[1][1] = v[1];
		return Result;		
	}
}//namespace glm
