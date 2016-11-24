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
/// @ref gtx_orthonormalize
/// @file glm/gtx/orthonormalize.inl
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x3<T, P> orthonormalize(tmat3x3<T, P> const & m)
	{
		tmat3x3<T, P> r = m;

		r[0] = normalize(r[0]);

		T d0 = dot(r[0], r[1]);
		r[1] -= r[0] * d0;
		r[1] = normalize(r[1]);

		T d1 = dot(r[1], r[2]);
		d0 = dot(r[0], r[2]);
		r[2] -= r[0] * d0 + r[1] * d1;
		r[2] = normalize(r[2]);

		return r;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tvec3<T, P> orthonormalize(tvec3<T, P> const & x, tvec3<T, P> const & y)
	{
		return normalize(x - y * dot(y, x));
	}
}//namespace glm
