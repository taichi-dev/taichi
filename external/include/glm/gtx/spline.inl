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
/// @ref gtx_spline
/// @file glm/gtx/spline.inl
/// @date 2007-01-25 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename genType>
	GLM_FUNC_QUALIFIER genType catmullRom
	(
		genType const & v1, 
		genType const & v2, 
		genType const & v3, 
		genType const & v4, 
		typename genType::value_type const & s
	)
	{
		typename genType::value_type s1 = s;
		typename genType::value_type s2 = pow2(s);
		typename genType::value_type s3 = pow3(s);

		typename genType::value_type f1 = -s3 + typename genType::value_type(2) * s2 - s;
		typename genType::value_type f2 = typename genType::value_type(3) * s3 - typename genType::value_type(5) * s2 + typename genType::value_type(2);
		typename genType::value_type f3 = typename genType::value_type(-3) * s3 + typename genType::value_type(4) * s2 + s;
		typename genType::value_type f4 = s3 - s2;

		return (f1 * v1 + f2 * v2 + f3 * v3 + f4 * v4) / typename genType::value_type(2);

	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType hermite
	(
		genType const & v1, 
		genType const & t1, 
		genType const & v2, 
		genType const & t2, 
		typename genType::value_type const & s
	)
	{
		typename genType::value_type s1 = s;
		typename genType::value_type s2 = pow2(s);
		typename genType::value_type s3 = pow3(s);

		typename genType::value_type f1 = typename genType::value_type(2) * s3 - typename genType::value_type(3) * s2 + typename genType::value_type(1);
		typename genType::value_type f2 = typename genType::value_type(-2) * s3 + typename genType::value_type(3) * s2;
		typename genType::value_type f3 = s3 - typename genType::value_type(2) * s2 + s;
		typename genType::value_type f4 = s3 - s2;

		return f1 * v1 + f2 * v2 + f3 * t1 + f4 * t2;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType cubic
	(
		genType const & v1, 
		genType const & v2, 
		genType const & v3, 
		genType const & v4, 
		typename genType::value_type const & s
	)
	{
		return ((v1 * s + v2) * s + v3) * s + v4;
	}
}//namespace glm
