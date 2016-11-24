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
/// @ref gtx_extend
/// @file glm/gtx/extend.inl
/// @date 2006-01-07 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename genType>
	GLM_FUNC_QUALIFIER genType extend
	(
		genType const & Origin, 
		genType const & Source, 
		genType const & Distance
	)
	{
		return Origin + (Source - Origin) * Distance;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> extend
	(
		tvec2<T, P> const & Origin,
		tvec2<T, P> const & Source,
		T const & Distance
	)
	{
		return Origin + (Source - Origin) * Distance;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> extend
	(
		tvec3<T, P> const & Origin,
		tvec3<T, P> const & Source,
		T const & Distance
	)
	{
		return Origin + (Source - Origin) * Distance;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> extend
	(
		tvec4<T, P> const & Origin,
		tvec4<T, P> const & Source,
		T const & Distance
	)
	{
		return Origin + (Source - Origin) * Distance;
	}
}//namespace glm
