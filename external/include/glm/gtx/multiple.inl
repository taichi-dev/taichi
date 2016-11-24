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
/// @ref gtx_multiple
/// @file glm/gtx/multiple.inl
/// @date 2009-10-26 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	//////////////////////
	// higherMultiple

	template <typename genType>
	GLM_FUNC_QUALIFIER genType higherMultiple(genType Source, genType Multiple)
	{
		return detail::compute_ceilMultiple<std::numeric_limits<genType>::is_iec559, std::numeric_limits<genType>::is_signed>::call(Source, Multiple);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> higherMultiple(vecType<T, P> const & Source, vecType<T, P> const & Multiple)
	{
		return detail::functor2<T, P, vecType>::call(higherMultiple, Source, Multiple);
	}

	//////////////////////
	// lowerMultiple

	template <typename genType>
	GLM_FUNC_QUALIFIER genType lowerMultiple(genType Source, genType Multiple)
	{
		return detail::compute_floorMultiple<std::numeric_limits<genType>::is_iec559, std::numeric_limits<genType>::is_signed>::call(Source, Multiple);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> lowerMultiple(vecType<T, P> const & Source, vecType<T, P> const & Multiple)
	{
		return detail::functor2<T, P, vecType>::call(lowerMultiple, Source, Multiple);
	}
}//namespace glm
