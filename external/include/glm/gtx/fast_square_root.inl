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
/// @ref gtx_fast_square_root
/// @file glm/gtx/fast_square_root.inl
/// @date 2006-01-04 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	// fastSqrt
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastSqrt(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'fastSqrt' only accept floating-point input");

		return genType(1) / fastInverseSqrt(x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastSqrt(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(fastSqrt, x);
	}

	// fastInversesqrt
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastInverseSqrt(genType x)
	{
#		ifdef __CUDACC__ // Wordaround for a CUDA compiler bug up to CUDA6
			tvec1<T, P> tmp(detail::compute_inversesqrt<tvec1, genType, lowp>::call(tvec1<genType, lowp>(x)));
			return tmp.x;
#		else
			return detail::compute_inversesqrt<tvec1, genType, lowp>::call(tvec1<genType, lowp>(x)).x;
#		endif
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastInverseSqrt(vecType<T, P> const & x)
	{
		return detail::compute_inversesqrt<vecType, T, P>::call(x);
	}

	// fastLength
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastLength(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'fastLength' only accept floating-point inputs");

		return abs(x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER T fastLength(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'fastLength' only accept floating-point inputs");

		return fastSqrt(dot(x, x));
	}

	// fastDistance
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastDistance(genType x, genType y)
	{
		return fastLength(y - x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER T fastDistance(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		return fastLength(y - x);
	}

	// fastNormalize
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastNormalize(genType x)
	{
		return x > genType(0) ? genType(1) : -genType(1);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastNormalize(vecType<T, P> const & x)
	{
		return x * fastInverseSqrt(dot(x, x));
	}
}//namespace glm
