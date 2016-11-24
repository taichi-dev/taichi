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
/// @ref gtx_bit
/// @file glm/gtx/bit.inl
/// @date 2014-11-25 / 2014-11-25
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	///////////////////
	// highestBitValue

	template <typename genIUType>
	GLM_FUNC_QUALIFIER genIUType highestBitValue(genIUType Value)
	{
		genIUType tmp = Value;
		genIUType result = genIUType(0);
		while(tmp)
		{
			result = (tmp & (~tmp + 1)); // grab lowest bit
			tmp &= ~result; // clear lowest bit
		}
		return result;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> highestBitValue(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(highestBitValue, v);
	}

	///////////////////
	// powerOfTwoAbove

	template <typename genType>
	GLM_FUNC_QUALIFIER genType powerOfTwoAbove(genType value)
	{
		return isPowerOfTwo(value) ? value : highestBitValue(value) << 1;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> powerOfTwoAbove(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(powerOfTwoAbove, v);
	}

	///////////////////
	// powerOfTwoBelow

	template <typename genType>
	GLM_FUNC_QUALIFIER genType powerOfTwoBelow(genType value)
	{
		return isPowerOfTwo(value) ? value : highestBitValue(value);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> powerOfTwoBelow(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(powerOfTwoBelow, v);
	}

	/////////////////////
	// powerOfTwoNearest

	template <typename genType>
	GLM_FUNC_QUALIFIER genType powerOfTwoNearest(genType value)
	{
		if(isPowerOfTwo(value))
			return value;

		genType const prev = highestBitValue(value);
		genType const next = prev << 1;
		return (next - value) < (value - prev) ? next : prev;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> powerOfTwoNearest(vecType<T, P> const & v)
	{
		return detail::functor1<T, T, P, vecType>::call(powerOfTwoNearest, v);
	}

}//namespace glm
