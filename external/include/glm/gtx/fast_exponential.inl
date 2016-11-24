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
/// @ref gtx_fast_exponential
/// @file glm/gtx/fast_exponential.inl
/// @date 2006-01-09 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	// fastPow:
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastPow(genType x, genType y)
	{
		return exp(y * log(x));
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastPow(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		return exp(y * log(x));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T fastPow(T x, int y)
	{
		T f = static_cast<T>(1);
		for(int i = 0; i < y; ++i)
			f *= x;
		return f;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastPow(vecType<T, P> const & x, vecType<int, P> const & y)
	{
		vecType<T, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
			Result[i] = fastPow(x[i], y[i]);
		return Result;
	}

	// fastExp
	// Note: This function provides accurate results only for value between -1 and 1, else avoid it.
	template <typename T>
	GLM_FUNC_QUALIFIER T fastExp(T x)
	{
		// This has a better looking and same performance in release mode than the following code. However, in debug mode it's slower.
		// return 1.0f + x * (1.0f + x * 0.5f * (1.0f + x * 0.3333333333f * (1.0f + x * 0.25 * (1.0f + x * 0.2f))));
		T x2 = x * x;
		T x3 = x2 * x;
		T x4 = x3 * x;
		T x5 = x4 * x;
		return T(1) + x + (x2 * T(0.5)) + (x3 * T(0.1666666667)) + (x4 * T(0.041666667)) + (x5 * T(0.008333333333));
	}
	/*  // Try to handle all values of float... but often shower than std::exp, glm::floor and the loop kill the performance
	GLM_FUNC_QUALIFIER float fastExp(float x)
	{
		const float e = 2.718281828f;
		const float IntegerPart = floor(x);
		const float FloatPart = x - IntegerPart;
		float z = 1.f;

		for(int i = 0; i < int(IntegerPart); ++i)
			z *= e;

		const float x2 = FloatPart * FloatPart;
		const float x3 = x2 * FloatPart;
		const float x4 = x3 * FloatPart;
		const float x5 = x4 * FloatPart;
		return z * (1.0f + FloatPart + (x2 * 0.5f) + (x3 * 0.1666666667f) + (x4 * 0.041666667f) + (x5 * 0.008333333333f));
	}

	// Increase accuracy on number bigger that 1 and smaller than -1 but it's not enough for high and negative numbers
	GLM_FUNC_QUALIFIER float fastExp(float x)
	{
		// This has a better looking and same performance in release mode than the following code. However, in debug mode it's slower.
		// return 1.0f + x * (1.0f + x * 0.5f * (1.0f + x * 0.3333333333f * (1.0f + x * 0.25 * (1.0f + x * 0.2f))));
		float x2 = x * x;
		float x3 = x2 * x;
		float x4 = x3 * x;
		float x5 = x4 * x;
		float x6 = x5 * x;
		float x7 = x6 * x;
		float x8 = x7 * x;
		return 1.0f + x + (x2 * 0.5f) + (x3 * 0.1666666667f) + (x4 * 0.041666667f) + (x5 * 0.008333333333f)+ (x6 * 0.00138888888888f) + (x7 * 0.000198412698f) + (x8 * 0.0000248015873f);;
	}
	*/

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastExp(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(fastExp, x);
	}

	// fastLog
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastLog(genType x)
	{
		return std::log(x);
	}

	/* Slower than the VC7.1 function...
	GLM_FUNC_QUALIFIER float fastLog(float x)
	{
		float y1 = (x - 1.0f) / (x + 1.0f);
		float y2 = y1 * y1;
		return 2.0f * y1 * (1.0f + y2 * (0.3333333333f + y2 * (0.2f + y2 * 0.1428571429f)));
	}
	*/

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastLog(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(fastLog, x);
	}

	//fastExp2, ln2 = 0.69314718055994530941723212145818f
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastExp2(genType x)
	{
		return fastExp(0.69314718055994530941723212145818f * x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastExp2(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(fastExp2, x);
	}

	// fastLog2, ln2 = 0.69314718055994530941723212145818f
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fastLog2(genType x)
	{
		return fastLog(x) / 0.69314718055994530941723212145818f;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fastLog2(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(fastLog2, x);
	}
}//namespace glm
