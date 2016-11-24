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
/// @ref core
/// @file glm/detail/func_common.inl
/// @date 2008-08-03 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "func_vector_relational.hpp"
#include "type_vec2.hpp"
#include "type_vec3.hpp"
#include "type_vec4.hpp"
#include "_vectorize.hpp"
#include <limits>

namespace glm{
namespace detail
{
	template <typename genFIType, bool /*signed*/>
	struct compute_abs
	{};

	template <typename genFIType>
	struct compute_abs<genFIType, true>
	{
		GLM_FUNC_QUALIFIER static genFIType call(genFIType x)
		{
			GLM_STATIC_ASSERT(
				std::numeric_limits<genFIType>::is_iec559 || std::numeric_limits<genFIType>::is_signed,
				"'abs' only accept floating-point and integer scalar or vector inputs");

			return x >= genFIType(0) ? x : -x;
			// TODO, perf comp with: *(((int *) &x) + 1) &= 0x7fffffff;
		}
	};

	template <typename genFIType>
	struct compute_abs<genFIType, false>
	{
		GLM_FUNC_QUALIFIER static genFIType call(genFIType x)
		{
			GLM_STATIC_ASSERT(
				!std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer,
				"'abs' only accept floating-point and integer scalar or vector inputs");
			return x;
		}
	};

	template <typename T, typename U, precision P, template <class, precision> class vecType>
	struct compute_mix_vector
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x, vecType<T, P> const & y, vecType<U, P> const & a)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<U>::is_iec559, "'mix' only accept floating-point inputs for the interpolator a");

			return vecType<T, P>(vecType<U, P>(x) + a * vecType<U, P>(y - x));
		}
	};

	template <typename T, precision P, template <class, precision> class vecType>
	struct compute_mix_vector<T, bool, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x, vecType<T, P> const & y, vecType<bool, P> const & a)
		{
			vecType<T, P> Result(uninitialize);
			for(detail::component_count_t i = 0; i < detail::component_count(x); ++i)
				Result[i] = a[i] ? y[i] : x[i];
			return Result;
		}
	};

	template <typename T, typename U, precision P, template <class, precision> class vecType>
	struct compute_mix_scalar
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x, vecType<T, P> const & y, U const & a)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<U>::is_iec559, "'mix' only accept floating-point inputs for the interpolator a");

			return vecType<T, P>(vecType<U, P>(x) + a * vecType<U, P>(y - x));
		}
	};

	template <typename T, precision P, template <class, precision> class vecType>
	struct compute_mix_scalar<T, bool, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x, vecType<T, P> const & y, bool const & a)
		{
			return a ? y : x;
		}
	};

	template <typename T, typename U>
	struct compute_mix
	{
		GLM_FUNC_QUALIFIER static T call(T const & x, T const & y, U const & a)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<U>::is_iec559, "'mix' only accept floating-point inputs for the interpolator a");

			return static_cast<T>(static_cast<U>(x) + a * static_cast<U>(y - x));
		}
	};

	template <typename T>
	struct compute_mix<T, bool>
	{
		GLM_FUNC_QUALIFIER static T call(T const & x, T const & y, bool const & a)
		{
			return a ? y : x;
		}
	};

	template <typename T, precision P, template <class, precision> class vecType, bool isFloat = true, bool isSigned = true>
	struct compute_sign
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x)
		{
			return vecType<T, P>(glm::lessThan(vecType<T, P>(0), x)) - vecType<T, P>(glm::lessThan(x, vecType<T, P>(0)));
		}
	};

	template <typename T, precision P, template <class, precision> class vecType>
	struct compute_sign<T, P, vecType, false, false>
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x)
		{
			return vecType<T, P>(glm::greaterThan(x , vecType<T, P>(0)));
		}
	};

	template <typename T, precision P, template <class, precision> class vecType>
	struct compute_sign<T, P, vecType, false, true>
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & x)
		{
			T const Shift(static_cast<T>(sizeof(T) * 8 - 1));
			vecType<T, P> const y(vecType<typename make_unsigned<T>::type, P>(-x) >> typename make_unsigned<T>::type(Shift));

			return (x >> Shift) | y;
		}
	};

	template <typename T, precision P, template <class, precision> class vecType, typename genType, bool isFloat = true>
	struct compute_mod
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & a, genType const & b)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'mod' only accept floating-point inputs. Include <glm/gtc/integer.hpp> for integer inputs.");
			return a - b * floor(a / b);
		}
	};
}//namespace detail

	// abs
	template <>
	GLM_FUNC_QUALIFIER int32 abs(int32 x)
	{
		int32 const y = x >> 31;
		return (x ^ y) - y;
	}

	template <typename genFIType>
	GLM_FUNC_QUALIFIER genFIType abs(genFIType x)
	{
		return detail::compute_abs<genFIType, std::numeric_limits<genFIType>::is_signed>::call(x);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> abs(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(abs, x);
	}

	// sign
	// fast and works for any type
	template <typename genFIType> 
	GLM_FUNC_QUALIFIER genFIType sign(genFIType x)
	{
		GLM_STATIC_ASSERT(
			std::numeric_limits<genFIType>::is_iec559 || (std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer),
			"'sign' only accept signed inputs");
		
		return detail::compute_sign<genFIType, defaultp, tvec1, std::numeric_limits<genFIType>::is_iec559>::call(tvec1<genFIType>(x)).x;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> sign(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(
			std::numeric_limits<T>::is_iec559 || (std::numeric_limits<T>::is_signed && std::numeric_limits<T>::is_integer),
			"'sign' only accept signed inputs");

		return detail::compute_sign<T, P, vecType, std::numeric_limits<T>::is_iec559>::call(x);
	}

	// floor
	using ::std::floor;
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> floor(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(floor, x);
	}

	// trunc
#	if GLM_HAS_CXX11_STL
		using ::std::trunc;
#	else
		template <typename genType>
		GLM_FUNC_QUALIFIER genType trunc(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'trunc' only accept floating-point inputs");

			return x < static_cast<genType>(0) ? -floor(-x) : floor(x);
		}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> trunc(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(trunc, x);
	}

	// round
#	if GLM_HAS_CXX11_STL
		using ::std::round;
#	else
		template <typename genType>
		GLM_FUNC_QUALIFIER genType round(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'round' only accept floating-point inputs");

			return x < static_cast<genType>(0) ? static_cast<genType>(int(x - static_cast<genType>(0.5))) : static_cast<genType>(int(x + static_cast<genType>(0.5)));
		}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> round(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(round, x);
	}

/*
	// roundEven
	template <typename genType>
	GLM_FUNC_QUALIFIER genType roundEven(genType const& x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'roundEven' only accept floating-point inputs");

		return genType(int(x + genType(int(x) % 2)));
	}
*/

	// roundEven
	template <typename genType>
	GLM_FUNC_QUALIFIER genType roundEven(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'roundEven' only accept floating-point inputs");
		
		int Integer = static_cast<int>(x);
		genType IntegerPart = static_cast<genType>(Integer);
		genType FractionalPart = fract(x);

		if(FractionalPart > static_cast<genType>(0.5) || FractionalPart < static_cast<genType>(0.5))
		{
			return round(x);
		}
		else if((Integer % 2) == 0)
		{
			return IntegerPart;
		}
		else if(x <= static_cast<genType>(0)) // Work around... 
		{
			return IntegerPart - static_cast<genType>(1);
		}
		else
		{
			return IntegerPart + static_cast<genType>(1);
		}
		//else // Bug on MinGW 4.5.2
		//{
		//	return mix(IntegerPart + genType(-1), IntegerPart + genType(1), x <= genType(0));
		//}
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> roundEven(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(roundEven, x);
	}

	// ceil
	using ::std::ceil;
	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> ceil(vecType<T, P> const & x)
	{
		return detail::functor1<T, T, P, vecType>::call(ceil, x);
	}

	// fract
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fract(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'fract' only accept floating-point inputs");

		return fract(tvec1<genType>(x)).x;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> fract(vecType<T, P> const & x)
	{
		return x - floor(x);
	}

	// mod
	template <typename genType>
	GLM_FUNC_QUALIFIER genType mod(genType x, genType y)
	{
		return mod(tvec1<genType>(x), y).x;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> mod(vecType<T, P> const & x, T y)
	{
		return detail::compute_mod<T, P, vecType, T, std::numeric_limits<T>::is_iec559>::call(x, y);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> mod(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		return detail::compute_mod<T, P, vecType, vecType<T, P>, std::numeric_limits<T>::is_iec559>::call(x, y);
	}

	// modf
	template <typename genType>
	GLM_FUNC_QUALIFIER genType modf(genType x, genType & i)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'modf' only accept floating-point inputs");

		return std::modf(x, &i);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec1<T, P> modf(tvec1<T, P> const & x, tvec1<T, P> & i)
	{
		return tvec1<T, P>(
			modf(x.x, i.x));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> modf(tvec2<T, P> const & x, tvec2<T, P> & i)
	{
		return tvec2<T, P>(
			modf(x.x, i.x),
			modf(x.y, i.y));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> modf(tvec3<T, P> const & x, tvec3<T, P> & i)
	{
		return tvec3<T, P>(
			modf(x.x, i.x),
			modf(x.y, i.y),
			modf(x.z, i.z));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> modf(tvec4<T, P> const & x, tvec4<T, P> & i)
	{
		return tvec4<T, P>(
			modf(x.x, i.x),
			modf(x.y, i.y),
			modf(x.z, i.z),
			modf(x.w, i.w));
	}

	//// Only valid if (INT_MIN <= x-y <= INT_MAX)
	//// min(x,y)
	//r = y + ((x - y) & ((x - y) >> (sizeof(int) *
	//CHAR_BIT - 1)));
	//// max(x,y)
	//r = x - ((x - y) & ((x - y) >> (sizeof(int) *
	//CHAR_BIT - 1)));

	// min
	template <typename genType>
	GLM_FUNC_QUALIFIER genType min(genType x, genType y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || std::numeric_limits<genType>::is_integer, "'min' only accept floating-point or integer inputs");

		return x < y ? x : y;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> min(vecType<T, P> const & a, T b)
	{
		return detail::functor2_vec_sca<T, P, vecType>::call(min, a, b);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> min(vecType<T, P> const & a, vecType<T, P> const & b)
	{
		return detail::functor2<T, P, vecType>::call(min, a, b);
	}

	// max
	template <typename genType>
	GLM_FUNC_QUALIFIER genType max(genType x, genType y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || std::numeric_limits<genType>::is_integer, "'max' only accept floating-point or integer inputs");

		return x > y ? x : y;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> max(vecType<T, P> const & a, T b)
	{
		return detail::functor2_vec_sca<T, P, vecType>::call(max, a, b);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> max(vecType<T, P> const & a, vecType<T, P> const & b)
	{
		return detail::functor2<T, P, vecType>::call(max, a, b);
	}

	// clamp
	template <typename genType>
	GLM_FUNC_QUALIFIER genType clamp(genType x, genType minVal, genType maxVal)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || std::numeric_limits<genType>::is_integer, "'clamp' only accept floating-point or integer inputs");
		
		return min(max(x, minVal), maxVal);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> clamp(vecType<T, P> const & x, T minVal, T maxVal)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || std::numeric_limits<T>::is_integer, "'clamp' only accept floating-point or integer inputs");

		return min(max(x, minVal), maxVal);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> clamp(vecType<T, P> const & x, vecType<T, P> const & minVal, vecType<T, P> const & maxVal)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || std::numeric_limits<T>::is_integer, "'clamp' only accept floating-point or integer inputs");

		return min(max(x, minVal), maxVal);
	}

	template <typename genTypeT, typename genTypeU>
	GLM_FUNC_QUALIFIER genTypeT mix(genTypeT x, genTypeT y, genTypeU a)
	{
		return detail::compute_mix<genTypeT, genTypeU>::call(x, y, a);
	}

	template <typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> mix(vecType<T, P> const & x, vecType<T, P> const & y, U a)
	{
		return detail::compute_mix_scalar<T, U, P, vecType>::call(x, y, a);
	}
	
	template <typename T, typename U, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> mix(vecType<T, P> const & x, vecType<T, P> const & y, vecType<U, P> const & a)
	{
		return detail::compute_mix_vector<T, U, P, vecType>::call(x, y, a);
	}

	// step
	template <typename genType>
	GLM_FUNC_QUALIFIER genType step(genType edge, genType x)
	{
		return mix(static_cast<genType>(1), static_cast<genType>(0), glm::lessThan(x, edge));
	}

	template <template <typename, precision> class vecType, typename T, precision P>
	GLM_FUNC_QUALIFIER vecType<T, P> step(T edge, vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'step' only accept floating-point inputs");

		return mix(vecType<T, P>(1), vecType<T, P>(0), glm::lessThan(x, vecType<T, P>(edge)));
	}

	template <template <typename, precision> class vecType, typename T, precision P>
	GLM_FUNC_QUALIFIER vecType<T, P> step(vecType<T, P> const & edge, vecType<T, P> const & x)
	{
		return mix(vecType<T, P>(1), vecType<T, P>(0), glm::lessThan(x, edge));
	}

	// smoothstep
	template <typename genType>
	GLM_FUNC_QUALIFIER genType smoothstep(genType edge0, genType edge1, genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'smoothstep' only accept floating-point inputs");

		genType const tmp(clamp((x - edge0) / (edge1 - edge0), genType(0), genType(1)));
		return tmp * tmp * (genType(3) - genType(2) * tmp);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> smoothstep(T edge0, T edge1, vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'smoothstep' only accept floating-point inputs");

		vecType<T, P> const tmp(clamp((x - edge0) / (edge1 - edge0), static_cast<T>(0), static_cast<T>(1)));
		return tmp * tmp * (static_cast<T>(3) - static_cast<T>(2) * tmp);
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> smoothstep(vecType<T, P> const & edge0, vecType<T, P> const & edge1, vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'smoothstep' only accept floating-point inputs");

		vecType<T, P> const tmp(clamp((x - edge0) / (edge1 - edge0), static_cast<T>(0), static_cast<T>(1)));
		return tmp * tmp * (static_cast<T>(3) - static_cast<T>(2) * tmp);
	}

#	if GLM_HAS_CXX11_STL
		using std::isnan;
#	else
		template <typename genType> 
		GLM_FUNC_QUALIFIER bool isnan(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'isnan' only accept floating-point inputs");

#			if GLM_HAS_CXX11_STL
				return std::isnan(x);
#			elif GLM_COMPILER & (GLM_COMPILER_VC | GLM_COMPILER_INTEL)
				return _isnan(x) != 0;
#			elif GLM_COMPILER & (GLM_COMPILER_GCC | (GLM_COMPILER_APPLE_CLANG | GLM_COMPILER_LLVM))
#				if GLM_PLATFORM & GLM_PLATFORM_ANDROID && __cplusplus < 201103L
					return _isnan(x) != 0;
#				else
					return std::isnan(x);
#				endif
#			elif GLM_COMPILER & GLM_COMPILER_CUDA
				return isnan(x) != 0;
#			else
				return std::isnan(x);
#			endif
		}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> isnan(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isnan' only accept floating-point inputs");

		return detail::functor1<bool, T, P, vecType>::call(isnan, x);
	}

#	if GLM_HAS_CXX11_STL
		using std::isinf;
#	else
		template <typename genType> 
		GLM_FUNC_QUALIFIER bool isinf(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'isinf' only accept floating-point inputs");

#			if GLM_HAS_CXX11_STL
				return std::isinf(x);
#			elif GLM_COMPILER & (GLM_COMPILER_INTEL | GLM_COMPILER_VC)
				return _fpclass(x) == _FPCLASS_NINF || _fpclass(x) == _FPCLASS_PINF;
#			elif GLM_COMPILER & (GLM_COMPILER_GCC | (GLM_COMPILER_APPLE_CLANG | GLM_COMPILER_LLVM))
#				if(GLM_PLATFORM & GLM_PLATFORM_ANDROID && __cplusplus < 201103L)
					return _isinf(x) != 0;
#				else
					return std::isinf(x);
#				endif
#			elif GLM_COMPILER & GLM_COMPILER_CUDA
				// http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MATH__DOUBLE_g13431dd2b40b51f9139cbb7f50c18fab.html#g13431dd2b40b51f9139cbb7f50c18fab
				return isinf(double(x)) != 0;
#			else
				return std::isinf(x);
#			endif
	}
#	endif

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<bool, P> isinf(vecType<T, P> const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isnan' only accept floating-point inputs");

		return detail::functor1<bool, T, P, vecType>::call(isinf, x);
	}

	GLM_FUNC_QUALIFIER int floatBitsToInt(float const & v)
	{
		return reinterpret_cast<int&>(const_cast<float&>(v));
	}

	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_QUALIFIER vecType<int, P> floatBitsToInt(vecType<float, P> const & v)
	{
		return reinterpret_cast<vecType<int, P>&>(const_cast<vecType<float, P>&>(v));
	}

	GLM_FUNC_QUALIFIER uint floatBitsToUint(float const & v)
	{
		return reinterpret_cast<uint&>(const_cast<float&>(v));
	}

	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_QUALIFIER vecType<uint, P> floatBitsToUint(vecType<float, P> const & v)
	{
		return reinterpret_cast<vecType<uint, P>&>(const_cast<vecType<float, P>&>(v));
	}

	GLM_FUNC_QUALIFIER float intBitsToFloat(int const & v)
	{
		return reinterpret_cast<float&>(const_cast<int&>(v));
	}

	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_QUALIFIER vecType<float, P> intBitsToFloat(vecType<int, P> const & v)
	{
		return reinterpret_cast<vecType<float, P>&>(const_cast<vecType<int, P>&>(v));
	}

	GLM_FUNC_QUALIFIER float uintBitsToFloat(uint const & v)
	{
		return reinterpret_cast<float&>(const_cast<uint&>(v));
	}

	template <template <typename, precision> class vecType, precision P>
	GLM_FUNC_QUALIFIER vecType<float, P> uintBitsToFloat(vecType<uint, P> const & v)
	{
		return reinterpret_cast<vecType<float, P>&>(const_cast<vecType<uint, P>&>(v));
	}
	
	template <typename genType>
	GLM_FUNC_QUALIFIER genType fma(genType const & a, genType const & b, genType const & c)
	{
		return a * b + c;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType frexp(genType x, int & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'frexp' only accept floating-point inputs");

		return std::frexp(x, exp);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec1<T, P> frexp(tvec1<T, P> const & x, tvec1<int, P> & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'frexp' only accept floating-point inputs");

		return tvec1<T, P>(std::frexp(x.x, exp.x));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> frexp(tvec2<T, P> const & x, tvec2<int, P> & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'frexp' only accept floating-point inputs");

		return tvec2<T, P>(
			frexp(x.x, exp.x),
			frexp(x.y, exp.y));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> frexp(tvec3<T, P> const & x, tvec3<int, P> & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'frexp' only accept floating-point inputs");

		return tvec3<T, P>(
			frexp(x.x, exp.x),
			frexp(x.y, exp.y),
			frexp(x.z, exp.z));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> frexp(tvec4<T, P> const & x, tvec4<int, P> & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'frexp' only accept floating-point inputs");

		return tvec4<T, P>(
			frexp(x.x, exp.x),
			frexp(x.y, exp.y),
			frexp(x.z, exp.z),
			frexp(x.w, exp.w));
	}

	template <typename genType, precision P>
	GLM_FUNC_QUALIFIER genType ldexp(genType const & x, int const & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'ldexp' only accept floating-point inputs");

		return std::ldexp(x, exp);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec1<T, P> ldexp(tvec1<T, P> const & x, tvec1<int, P> const & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'ldexp' only accept floating-point inputs");

		return tvec1<T, P>(
			ldexp(x.x, exp.x));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> ldexp(tvec2<T, P> const & x, tvec2<int, P> const & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'ldexp' only accept floating-point inputs");

		return tvec2<T, P>(
			ldexp(x.x, exp.x),
			ldexp(x.y, exp.y));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> ldexp(tvec3<T, P> const & x, tvec3<int, P> const & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'ldexp' only accept floating-point inputs");

		return tvec3<T, P>(
			ldexp(x.x, exp.x),
			ldexp(x.y, exp.y),
			ldexp(x.z, exp.z));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> ldexp(tvec4<T, P> const & x, tvec4<int, P> const & exp)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'ldexp' only accept floating-point inputs");

		return tvec4<T, P>(
			ldexp(x.x, exp.x),
			ldexp(x.y, exp.y),
			ldexp(x.z, exp.z),
			ldexp(x.w, exp.w));
	}
}//namespace glm
