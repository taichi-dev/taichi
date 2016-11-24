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
/// @ref gtc_random
/// @file glm/gtc/random.inl
/// @date 2011-09-19 / 2012-04-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "../geometric.hpp"
#include "../exponential.hpp"
#include <cstdlib>
#include <ctime>
#include <cassert>

namespace glm{
namespace detail
{
	template <typename T, precision P, template <class, precision> class vecType>
	struct compute_rand
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call();
	};

	template <precision P>
	struct compute_rand<uint8, P, tvec1>
	{
		GLM_FUNC_QUALIFIER static tvec1<uint8, P> call()
		{
			return tvec1<uint8, P>(
				std::rand()) % std::numeric_limits<uint8>::max();
		}
	};

	template <precision P>
	struct compute_rand<uint8, P, tvec2>
	{
		GLM_FUNC_QUALIFIER static tvec2<uint8, P> call()
		{
			return tvec2<uint8, P>(
				std::rand(),
				std::rand()) % std::numeric_limits<uint8>::max();
		}
	};

	template <precision P>
	struct compute_rand<uint8, P, tvec3>
	{
		GLM_FUNC_QUALIFIER static tvec3<uint8, P> call()
		{
			return tvec3<uint8, P>(
				std::rand(),
				std::rand(),
				std::rand()) % std::numeric_limits<uint8>::max();
		}
	};

	template <precision P>
	struct compute_rand<uint8, P, tvec4>
	{
		GLM_FUNC_QUALIFIER static tvec4<uint8, P> call()
		{
			return tvec4<uint8, P>(
				std::rand(),
				std::rand(),
				std::rand(),
				std::rand()) % std::numeric_limits<uint8>::max();
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_rand<uint16, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint16, P> call()
		{
			return
				(vecType<uint16, P>(compute_rand<uint8, P, vecType>::call()) << static_cast<uint16>(8)) |
				(vecType<uint16, P>(compute_rand<uint8, P, vecType>::call()) << static_cast<uint16>(0));
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_rand<uint32, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint32, P> call()
		{
			return
				(vecType<uint32, P>(compute_rand<uint16, P, vecType>::call()) << static_cast<uint32>(16)) |
				(vecType<uint32, P>(compute_rand<uint16, P, vecType>::call()) << static_cast<uint32>(0));
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_rand<uint64, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint64, P> call()
		{
			return
				(vecType<uint64, P>(compute_rand<uint32, P, vecType>::call()) << static_cast<uint64>(32)) |
				(vecType<uint64, P>(compute_rand<uint32, P, vecType>::call()) << static_cast<uint64>(0));
		}
	};

	template <typename T, precision P, template <class, precision> class vecType>
	struct compute_linearRand
	{
		GLM_FUNC_QUALIFIER static vecType<T, P> call(vecType<T, P> const & Min, vecType<T, P> const & Max);
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<int8, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<int8, P> call(vecType<int8, P> const & Min, vecType<int8, P> const & Max)
		{
			return (vecType<int8, P>(compute_rand<uint8, P, vecType>::call() % vecType<uint8, P>(Max + static_cast<int8>(1) - Min))) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<uint8, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint8, P> call(vecType<uint8, P> const & Min, vecType<uint8, P> const & Max)
		{
			return (compute_rand<uint8, P, vecType>::call() % (Max + static_cast<uint8>(1) - Min)) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<int16, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<int16, P> call(vecType<int16, P> const & Min, vecType<int16, P> const & Max)
		{
			return (vecType<int16, P>(compute_rand<uint16, P, vecType>::call() % vecType<uint16, P>(Max + static_cast<int16>(1) - Min))) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<uint16, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint16, P> call(vecType<uint16, P> const & Min, vecType<uint16, P> const & Max)
		{
			return (compute_rand<uint16, P, vecType>::call() % (Max + static_cast<uint16>(1) - Min)) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<int32, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<int32, P> call(vecType<int32, P> const & Min, vecType<int32, P> const & Max)
		{
			return (vecType<int32, P>(compute_rand<uint32, P, vecType>::call() % vecType<uint32, P>(Max + static_cast<int32>(1) - Min))) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<uint32, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint32, P> call(vecType<uint32, P> const & Min, vecType<uint32, P> const & Max)
		{
			return (compute_rand<uint32, P, vecType>::call() % (Max + static_cast<uint32>(1) - Min)) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<int64, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<int64, P> call(vecType<int64, P> const & Min, vecType<int64, P> const & Max)
		{
			return (vecType<int64, P>(compute_rand<uint64, P, vecType>::call() % vecType<uint64, P>(Max + static_cast<int64>(1) - Min))) + Min;
		}
	};

	template <precision P, template <class, precision> class vecType>
	struct compute_linearRand<uint64, P, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<uint64, P> call(vecType<uint64, P> const & Min, vecType<uint64, P> const & Max)
		{
			return (compute_rand<uint64, P, vecType>::call() % (Max + static_cast<uint64>(1) - Min)) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<float, lowp, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<float, lowp> call(vecType<float, lowp> const & Min, vecType<float, lowp> const & Max)
		{
			return vecType<float, lowp>(compute_rand<uint8, lowp, vecType>::call()) / static_cast<float>(std::numeric_limits<uint8>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<float, mediump, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<float, mediump> call(vecType<float, mediump> const & Min, vecType<float, mediump> const & Max)
		{
			return vecType<float, mediump>(compute_rand<uint16, mediump, vecType>::call()) / static_cast<float>(std::numeric_limits<uint16>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<float, highp, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<float, highp> call(vecType<float, highp> const & Min, vecType<float, highp> const & Max)
		{
			return vecType<float, highp>(compute_rand<uint32, highp, vecType>::call()) / static_cast<float>(std::numeric_limits<uint32>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<double, lowp, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<double, lowp> call(vecType<double, lowp> const & Min, vecType<double, lowp> const & Max)
		{
			return vecType<double, lowp>(compute_rand<uint16, lowp, vecType>::call()) / static_cast<double>(std::numeric_limits<uint16>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<double, mediump, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<double, mediump> call(vecType<double, mediump> const & Min, vecType<double, mediump> const & Max)
		{
			return vecType<double, mediump>(compute_rand<uint32, mediump, vecType>::call()) / static_cast<double>(std::numeric_limits<uint32>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<double, highp, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<double, highp> call(vecType<double, highp> const & Min, vecType<double, highp> const & Max)
		{
			return vecType<double, highp>(compute_rand<uint64, highp, vecType>::call()) / static_cast<double>(std::numeric_limits<uint64>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<long double, lowp, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<long double, lowp> call(vecType<long double, lowp> const & Min, vecType<long double, lowp> const & Max)
		{
			return vecType<long double, lowp>(compute_rand<uint32, lowp, vecType>::call()) / static_cast<long double>(std::numeric_limits<uint32>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<long double, mediump, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<long double, mediump> call(vecType<long double, mediump> const & Min, vecType<long double, mediump> const & Max)
		{
			return vecType<long double, mediump>(compute_rand<uint64, mediump, vecType>::call()) / static_cast<long double>(std::numeric_limits<uint64>::max()) * (Max - Min) + Min;
		}
	};

	template <template <class, precision> class vecType>
	struct compute_linearRand<long double, highp, vecType>
	{
		GLM_FUNC_QUALIFIER static vecType<long double, highp> call(vecType<long double, highp> const & Min, vecType<long double, highp> const & Max)
		{
			return vecType<long double, highp>(compute_rand<uint64, highp, vecType>::call()) / static_cast<long double>(std::numeric_limits<uint64>::max()) * (Max - Min) + Min;
		}
	};
}//namespace detail

	template <typename genType>
	GLM_FUNC_QUALIFIER genType linearRand(genType Min, genType Max)
	{
		return detail::compute_linearRand<genType, highp, tvec1>::call(
			tvec1<genType, highp>(Min),
			tvec1<genType, highp>(Max)).x;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> linearRand(vecType<T, P> const & Min, vecType<T, P> const & Max)
	{
		return detail::compute_linearRand<T, P, vecType>::call(Min, Max);
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType gaussRand(genType Mean, genType Deviation)
	{
		genType w, x1, x2;
	
		do
		{
			x1 = linearRand(genType(-1), genType(1));
			x2 = linearRand(genType(-1), genType(1));
		
			w = x1 * x1 + x2 * x2;
		} while(w > genType(1));
	
		return x2 * Deviation * Deviation * sqrt((genType(-2) * log(w)) / w) + Mean;
	}

	template <typename T, precision P, template <typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> gaussRand(vecType<T, P> const & Mean, vecType<T, P> const & Deviation)
	{
		return detail::functor2<T, P, vecType>::call(gaussRand, Mean, Deviation);
	}

	template <typename T>
	GLM_FUNC_QUALIFIER tvec2<T, defaultp> diskRand(T Radius)
	{		
		tvec2<T, defaultp> Result(T(0));
		T LenRadius(T(0));
		
		do
		{
			Result = linearRand(
				tvec2<T, defaultp>(-Radius),
				tvec2<T, defaultp>(Radius));
			LenRadius = length(Result);
		}
		while(LenRadius > Radius);
		
		return Result;
	}
	
	template <typename T>
	GLM_FUNC_QUALIFIER tvec3<T, defaultp> ballRand(T Radius)
	{		
		tvec3<T, defaultp> Result(T(0));
		T LenRadius(T(0));
		
		do
		{
			Result = linearRand(
				tvec3<T, defaultp>(-Radius),
				tvec3<T, defaultp>(Radius));
			LenRadius = length(Result);
		}
		while(LenRadius > Radius);
		
		return Result;
	}
	
	template <typename T>
	GLM_FUNC_QUALIFIER tvec2<T, defaultp> circularRand(T Radius)
	{
		T a = linearRand(T(0), T(6.283185307179586476925286766559f));
		return tvec2<T, defaultp>(cos(a), sin(a)) * Radius;		
	}
	
	template <typename T>
	GLM_FUNC_QUALIFIER tvec3<T, defaultp> sphericalRand(T Radius)
	{
		T z = linearRand(T(-1), T(1));
		T a = linearRand(T(0), T(6.283185307179586476925286766559f));
	
		T r = sqrt(T(1) - z * z);
	
		T x = r * cos(a);
		T y = r * sin(a);
	
		return tvec3<T, defaultp>(x, y, z) * Radius;	
	}
}//namespace glm
