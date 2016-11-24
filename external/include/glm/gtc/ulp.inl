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
/// @ref gtc_ulp
/// @file glm/gtc/ulp.inl
/// @date 2011-03-07 / 2012-04-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////
/// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
///
/// Developed at SunPro, a Sun Microsystems, Inc. business.
/// Permission to use, copy, modify, and distribute this
/// software is freely granted, provided that this notice
/// is preserved.
///////////////////////////////////////////////////////////////////////////////////

#include "../detail/type_int.hpp"
#include <cmath>
#include <cfloat>
#include <limits>

#if(GLM_COMPILER & GLM_COMPILER_VC)
#	pragma warning(push)
#	pragma warning(disable : 4127)
#endif

typedef union
{
	float value;
	/* FIXME: Assumes 32 bit int.  */
	unsigned int word;
} ieee_float_shape_type;

typedef union
{
	double value;
	struct
	{
		glm::detail::int32 lsw;
		glm::detail::int32 msw;
	} parts;
} ieee_double_shape_type;

#define GLM_EXTRACT_WORDS(ix0,ix1,d)		\
	do {									\
		ieee_double_shape_type ew_u;		\
		ew_u.value = (d);					\
		(ix0) = ew_u.parts.msw;				\
		(ix1) = ew_u.parts.lsw;				\
	} while (0)

#define GLM_GET_FLOAT_WORD(i,d)				\
	do {									\
		ieee_float_shape_type gf_u;			\
		gf_u.value = (d);					\
		(i) = gf_u.word;					\
	} while (0)

#define GLM_SET_FLOAT_WORD(d,i)				\
	do {									\
		ieee_float_shape_type sf_u;			\
		sf_u.word = (i);					\
		(d) = sf_u.value;					\
	} while (0)

#define GLM_INSERT_WORDS(d,ix0,ix1)			\
	do {									\
		ieee_double_shape_type iw_u;		\
		iw_u.parts.msw = (ix0);				\
		iw_u.parts.lsw = (ix1);				\
		(d) = iw_u.value;					\
	} while (0)

namespace glm{
namespace detail
{
	GLM_FUNC_QUALIFIER float nextafterf(float x, float y)
	{
		volatile float t;
		glm::detail::int32 hx, hy, ix, iy;

		GLM_GET_FLOAT_WORD(hx, x);
		GLM_GET_FLOAT_WORD(hy, y);
		ix = hx&0x7fffffff;		// |x|
		iy = hy&0x7fffffff;		// |y|

		if((ix>0x7f800000) ||	// x is nan 
			(iy>0x7f800000))	// y is nan 
			return x+y;
		if(x==y) return y;		// x=y, return y
		if(ix==0) {				// x == 0
			GLM_SET_FLOAT_WORD(x,(hy&0x80000000)|1);// return +-minsubnormal
			t = x*x;
			if(t==x) return t; else return x;	// raise underflow flag
		}
		if(hx>=0) {				// x > 0 
			if(hx>hy) {			// x > y, x -= ulp
				hx -= 1;
			} else {			// x < y, x += ulp
				hx += 1;
			}
		} else {				// x < 0
			if(hy>=0||hx>hy){	// x < y, x -= ulp
				hx -= 1;
			} else {			// x > y, x += ulp
				hx += 1;
			}
		}
		hy = hx&0x7f800000;
		if(hy>=0x7f800000) return x+x;  // overflow
		if(hy<0x00800000) {             // underflow
			t = x*x;
			if(t!=x) {          // raise underflow flag
				GLM_SET_FLOAT_WORD(y,hx);
				return y;
			}
		}
		GLM_SET_FLOAT_WORD(x,hx);
		return x;
	}

	GLM_FUNC_QUALIFIER double nextafter(double x, double y)
	{
		volatile double t;
		glm::detail::int32 hx, hy, ix, iy;
		glm::detail::uint32 lx, ly;

		GLM_EXTRACT_WORDS(hx, lx, x);
		GLM_EXTRACT_WORDS(hy, ly, y);
		ix = hx & 0x7fffffff;             // |x| 
		iy = hy & 0x7fffffff;             // |y| 

		if(((ix>=0x7ff00000)&&((ix-0x7ff00000)|lx)!=0) ||   // x is nan
			((iy>=0x7ff00000)&&((iy-0x7ff00000)|ly)!=0))     // y is nan
			return x+y;
		if(x==y) return y;              // x=y, return y
		if((ix|lx)==0) {                        // x == 0 
			GLM_INSERT_WORDS(x, hy & 0x80000000, 1);    // return +-minsubnormal
			t = x*x;
			if(t==x) return t; else return x;   // raise underflow flag 
		}
		if(hx>=0) {                             // x > 0 
			if(hx>hy||((hx==hy)&&(lx>ly))) {    // x > y, x -= ulp 
				if(lx==0) hx -= 1;
				lx -= 1;
			} else {                            // x < y, x += ulp
				lx += 1;
				if(lx==0) hx += 1;
			}
		} else {                                // x < 0 
			if(hy>=0||hx>hy||((hx==hy)&&(lx>ly))){// x < y, x -= ulp
				if(lx==0) hx -= 1;
				lx -= 1;
			} else {                            // x > y, x += ulp
				lx += 1;
				if(lx==0) hx += 1;
			}
		}
		hy = hx&0x7ff00000;
		if(hy>=0x7ff00000) return x+x;  // overflow
		if(hy<0x00100000) {             // underflow
			t = x*x;
			if(t!=x) {          // raise underflow flag
				GLM_INSERT_WORDS(y,hx,lx);
				return y;
			}
		}
		GLM_INSERT_WORDS(x,hx,lx);
		return x;
	}
}//namespace detail
}//namespace glm

#if(GLM_COMPILER & GLM_COMPILER_VC)
#	pragma warning(pop)
#endif

namespace glm
{
	template <>
	GLM_FUNC_QUALIFIER float next_float(float const & x)
	{
#		if GLM_HAS_CXX11_STL
			return std::nextafter(x, std::numeric_limits<float>::max());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
			return detail::nextafterf(x, FLT_MAX);
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
			return __builtin_nextafterf(x, FLT_MAX);
#		else
			return nextafterf(x, FLT_MAX);
#		endif
	}

	template <>
	GLM_FUNC_QUALIFIER double next_float(double const & x)
	{
#		if GLM_HAS_CXX11_STL
			return std::nextafter(x, std::numeric_limits<double>::max());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
			return detail::nextafter(x, std::numeric_limits<double>::max());
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
			return __builtin_nextafter(x, FLT_MAX);
#		else
			return nextafter(x, DBL_MAX);
#		endif
	}

	template<typename T, precision P, template<typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> next_float(vecType<T, P> const & x)
	{
		vecType<T, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(Result); ++i)
			Result[i] = next_float(x[i]);
		return Result;
	}

	GLM_FUNC_QUALIFIER float prev_float(float const & x)
	{
#		if GLM_HAS_CXX11_STL
			return std::nextafter(x, std::numeric_limits<float>::min());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
			return detail::nextafterf(x, FLT_MIN);
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
			return __builtin_nextafterf(x, FLT_MIN);
#		else
			return nextafterf(x, FLT_MIN);
#		endif
	}

	GLM_FUNC_QUALIFIER double prev_float(double const & x)
	{
#		if GLM_HAS_CXX11_STL
			return std::nextafter(x, std::numeric_limits<double>::min());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
			return _nextafter(x, DBL_MIN);
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
			return __builtin_nextafter(x, DBL_MIN);
#		else
			return nextafter(x, DBL_MIN);
#		endif
	}

	template<typename T, precision P, template<typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> prev_float(vecType<T, P> const & x)
	{
		vecType<T, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(Result); ++i)
			Result[i] = prev_float(x[i]);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T next_float(T const & x, uint const & ulps)
	{
		T temp = x;
		for(uint i = 0; i < ulps; ++i)
			temp = next_float(temp);
		return temp;
	}

	template<typename T, precision P, template<typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> next_float(vecType<T, P> const & x, vecType<uint, P> const & ulps)
	{
		vecType<T, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(Result); ++i)
			Result[i] = next_float(x[i], ulps[i]);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T prev_float(T const & x, uint const & ulps)
	{
		T temp = x;
		for(uint i = 0; i < ulps; ++i)
			temp = prev_float(temp);
		return temp;
	}

	template<typename T, precision P, template<typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<T, P> prev_float(vecType<T, P> const & x, vecType<uint, P> const & ulps)
	{
		vecType<T, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(Result); ++i)
			Result[i] = prev_float(x[i], ulps[i]);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER uint float_distance(T const & x, T const & y)
	{
		uint ulp = 0;

		if(x < y)
		{
			T temp = x;
			while(temp != y)// && ulp < std::numeric_limits<std::size_t>::max())
			{
				++ulp;
				temp = next_float(temp);
			}
		}
		else if(y < x)
		{
			T temp = y;
			while(temp != x)// && ulp < std::numeric_limits<std::size_t>::max())
			{
				++ulp;
				temp = next_float(temp);
			}
		}
		else // ==
		{

		}

		return ulp;
	}

	template<typename T, precision P, template<typename, precision> class vecType>
	GLM_FUNC_QUALIFIER vecType<uint, P> float_distance(vecType<T, P> const & x, vecType<T, P> const & y)
	{
		vecType<uint, P> Result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(Result); ++i)
			Result[i] = float_distance(x[i], y[i]);
		return Result;
	}
}//namespace glm
