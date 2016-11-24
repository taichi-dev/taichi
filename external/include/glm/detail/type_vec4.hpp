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
/// @file glm/detail/type_vec4.hpp
/// @date 2008-08-22 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#pragma once

//#include "../fwd.hpp"
#include "setup.hpp"
#include "type_vec.hpp"
#ifdef GLM_SWIZZLE
#	if GLM_HAS_ANONYMOUS_UNION
#		include "_swizzle.hpp"
#	else
#		include "_swizzle_func.hpp"
#	endif
#endif //GLM_SWIZZLE
#include <cstddef>

namespace glm{
namespace detail
{
	template <typename T>
	struct simd
	{
		typedef T type[4];
	};

#	define GLM_NOT_BUGGY_VC32BITS !(GLM_MODEL == GLM_MODEL_32 && GLM_COMPILER & GLM_COMPILER_VC && GLM_COMPILER < GLM_COMPILER_VC2013)

#	if GLM_ARCH & GLM_ARCH_SSE2 && GLM_NOT_BUGGY_VC32BITS
		template <>
		struct simd<float>
		{
			typedef __m128 type;
		};

		template <>
		struct simd<int>
		{
			typedef __m128i type;
		};

		template <>
		struct simd<unsigned int>
		{
			typedef __m128i type;
		};
#	endif

#	if GLM_ARCH & GLM_ARCH_AVX && GLM_NOT_BUGGY_VC32BITS
		template <>
		struct simd<double>
		{
			typedef __m256d type;
		};
#	endif

#	if GLM_ARCH & GLM_ARCH_AVX2 && GLM_NOT_BUGGY_VC32BITS
		template <>
		struct simd<int64>
		{
			typedef __m256i type;
		};

		template <>
		struct simd<uint64>
		{
			typedef __m256i type;
		};
#	endif

}//namespace detail

	template <typename T, precision P = defaultp>
	struct tvec4
	{
		//////////////////////////////////////
		// Implementation detail

		typedef tvec4<T, P> type;
		typedef tvec4<bool, P> bool_type;
		typedef T value_type;

		//////////////////////////////////////
		// Data

#		if GLM_HAS_ANONYMOUS_UNION
			union
			{
				struct { T x, y, z, w;};
				struct { T r, g, b, a; };
				struct { T s, t, p, q; };

				typename detail::simd<T>::type data;

#				ifdef GLM_SWIZZLE
					_GLM_SWIZZLE4_2_MEMBERS(T, P, tvec2, x, y, z, w)
					_GLM_SWIZZLE4_2_MEMBERS(T, P, tvec2, r, g, b, a)
					_GLM_SWIZZLE4_2_MEMBERS(T, P, tvec2, s, t, p, q)
					_GLM_SWIZZLE4_3_MEMBERS(T, P, tvec3, x, y, z, w)
					_GLM_SWIZZLE4_3_MEMBERS(T, P, tvec3, r, g, b, a)
					_GLM_SWIZZLE4_3_MEMBERS(T, P, tvec3, s, t, p, q)
					_GLM_SWIZZLE4_4_MEMBERS(T, P, tvec4, x, y, z, w)
					_GLM_SWIZZLE4_4_MEMBERS(T, P, tvec4, r, g, b, a)
					_GLM_SWIZZLE4_4_MEMBERS(T, P, tvec4, s, t, p, q)
#				endif//GLM_SWIZZLE
			};
#		else
			union { T x, r, s; };
			union { T y, g, t; };
			union { T z, b, p; };
			union { T w, a, q; };

#			ifdef GLM_SWIZZLE
				GLM_SWIZZLE_GEN_VEC_FROM_VEC4(T, P, tvec4, tvec2, tvec3, tvec4)
#			endif//GLM_SWIZZLE
#		endif//GLM_LANG

		//////////////////////////////////////
		// Component accesses

#		ifdef GLM_FORCE_SIZE_FUNC
			/// Return the count of components of the vector
			typedef size_t size_type;
			GLM_FUNC_DECL GLM_CONSTEXPR size_type size() const;

			GLM_FUNC_DECL T & operator[](size_type i);
			GLM_FUNC_DECL T const & operator[](size_type i) const;
#		else
			/// Return the count of components of the vector
			typedef length_t length_type;
			GLM_FUNC_DECL GLM_CONSTEXPR length_type length() const;

			GLM_FUNC_DECL T & operator[](length_type i);
			GLM_FUNC_DECL T const & operator[](length_type i) const;
#		endif//GLM_FORCE_SIZE_FUNC

		//////////////////////////////////////
		// Implicit basic constructors

		GLM_FUNC_DECL tvec4();
		GLM_FUNC_DECL tvec4(tvec4<T, P> const & v);
		template <precision Q>
		GLM_FUNC_DECL tvec4(tvec4<T, Q> const & v);

		//////////////////////////////////////
		// Explicit basic constructors

		GLM_FUNC_DECL explicit tvec4(ctor);
		GLM_FUNC_DECL explicit tvec4(T s);
		GLM_FUNC_DECL tvec4(T a, T b, T c, T d);
		GLM_FUNC_DECL ~tvec4(){}

		//////////////////////////////////////
		// Conversion scalar constructors

		/// Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, typename D>
		GLM_FUNC_DECL tvec4(A a, B b, C c, D d);
		template <typename A, typename B, typename C, typename D>
		GLM_FUNC_DECL tvec4(tvec1<A, P> const & a, tvec1<B, P> const & b, tvec1<C, P> const & c, tvec1<D, P> const & d);

		//////////////////////////////////////
		// Conversion vector constructors

		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec2<A, Q> const & a, B b, C c);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec2<A, Q> const & a, tvec1<B, Q> const & b, tvec1<C, Q> const & c);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, precision Q>
		GLM_FUNC_DECL explicit tvec4(A a, tvec2<B, Q> const & b, C c);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec1<A, Q> const & a, tvec2<B, Q> const & b, tvec1<C, Q> const & c);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, precision Q>
		GLM_FUNC_DECL explicit tvec4(A a, B b, tvec2<C, Q> const & c);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, typename C, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec1<A, Q> const & a, tvec1<B, Q> const & b, tvec2<C, Q> const & c);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec3<A, Q> const & a, B b);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec3<A, Q> const & a, tvec1<B, Q> const & b);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, precision Q>
		GLM_FUNC_DECL explicit tvec4(A a, tvec3<B, Q> const & b);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec1<A, Q> const & a, tvec3<B, Q> const & b);
		//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
		template <typename A, typename B, precision Q>
		GLM_FUNC_DECL explicit tvec4(tvec2<A, Q> const & a, tvec2<B, Q> const & b);
		
#		ifdef GLM_FORCE_EXPLICIT_CTOR
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U, precision Q>
			GLM_FUNC_DECL explicit tvec4(tvec4<U, Q> const & v);
#		else
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U, precision Q>
			GLM_FUNC_DECL tvec4(tvec4<U, Q> const & v);
#		endif

		//////////////////////////////////////
		// Swizzle constructors

#		if GLM_HAS_ANONYMOUS_UNION && defined(GLM_SWIZZLE)
			template <int E0, int E1, int E2, int E3>
			GLM_FUNC_DECL tvec4(detail::_swizzle<4, T, P, tvec4<T, P>, E0, E1, E2, E3> const & that)
			{
				*this = that();
			}

			template <int E0, int E1, int F0, int F1>
			GLM_FUNC_DECL tvec4(detail::_swizzle<2, T, P, tvec2<T, P>, E0, E1, -1, -2> const & v, detail::_swizzle<2, T, P, tvec2<T, P>, F0, F1, -1, -2> const & u)
			{
				*this = tvec4<T, P>(v(), u());
			}

			template <int E0, int E1>
			GLM_FUNC_DECL tvec4(T const & x, T const & y, detail::_swizzle<2, T, P, tvec2<T, P>, E0, E1, -1, -2> const & v)
			{
				*this = tvec4<T, P>(x, y, v());
			}

			template <int E0, int E1>
			GLM_FUNC_DECL tvec4(T const & x, detail::_swizzle<2, T, P, tvec2<T, P>, E0, E1, -1, -2> const & v, T const & w)
			{
				*this = tvec4<T, P>(x, v(), w);
			}

			template <int E0, int E1>
			GLM_FUNC_DECL tvec4(detail::_swizzle<2, T, P, tvec2<T, P>, E0, E1, -1, -2> const & v, T const & z, T const & w)
			{
				*this = tvec4<T, P>(v(), z, w);
			}

			template <int E0, int E1, int E2>
			GLM_FUNC_DECL tvec4(detail::_swizzle<3, T, P, tvec3<T, P>, E0, E1, E2, -1> const & v, T const & w)
			{
				*this = tvec4<T, P>(v(), w);
			}

			template <int E0, int E1, int E2>
			GLM_FUNC_DECL tvec4(T const & x, detail::_swizzle<3, T, P, tvec3<T, P>, E0, E1, E2, -1> const & v)
			{
				*this = tvec4<T, P>(x, v());
			}
#		endif// GLM_HAS_ANONYMOUS_UNION && defined(GLM_SWIZZLE)

		//////////////////////////////////////
		// Unary arithmetic operators

		GLM_FUNC_DECL tvec4<T, P> & operator=(tvec4<T, P> const & v);

		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator+=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator+=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator+=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator-=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator-=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator-=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator*=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator*=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator*=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator/=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator/=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator/=(tvec4<U, P> const & v);

		//////////////////////////////////////
		// Increment and decrement operators

		GLM_FUNC_DECL tvec4<T, P> & operator++();
		GLM_FUNC_DECL tvec4<T, P> & operator--();
		GLM_FUNC_DECL tvec4<T, P> operator++(int);
		GLM_FUNC_DECL tvec4<T, P> operator--(int);

		//////////////////////////////////////
		// Unary bit operators

		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator%=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator%=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator%=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator&=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator&=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator&=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator|=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator|=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator|=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator^=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator^=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator^=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator<<=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator<<=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator<<=(tvec4<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator>>=(U scalar);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator>>=(tvec1<U, P> const & v);
		template <typename U>
		GLM_FUNC_DECL tvec4<T, P> & operator>>=(tvec4<U, P> const & v);
	};

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator+(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator+(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator+(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator+(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator+(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator-(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator-(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator-(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator-(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator-(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator*(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator*(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator*(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator*(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator*(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator/(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator/(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator/(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator/(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator/(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator-(tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL bool operator==(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL bool operator!=(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator%(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator%(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator%(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator%(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator%(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator&(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator&(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator&(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator&(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator&(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator|(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator|(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator|(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator|(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator|(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator^(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator^(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator^(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator^(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator^(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator<<(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator<<(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator<<(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator<<(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator<<(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator>>(tvec4<T, P> const & v, T scalar);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator>>(tvec4<T, P> const & v, tvec1<T, P> const & s);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator>>(T scalar, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator>>(tvec1<T, P> const & s, tvec4<T, P> const & v);

	template <typename T, precision P>
	GLM_FUNC_DECL tvec4<T, P> operator>>(tvec4<T, P> const & v1, tvec4<T, P> const & v2);

	template <typename T, precision P> 
	GLM_FUNC_DECL tvec4<T, P> operator~(tvec4<T, P> const & v);
}//namespace glm

#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_vec4.inl"
#endif//GLM_EXTERNAL_TEMPLATE
