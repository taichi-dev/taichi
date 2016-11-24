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
/// @file glm/detail/_swizzle.hpp
/// @date 2006-04-20 / 2011-02-16
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace glm{
namespace detail
{
	// Internal class for implementing swizzle operators
	template <typename T, int N>
	struct _swizzle_base0
	{
		typedef T       value_type;

	protected:
		GLM_FUNC_QUALIFIER value_type&         elem   (size_t i)       { return (reinterpret_cast<value_type*>(_buffer))[i]; }
		GLM_FUNC_QUALIFIER const value_type&   elem   (size_t i) const { return (reinterpret_cast<const value_type*>(_buffer))[i]; }

		// Use an opaque buffer to *ensure* the compiler doesn't call a constructor.
		// The size 1 buffer is assumed to aligned to the actual members so that the
		// elem() 
		char    _buffer[1];
	};

	template <typename T, precision P, typename V, int E0, int E1, int E2, int E3, int N>
	struct _swizzle_base1 : public _swizzle_base0<T, N>
	{
	};

	template <typename T, precision P, typename V, int E0, int E1>
	struct _swizzle_base1<T, P, V,E0,E1,-1,-2,2> : public _swizzle_base0<T, 2>
	{
		GLM_FUNC_QUALIFIER V operator ()()  const { return V(this->elem(E0), this->elem(E1)); }
	};

	template <typename T, precision P, typename V, int E0, int E1, int E2>
	struct _swizzle_base1<T, P, V,E0,E1,E2,-1,3> : public _swizzle_base0<T, 3>
	{
		GLM_FUNC_QUALIFIER V operator ()()  const { return V(this->elem(E0), this->elem(E1), this->elem(E2)); }
	};

	template <typename T, precision P, typename V, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<T, P, V,E0,E1,E2,E3,4> : public _swizzle_base0<T, 4>
	{ 
		GLM_FUNC_QUALIFIER V operator ()()  const { return V(this->elem(E0), this->elem(E1), this->elem(E2), this->elem(E3)); }
	};

	// Internal class for implementing swizzle operators
	/*
		Template parameters:

		ValueType = type of scalar values (e.g. float, double)
		VecType   = class the swizzle is applies to (e.g. tvec3<float>)
		N         = number of components in the vector (e.g. 3)
		E0...3    = what index the n-th element of this swizzle refers to in the unswizzled vec

		DUPLICATE_ELEMENTS = 1 if there is a repeated element, 0 otherwise (used to specialize swizzles
			containing duplicate elements so that they cannot be used as r-values).            
	*/
	template <typename ValueType, precision P, typename VecType, int N, int E0, int E1, int E2, int E3, int DUPLICATE_ELEMENTS>
	struct _swizzle_base2 : public _swizzle_base1<ValueType, P, VecType,E0,E1,E2,E3,N>
	{
		typedef VecType vec_type;
		typedef ValueType value_type;

		GLM_FUNC_QUALIFIER _swizzle_base2& operator= (const ValueType& t)
		{
			for (int i = 0; i < N; ++i)
				(*this)[i] = t;
			return *this;
		}

		GLM_FUNC_QUALIFIER _swizzle_base2& operator= (const VecType& that)
		{
			struct op { 
				GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e = t; } 
			};
			_apply_op(that, op());
			return *this;
		}

		GLM_FUNC_QUALIFIER void operator -= (const VecType& that)
		{
			struct op { 
				GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e -= t; } 
			};
			_apply_op(that, op());
		}

		GLM_FUNC_QUALIFIER void operator += (const VecType& that)
		{
			struct op { 
				GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e += t; } 
			};
			_apply_op(that, op());
		}

		GLM_FUNC_QUALIFIER void operator *= (const VecType& that)
		{
			struct op { 
				GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e *= t; } 
			};
			_apply_op(that, op());
		}

		GLM_FUNC_QUALIFIER void operator /= (const VecType& that)
		{
			struct op { 
				GLM_FUNC_QUALIFIER void operator() (value_type& e, value_type& t) { e /= t; } 
			};
			_apply_op(that, op());
		}

		GLM_FUNC_QUALIFIER value_type& operator[]  (size_t i)
		{
			const int offset_dst[4] = { E0, E1, E2, E3 };
			return this->elem(offset_dst[i]);
		}
		GLM_FUNC_QUALIFIER value_type  operator[]  (size_t i) const
		{
			const int offset_dst[4] = { E0, E1, E2, E3 };
			return this->elem(offset_dst[i]);
		}

	protected:
		template <typename T>
		GLM_FUNC_QUALIFIER void _apply_op(const VecType& that, T op)
		{
			// Make a copy of the data in this == &that.
			// The copier should optimize out the copy in cases where the function is
			// properly inlined and the copy is not necessary.
			ValueType t[N];
			for (int i = 0; i < N; ++i)
				t[i] = that[i];
			for (int i = 0; i < N; ++i)
				op( (*this)[i], t[i] );
		}
	};

	// Specialization for swizzles containing duplicate elements.  These cannot be modified.
	template <typename ValueType, precision P, typename VecType, int N, int E0, int E1, int E2, int E3>
	struct _swizzle_base2<ValueType, P, VecType,N,E0,E1,E2,E3,1> : public _swizzle_base1<ValueType, P, VecType,E0,E1,E2,E3,N>
	{
		typedef VecType         vec_type;        
		typedef ValueType       value_type;

		struct Stub {};
		GLM_FUNC_QUALIFIER _swizzle_base2& operator= (Stub const &) { return *this; }

		GLM_FUNC_QUALIFIER value_type  operator[]  (size_t i) const
		{
			const int offset_dst[4] = { E0, E1, E2, E3 };
			return this->elem(offset_dst[i]);
		}
	};

	template <int N,typename ValueType, precision P, typename VecType, int E0,int E1,int E2,int E3>
	struct _swizzle : public _swizzle_base2<ValueType, P, VecType, N, E0, E1, E2, E3, (E0==E1||E0==E2||E0==E3||E1==E2||E1==E3||E2==E3)>
	{
		typedef _swizzle_base2<ValueType, P, VecType,N,E0,E1,E2,E3,(E0==E1||E0==E2||E0==E3||E1==E2||E1==E3||E2==E3)> base_type;

		using base_type::operator=;

		GLM_FUNC_QUALIFIER operator VecType () const { return (*this)(); }
	};

//
// To prevent the C++ syntax from getting entirely overwhelming, define some alias macros
//
#define _GLM_SWIZZLE_TEMPLATE1   template <int N, typename T, precision P, typename V, int E0, int E1, int E2, int E3>
#define _GLM_SWIZZLE_TEMPLATE2   template <int N, typename T, precision P, typename V, int E0, int E1, int E2, int E3, int F0, int F1, int F2, int F3>
#define _GLM_SWIZZLE_TYPE1       _swizzle<N, T, P, V, E0, E1, E2, E3>
#define _GLM_SWIZZLE_TYPE2       _swizzle<N, T, P, V, F0, F1, F2, F3>

//
// Wrapper for a binary operator (e.g. u.yy + v.zy)
//
#define _GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(OPERAND)                 \
	_GLM_SWIZZLE_TEMPLATE2                                                          \
	GLM_FUNC_QUALIFIER V operator OPERAND ( const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE2& b)  \
	{                                                                               \
		return a() OPERAND b();                                                     \
	}                                                                               \
	_GLM_SWIZZLE_TEMPLATE1                                                          \
	GLM_FUNC_QUALIFIER V operator OPERAND ( const _GLM_SWIZZLE_TYPE1& a, const V& b)                   \
	{                                                                               \
		return a() OPERAND b;                                                       \
	}                                                                               \
	_GLM_SWIZZLE_TEMPLATE1                                                          \
	GLM_FUNC_QUALIFIER V operator OPERAND ( const V& a, const _GLM_SWIZZLE_TYPE1& b)                   \
	{                                                                               \
		return a OPERAND b();                                                       \
	}

//
// Wrapper for a operand between a swizzle and a binary (e.g. 1.0f - u.xyz)
//
#define _GLM_SWIZZLE_SCALAR_BINARY_OPERATOR_IMPLEMENTATION(OPERAND)                 \
	_GLM_SWIZZLE_TEMPLATE1                                                          \
	GLM_FUNC_QUALIFIER V operator OPERAND ( const _GLM_SWIZZLE_TYPE1& a, const T& b)                   \
	{                                                                               \
		return a() OPERAND b;                                                       \
	}                                                                               \
	_GLM_SWIZZLE_TEMPLATE1                                                          \
	GLM_FUNC_QUALIFIER V operator OPERAND ( const T& a, const _GLM_SWIZZLE_TYPE1& b)                   \
	{                                                                               \
		return a OPERAND b();                                                       \
	}

//
// Macro for wrapping a function taking one argument (e.g. abs())
//
#define _GLM_SWIZZLE_FUNCTION_1_ARGS(RETURN_TYPE,FUNCTION)                          \
	_GLM_SWIZZLE_TEMPLATE1                                                          \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a)  \
	{                                                                               \
		return FUNCTION(a());                                                       \
	}

//
// Macro for wrapping a function taking two vector arguments (e.g. dot()).
//
#define _GLM_SWIZZLE_FUNCTION_2_ARGS(RETURN_TYPE,FUNCTION)                                                      \
	_GLM_SWIZZLE_TEMPLATE2                                                                                      \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE2& b) \
	{                                                                                                           \
		return FUNCTION(a(), b());                                                                              \
	}                                                                                                           \
	_GLM_SWIZZLE_TEMPLATE1                                                                                      \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE1& b) \
	{                                                                                                           \
		return FUNCTION(a(), b());                                                                              \
	}                                                                                                           \
	_GLM_SWIZZLE_TEMPLATE1                                                                                      \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const typename V& b)         \
	{                                                                                                           \
		return FUNCTION(a(), b);                                                                                \
	}                                                                                                           \
	_GLM_SWIZZLE_TEMPLATE1                                                                                      \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const V& a, const _GLM_SWIZZLE_TYPE1& b)                  \
	{                                                                                                           \
		return FUNCTION(a, b());                                                                                \
	} 

//
// Macro for wrapping a function take 2 vec arguments followed by a scalar (e.g. mix()).
//
#define _GLM_SWIZZLE_FUNCTION_2_ARGS_SCALAR(RETURN_TYPE,FUNCTION)                                                             \
	_GLM_SWIZZLE_TEMPLATE2                                                                                                    \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE2& b, const T& c)   \
	{                                                                                                                         \
		return FUNCTION(a(), b(), c);                                                                                         \
	}                                                                                                                         \
	_GLM_SWIZZLE_TEMPLATE1                                                                                                    \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const _GLM_SWIZZLE_TYPE1& b, const T& c)   \
	{                                                                                                                         \
		return FUNCTION(a(), b(), c);                                                                                         \
	}                                                                                                                         \
	_GLM_SWIZZLE_TEMPLATE1                                                                                                    \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const _GLM_SWIZZLE_TYPE1& a, const typename S0::vec_type& b, const T& c)\
	{                                                                                                                         \
		return FUNCTION(a(), b, c);                                                                                           \
	}                                                                                                                         \
	_GLM_SWIZZLE_TEMPLATE1                                                                                                    \
	GLM_FUNC_QUALIFIER typename _GLM_SWIZZLE_TYPE1::RETURN_TYPE FUNCTION(const typename V& a, const _GLM_SWIZZLE_TYPE1& b, const T& c)           \
	{                                                                                                                         \
		return FUNCTION(a, b(), c);                                                                                           \
	} 
 
}//namespace detail 
}//namespace glm

namespace glm
{
	namespace detail
	{
		_GLM_SWIZZLE_SCALAR_BINARY_OPERATOR_IMPLEMENTATION(-)
		_GLM_SWIZZLE_SCALAR_BINARY_OPERATOR_IMPLEMENTATION(*)
		_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(+)
		_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(-)
		_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(*)
		_GLM_SWIZZLE_VECTOR_BINARY_OPERATOR_IMPLEMENTATION(/)
	}

	//
	// Swizzles are distinct types from the unswizzled type.  The below macros will
	// provide template specializations for the swizzle types for the given functions
	// so that the compiler does not have any ambiguity to choosing how to handle
	// the function.
	//
	// The alternative is to use the operator()() when calling the function in order
	// to explicitly convert the swizzled type to the unswizzled type.
	//

	//_GLM_SWIZZLE_FUNCTION_1_ARGS(vec_type,    abs);
	//_GLM_SWIZZLE_FUNCTION_1_ARGS(vec_type,    acos);
	//_GLM_SWIZZLE_FUNCTION_1_ARGS(vec_type,    acosh);
	//_GLM_SWIZZLE_FUNCTION_1_ARGS(vec_type,    all);
	//_GLM_SWIZZLE_FUNCTION_1_ARGS(vec_type,    any);

	//_GLM_SWIZZLE_FUNCTION_2_ARGS(value_type,  dot);
	//_GLM_SWIZZLE_FUNCTION_2_ARGS(vec_type,    cross);
	//_GLM_SWIZZLE_FUNCTION_2_ARGS(vec_type,    step);    
	//_GLM_SWIZZLE_FUNCTION_2_ARGS_SCALAR(vec_type, mix);
}

#define _GLM_SWIZZLE2_2_MEMBERS(T, P, V, E0,E1) \
	struct { detail::_swizzle<2, T, P, V<T, P>, 0,0,-1,-2> E0 ## E0; }; \
	struct { detail::_swizzle<2, T, P, V<T, P>, 0,1,-1,-2> E0 ## E1; }; \
	struct { detail::_swizzle<2, T, P, V<T, P>, 1,0,-1,-2> E1 ## E0; }; \
	struct { detail::_swizzle<2, T, P, V<T, P>, 1,1,-1,-2> E1 ## E1; }; 

#define _GLM_SWIZZLE2_3_MEMBERS(T, P, V, E0,E1) \
	struct { detail::_swizzle<3,T, P, V<T, P>, 0,0,0,-1> E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 0,0,1,-1> E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 0,1,0,-1> E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 0,1,1,-1> E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 1,0,0,-1> E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 1,0,1,-1> E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 1,1,0,-1> E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T, P, V<T, P>, 1,1,1,-1> E1 ## E1 ## E1; };  

#define _GLM_SWIZZLE2_4_MEMBERS(T, P, V, E0,E1) \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,0> E0 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,1> E0 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,0> E0 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,1> E0 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,0> E0 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,1> E0 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,0> E0 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,1> E0 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,0> E1 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,1> E1 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,0> E1 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,1> E1 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,0> E1 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,1> E1 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,0> E1 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,1> E1 ## E1 ## E1 ## E1; };  

#define _GLM_SWIZZLE3_2_MEMBERS(T, P, V, E0,E1,E2) \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,0,-1,-2> E0 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,1,-1,-2> E0 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,2,-1,-2> E0 ## E2; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,0,-1,-2> E1 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,1,-1,-2> E1 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,2,-1,-2> E1 ## E2; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,0,-1,-2> E2 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,1,-1,-2> E2 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,2,-1,-2> E2 ## E2; }; 

#define _GLM_SWIZZLE3_3_MEMBERS(T, P, V ,E0,E1,E2) \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,0,-1> E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,1,-1> E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,2,-1> E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,0,-1> E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,1,-1> E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,2,-1> E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,0,-1> E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,1,-1> E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,2,-1> E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,0,-1> E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,1,-1> E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,2,-1> E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,0,-1> E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,1,-1> E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,2,-1> E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,0,-1> E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,1,-1> E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,2,-1> E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,0,-1> E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,1,-1> E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,2,-1> E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,0,-1> E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,1,-1> E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,2,-1> E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,0,-1> E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,1,-1> E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,2,-1> E2 ## E2 ## E2; };

#define _GLM_SWIZZLE3_4_MEMBERS(T, P, V, E0,E1,E2) \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,0> E0 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,1> E0 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,0,2> E0 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,0> E0 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,1> E0 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,1,2> E0 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,2,0> E0 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,2,1> E0 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,0,2,2> E0 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,0> E0 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,1> E0 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,0,2> E0 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,0> E0 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,1> E0 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,1,2> E0 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,2,0> E0 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,2,1> E0 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,1,2,2> E0 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,0,0> E0 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,0,1> E0 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,0,2> E0 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,1,0> E0 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,1,1> E0 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,1,2> E0 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,2,0> E0 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,2,1> E0 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 0,2,2,2> E0 ## E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,0> E1 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,1> E1 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,0,2> E1 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,0> E1 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,1> E1 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,1,2> E1 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,2,0> E1 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,2,1> E1 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,0,2,2> E1 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,0> E1 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,1> E1 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,0,2> E1 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,0> E1 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,1> E1 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,1,2> E1 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,2,0> E1 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,2,1> E1 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,1,2,2> E1 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,0,0> E1 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,0,1> E1 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,0,2> E1 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,1,0> E1 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,1,1> E1 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,1,2> E1 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,2,0> E1 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,2,1> E1 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 1,2,2,2> E1 ## E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,0,0> E2 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,0,1> E2 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,0,2> E2 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,1,0> E2 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,1,1> E2 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,1,2> E2 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,2,0> E2 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,2,1> E2 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,0,2,2> E2 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,0,0> E2 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,0,1> E2 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,0,2> E2 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,1,0> E2 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,1,1> E2 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,1,2> E2 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,2,0> E2 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,2,1> E2 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,1,2,2> E2 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,0,0> E2 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,0,1> E2 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,0,2> E2 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,1,0> E2 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,1,1> E2 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,1,2> E2 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,2,0> E2 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,2,1> E2 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4,T, P, V<T, P>, 2,2,2,2> E2 ## E2 ## E2 ## E2; }; 

#define _GLM_SWIZZLE4_2_MEMBERS(T, P, V, E0,E1,E2,E3) \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,0,-1,-2> E0 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,1,-1,-2> E0 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,2,-1,-2> E0 ## E2; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 0,3,-1,-2> E0 ## E3; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,0,-1,-2> E1 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,1,-1,-2> E1 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,2,-1,-2> E1 ## E2; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 1,3,-1,-2> E1 ## E3; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,0,-1,-2> E2 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,1,-1,-2> E2 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,2,-1,-2> E2 ## E2; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 2,3,-1,-2> E2 ## E3; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 3,0,-1,-2> E3 ## E0; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 3,1,-1,-2> E3 ## E1; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 3,2,-1,-2> E3 ## E2; }; \
	struct { detail::_swizzle<2,T, P, V<T, P>, 3,3,-1,-2> E3 ## E3; }; 

#define _GLM_SWIZZLE4_3_MEMBERS(T,P, V, E0,E1,E2,E3) \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,0,-1> E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,1,-1> E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,2,-1> E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,0,3,-1> E0 ## E0 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,0,-1> E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,1,-1> E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,2,-1> E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,1,3,-1> E0 ## E1 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,0,-1> E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,1,-1> E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,2,-1> E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,2,3,-1> E0 ## E2 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,0,-1> E0 ## E3 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,1,-1> E0 ## E3 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,2,-1> E0 ## E3 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 0,3,3,-1> E0 ## E3 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,0,-1> E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,1,-1> E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,2,-1> E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,0,3,-1> E1 ## E0 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,0,-1> E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,1,-1> E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,2,-1> E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,1,3,-1> E1 ## E1 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,0,-1> E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,1,-1> E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,2,-1> E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,2,3,-1> E1 ## E2 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,0,-1> E1 ## E3 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,1,-1> E1 ## E3 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,2,-1> E1 ## E3 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 1,3,3,-1> E1 ## E3 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,0,-1> E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,1,-1> E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,2,-1> E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,0,3,-1> E2 ## E0 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,0,-1> E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,1,-1> E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,2,-1> E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,1,3,-1> E2 ## E1 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,0,-1> E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,1,-1> E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,2,-1> E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,2,3,-1> E2 ## E2 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,0,-1> E2 ## E3 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,1,-1> E2 ## E3 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,2,-1> E2 ## E3 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 2,3,3,-1> E2 ## E3 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,0,-1> E3 ## E0 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,1,-1> E3 ## E0 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,2,-1> E3 ## E0 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,0,3,-1> E3 ## E0 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,0,-1> E3 ## E1 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,1,-1> E3 ## E1 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,2,-1> E3 ## E1 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,1,3,-1> E3 ## E1 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,0,-1> E3 ## E2 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,1,-1> E3 ## E2 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,2,-1> E3 ## E2 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,2,3,-1> E3 ## E2 ## E3; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,0,-1> E3 ## E3 ## E0; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,1,-1> E3 ## E3 ## E1; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,2,-1> E3 ## E3 ## E2; }; \
	struct { detail::_swizzle<3,T,P, V<T, P>, 3,3,3,-1> E3 ## E3 ## E3; };  

#define _GLM_SWIZZLE4_4_MEMBERS(T, P, V, E0,E1,E2,E3) \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,0> E0 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,1> E0 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,2> E0 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,0,3> E0 ## E0 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,0> E0 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,1> E0 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,2> E0 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,1,3> E0 ## E0 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,0> E0 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,1> E0 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,2> E0 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,2,3> E0 ## E0 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,0> E0 ## E0 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,1> E0 ## E0 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,2> E0 ## E0 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,0,3,3> E0 ## E0 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,0> E0 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,1> E0 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,2> E0 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,0,3> E0 ## E1 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,0> E0 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,1> E0 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,2> E0 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,1,3> E0 ## E1 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,0> E0 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,1> E0 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,2> E0 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,2,3> E0 ## E1 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,0> E0 ## E1 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,1> E0 ## E1 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,2> E0 ## E1 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,1,3,3> E0 ## E1 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,0> E0 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,1> E0 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,2> E0 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,0,3> E0 ## E2 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,0> E0 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,1> E0 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,2> E0 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,1,3> E0 ## E2 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,0> E0 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,1> E0 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,2> E0 ## E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,2,3> E0 ## E2 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,0> E0 ## E2 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,1> E0 ## E2 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,2> E0 ## E2 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,2,3,3> E0 ## E2 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,0> E0 ## E3 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,1> E0 ## E3 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,2> E0 ## E3 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,0,3> E0 ## E3 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,0> E0 ## E3 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,1> E0 ## E3 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,2> E0 ## E3 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,1,3> E0 ## E3 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,0> E0 ## E3 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,1> E0 ## E3 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,2> E0 ## E3 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,2,3> E0 ## E3 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,0> E0 ## E3 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,1> E0 ## E3 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,2> E0 ## E3 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 0,3,3,3> E0 ## E3 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,0> E1 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,1> E1 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,2> E1 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,0,3> E1 ## E0 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,0> E1 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,1> E1 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,2> E1 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,1,3> E1 ## E0 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,0> E1 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,1> E1 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,2> E1 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,2,3> E1 ## E0 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,0> E1 ## E0 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,1> E1 ## E0 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,2> E1 ## E0 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,0,3,3> E1 ## E0 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,0> E1 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,1> E1 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,2> E1 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,0,3> E1 ## E1 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,0> E1 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,1> E1 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,2> E1 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,1,3> E1 ## E1 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,0> E1 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,1> E1 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,2> E1 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,2,3> E1 ## E1 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,0> E1 ## E1 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,1> E1 ## E1 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,2> E1 ## E1 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,1,3,3> E1 ## E1 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,0> E1 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,1> E1 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,2> E1 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,0,3> E1 ## E2 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,0> E1 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,1> E1 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,2> E1 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,1,3> E1 ## E2 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,0> E1 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,1> E1 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,2> E1 ## E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,2,3> E1 ## E2 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,0> E1 ## E2 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,1> E1 ## E2 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,2> E1 ## E2 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,2,3,3> E1 ## E2 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,0> E1 ## E3 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,1> E1 ## E3 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,2> E1 ## E3 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,0,3> E1 ## E3 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,0> E1 ## E3 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,1> E1 ## E3 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,2> E1 ## E3 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,1,3> E1 ## E3 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,0> E1 ## E3 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,1> E1 ## E3 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,2> E1 ## E3 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,2,3> E1 ## E3 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,0> E1 ## E3 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,1> E1 ## E3 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,2> E1 ## E3 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 1,3,3,3> E1 ## E3 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,0> E2 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,1> E2 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,2> E2 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,0,3> E2 ## E0 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,0> E2 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,1> E2 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,2> E2 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,1,3> E2 ## E0 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,0> E2 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,1> E2 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,2> E2 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,2,3> E2 ## E0 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,0> E2 ## E0 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,1> E2 ## E0 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,2> E2 ## E0 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,0,3,3> E2 ## E0 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,0> E2 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,1> E2 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,2> E2 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,0,3> E2 ## E1 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,0> E2 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,1> E2 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,2> E2 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,1,3> E2 ## E1 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,0> E2 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,1> E2 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,2> E2 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,2,3> E2 ## E1 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,0> E2 ## E1 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,1> E2 ## E1 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,2> E2 ## E1 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,1,3,3> E2 ## E1 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,0> E2 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,1> E2 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,2> E2 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,0,3> E2 ## E2 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,0> E2 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,1> E2 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,2> E2 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,1,3> E2 ## E2 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,0> E2 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,1> E2 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,2> E2 ## E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,2,3> E2 ## E2 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,0> E2 ## E2 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,1> E2 ## E2 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,2> E2 ## E2 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,2,3,3> E2 ## E2 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,0> E2 ## E3 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,1> E2 ## E3 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,2> E2 ## E3 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,0,3> E2 ## E3 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,0> E2 ## E3 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,1> E2 ## E3 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,2> E2 ## E3 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,1,3> E2 ## E3 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,0> E2 ## E3 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,1> E2 ## E3 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,2> E2 ## E3 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,2,3> E2 ## E3 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,0> E2 ## E3 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,1> E2 ## E3 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,2> E2 ## E3 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 2,3,3,3> E2 ## E3 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,0> E3 ## E0 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,1> E3 ## E0 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,2> E3 ## E0 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,0,3> E3 ## E0 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,0> E3 ## E0 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,1> E3 ## E0 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,2> E3 ## E0 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,1,3> E3 ## E0 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,0> E3 ## E0 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,1> E3 ## E0 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,2> E3 ## E0 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,2,3> E3 ## E0 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,0> E3 ## E0 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,1> E3 ## E0 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,2> E3 ## E0 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,0,3,3> E3 ## E0 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,0> E3 ## E1 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,1> E3 ## E1 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,2> E3 ## E1 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,0,3> E3 ## E1 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,0> E3 ## E1 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,1> E3 ## E1 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,2> E3 ## E1 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,1,3> E3 ## E1 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,0> E3 ## E1 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,1> E3 ## E1 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,2> E3 ## E1 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,2,3> E3 ## E1 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,0> E3 ## E1 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,1> E3 ## E1 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,2> E3 ## E1 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,1,3,3> E3 ## E1 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,0> E3 ## E2 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,1> E3 ## E2 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,2> E3 ## E2 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,0,3> E3 ## E2 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,0> E3 ## E2 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,1> E3 ## E2 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,2> E3 ## E2 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,1,3> E3 ## E2 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,0> E3 ## E2 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,1> E3 ## E2 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,2> E3 ## E2 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,2,3> E3 ## E2 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,0> E3 ## E2 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,1> E3 ## E2 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,2> E3 ## E2 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,2,3,3> E3 ## E2 ## E3 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,0> E3 ## E3 ## E0 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,1> E3 ## E3 ## E0 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,2> E3 ## E3 ## E0 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,0,3> E3 ## E3 ## E0 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,0> E3 ## E3 ## E1 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,1> E3 ## E3 ## E1 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,2> E3 ## E3 ## E1 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,1,3> E3 ## E3 ## E1 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,0> E3 ## E3 ## E2 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,1> E3 ## E3 ## E2 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,2> E3 ## E3 ## E2 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,2,3> E3 ## E3 ## E2 ## E3; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,0> E3 ## E3 ## E3 ## E0; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,1> E3 ## E3 ## E3 ## E1; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,2> E3 ## E3 ## E3 ## E2; }; \
	struct { detail::_swizzle<4, T, P, V<T, P>, 3,3,3,3> E3 ## E3 ## E3 ## E3; };
