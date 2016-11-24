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
/// @ref gtc_type_ptr
/// @file glm/gtc/type_ptr.inl
/// @date 2011-06-15 / 2011-12-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <cstring>

namespace glm
{
	/// @addtogroup gtc_type_ptr
	/// @{

	/// Return the constant address to the data of the vector input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tvec2<T, P> const & vec
	)
	{
		return &(vec.x);
	}

	//! Return the address to the data of the vector input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tvec2<T, P> & vec
	)
	{
		return &(vec.x);
	}

	/// Return the constant address to the data of the vector input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tvec3<T, P> const & vec
	)
	{
		return &(vec.x);
	}

	//! Return the address to the data of the vector input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tvec3<T, P> & vec
	)
	{
		return &(vec.x);
	}
		
	/// Return the constant address to the data of the vector input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(	
		tvec4<T, P> const & vec
	)
	{
		return &(vec.x);
	}

	//! Return the address to the data of the vector input.
	//! From GLM_GTC_type_ptr extension.
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(	
		tvec4<T, P> & vec
	)
	{
		return &(vec.x);
	}

	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat2x2<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat2x2<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat3x3<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat3x3<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat4x4<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	//! From GLM_GTC_type_ptr extension.
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat4x4<T, P> & mat
	)
	{
		return &(mat[0].x);
	}

	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat2x3<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat2x3<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat3x2<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat3x2<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat2x4<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat2x4<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat4x2<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(	
		tmat4x2<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat3x4<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	//! Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tmat3x4<T, P> & mat
	)
	{
		return &(mat[0].x);
	}
		
	/// Return the constant address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tmat4x3<T, P> const & mat
	)
	{
		return &(mat[0].x);
	}

	/// Return the address to the data of the matrix input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr(tmat4x3<T, P> & mat)
	{
		return &(mat[0].x);
	}

	/// Return the constant address to the data of the input parameter.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T const * value_ptr
	(
		tquat<T, P> const & q
	)
	{
		return &(q[0]);
	}

	/// Return the address to the data of the quaternion input.
	/// @see gtc_type_ptr
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER T * value_ptr
	(
		tquat<T, P> & q
	)
	{
		return &(q[0]);
	}

	/// Build a vector from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tvec2<T, defaultp> make_vec2(T const * const ptr)
	{
		tvec2<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tvec2<T, defaultp>));
		return Result;
	}

	/// Build a vector from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tvec3<T, defaultp> make_vec3(T const * const ptr)
	{
		tvec3<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tvec3<T, defaultp>));
		return Result;
	}

	/// Build a vector from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tvec4<T, defaultp> make_vec4(T const * const ptr)
	{
		tvec4<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tvec4<T, defaultp>));
		return Result;
	}

	/// Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat2x2<T, defaultp> make_mat2x2(T const * const ptr)
	{
		tmat2x2<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat2x2<T, defaultp>));
		return Result;
	}

	/// Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat2x3<T, defaultp> make_mat2x3(T const * const ptr)
	{
		tmat2x3<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat2x3<T, defaultp>));
		return Result;
	}

	/// Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat2x4<T, defaultp> make_mat2x4(T const * const ptr)
	{
		tmat2x4<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat2x4<T, defaultp>));
		return Result;
	}

	/// Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat3x2<T, defaultp> make_mat3x2(T const * const ptr)
	{
		tmat3x2<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat3x2<T, defaultp>));
		return Result;
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat3x3<T, defaultp> make_mat3x3(T const * const ptr)
	{
		tmat3x3<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat3x3<T, defaultp>));
		return Result;
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat3x4<T, defaultp> make_mat3x4(T const * const ptr)
	{
		tmat3x4<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat3x4<T, defaultp>));
		return Result;
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat4x2<T, defaultp> make_mat4x2(T const * const ptr)
	{
		tmat4x2<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat4x2<T, defaultp>));
		return Result;
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat4x3<T, defaultp> make_mat4x3(T const * const ptr)
	{
		tmat4x3<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat4x3<T, defaultp>));
		return Result;
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> make_mat4x4(T const * const ptr)
	{
		tmat4x4<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tmat4x4<T, defaultp>));
		return Result;
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat2x2<T, defaultp> make_mat2(T const * const ptr)
	{
		return make_mat2x2(ptr);
	}

	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat3x3<T, defaultp> make_mat3(T const * const ptr)
	{
		return make_mat3x3(ptr);
	}
		
	//! Build a matrix from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> make_mat4(T const * const ptr)
	{
		return make_mat4x4(ptr);
	}

	//! Build a quaternion from a pointer.
	/// @see gtc_type_ptr
	template <typename T>
	GLM_FUNC_QUALIFIER tquat<T, defaultp> make_quat(T const * const ptr)
	{
		tquat<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(tquat<T, defaultp>));
		return Result;
	}

	/// @}
}//namespace glm

