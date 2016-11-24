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
/// @ref gtx_range
/// @file glm/gtx/range.hpp
/// @date 2014-09-19 / 2014-09-19
/// @author Joshua Moerman
///
/// @defgroup gtx_range GLM_GTX_range
/// @ingroup gtx
///
/// @brief Defines begin and end for vectors and matrices. Useful for range-based for loop.
/// The range is defined over the elements, not over columns or rows (e.g. mat4 has 16 elements).
///
/// <glm/gtx/range.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "../detail/setup.hpp"

#if !GLM_HAS_RANGE_FOR
#	error "GLM_GTX_range requires C++11 suppport or 'range for'"
#endif

#include "../gtc/type_ptr.hpp"

namespace glm{
namespace detail
{
	/* The glm types provide a .length() member, but for matrices
	 this only defines the number of columns, so we need to work around this */
	template <typename T, precision P>
	detail::component_count_t number_of_elements_(tvec2<T, P> const & v){
		return detail::component_count(v);
	}

	template <typename T, precision P>
	detail::component_count_t number_of_elements_(tvec3<T, P> const & v){
		return detail::component_count(v);
	}

	template <typename T, precision P>
	detail::component_count_t number_of_elements_(tvec4<T, P> const & v){
		return detail::component_count(v);
	}

	template <typename genType>
	detail::component_count_t number_of_elements_(genType const & m){
		return detail::component_count(m) * detail::component_count(m[0]);
	}
}//namespace

	/// @addtogroup gtx_range
	/// @{

	template <typename genType>
	const typename genType::value_type * begin(genType const & v){
		return value_ptr(v);
	}

	template <typename genType>
	const typename genType::value_type * end(genType const & v){
		return begin(v) + detail::number_of_elements_(v);
	}

	template <typename genType>
	typename genType::value_type * begin(genType& v){
		return value_ptr(v);
	}

	template <typename genType>
	typename genType::value_type * end(genType& v){
		return begin(v) + detail::number_of_elements_(v);
	}

	/// @}
}//namespace glm
