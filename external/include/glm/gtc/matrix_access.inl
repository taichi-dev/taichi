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
/// @ref gtc_matrix_access
/// @file glm/gtc/matrix_access.inl
/// @date 2005-12-27 / 2011-06-05
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename genType>
	GLM_FUNC_QUALIFIER genType row
	(
		genType const & m,
		length_t index,
		typename genType::row_type const & x
	)
	{
		assert(index >= 0 && static_cast<detail::component_count_t>(index) < detail::component_count(m[0]));

		genType Result = m;
		for(detail::component_count_t i = 0; i < detail::component_count(m); ++i)
			Result[i][index] = x[i];
		return Result;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER typename genType::row_type row
	(
		genType const & m,
		length_t index
	)
	{
		assert(index >= 0 && static_cast<detail::component_count_t>(index) < detail::component_count(m[0]));

		typename genType::row_type Result;
		for(detail::component_count_t i = 0; i < detail::component_count(m); ++i)
			Result[i] = m[i][index];
		return Result;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType column
	(
		genType const & m,
		length_t index,
		typename genType::col_type const & x
	)
	{
		assert(index >= 0 && static_cast<detail::component_count_t>(index) < detail::component_count(m));

		genType Result = m;
		Result[index] = x;
		return Result;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER typename genType::col_type column
	(
		genType const & m,
		length_t index
	)
	{
		assert(index >= 0 && static_cast<detail::component_count_t>(index) < detail::component_count(m));

		return m[index];
	}
}//namespace glm
