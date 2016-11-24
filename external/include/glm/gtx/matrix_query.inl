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
/// @ref gtx_matrix_query
/// @file glm/gtx/matrix_query.inl
/// @date 2007-03-05 / 2007-03-05
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template<typename T, precision P>
	GLM_FUNC_QUALIFIER bool isNull(tmat2x2<T, P> const & m, T const & epsilon)
	{
		bool result = true;
		for(detail::component_count_t i = 0; result && i < 2 ; ++i)
			result = isNull(m[i], epsilon);
		return result;
	}

	template<typename T, precision P>
	GLM_FUNC_QUALIFIER bool isNull(tmat3x3<T, P> const & m, T const & epsilon)
	{
		bool result = true;
		for(detail::component_count_t i = 0; result && i < 3 ; ++i)
			result = isNull(m[i], epsilon);
		return result;
	}

	template<typename T, precision P>
	GLM_FUNC_QUALIFIER bool isNull(tmat4x4<T, P> const & m, T const & epsilon)
	{
		bool result = true;
		for(detail::component_count_t i = 0; result && i < 4 ; ++i)
			result = isNull(m[i], epsilon);
		return result;
	}

	template<typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_QUALIFIER bool isIdentity(matType<T, P> const & m, T const & epsilon)
	{
		bool result = true;
		for(detail::component_count_t i(0); result && i < detail::component_count(m[0]); ++i)
		{
			for(detail::component_count_t j(0); result && j < i ; ++j)
				result = abs(m[i][j]) <= epsilon;
			if(result)
				result = abs(m[i][i] - 1) <= epsilon;
			for(detail::component_count_t j(i + 1); result && j < detail::component_count(m); ++j)
				result = abs(m[i][j]) <= epsilon;
		}
		return result;
	}

	template<typename T, precision P>
	GLM_FUNC_QUALIFIER bool isNormalized(tmat2x2<T, P> const & m, T const & epsilon)
	{
		bool result(true);
		for(detail::component_count_t i(0); result && i < detail::component_count(m); ++i)
			result = isNormalized(m[i], epsilon);
		for(detail::component_count_t i(0); result && i < detail::component_count(m); ++i)
		{
			typename tmat2x2<T, P>::col_type v;
			for(detail::component_count_t j(0); j < detail::component_count(m); ++j)
				v[j] = m[j][i];
			result = isNormalized(v, epsilon);
		}
		return result;
	}

	template<typename T, precision P>
	GLM_FUNC_QUALIFIER bool isNormalized(tmat3x3<T, P> const & m, T const & epsilon)
	{
		bool result(true);
		for(detail::component_count_t i(0); result && i < detail::component_count(m); ++i)
			result = isNormalized(m[i], epsilon);
		for(detail::component_count_t i(0); result && i < detail::component_count(m); ++i)
		{
			typename tmat3x3<T, P>::col_type v;
			for(detail::component_count_t j(0); j < detail::component_count(m); ++j)
				v[j] = m[j][i];
			result = isNormalized(v, epsilon);
		}
		return result;
	}

	template<typename T, precision P>
	GLM_FUNC_QUALIFIER bool isNormalized(tmat4x4<T, P> const & m, T const & epsilon)
	{
		bool result(true);
		for(detail::component_count_t i(0); result && i < detail::component_count(m); ++i)
			result = isNormalized(m[i], epsilon);
		for(detail::component_count_t i(0); result && i < detail::component_count(m); ++i)
		{
			typename tmat4x4<T, P>::col_type v;
			for(detail::component_count_t j(0); j < detail::component_count(m); ++j)
				v[j] = m[j][i];
			result = isNormalized(v, epsilon);
		}
		return result;
	}

	template<typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_QUALIFIER bool isOrthogonal(matType<T, P> const & m, T const & epsilon)
	{
		bool result(true);
		for(detail::component_count_t i(0); result && i < detail::component_count(m) - 1; ++i)
		for(detail::component_count_t j(i + 1); result && j < detail::component_count(m); ++j)
			result = areOrthogonal(m[i], m[j], epsilon);

		if(result)
		{
			matType<T, P> tmp = transpose(m);
			for(detail::component_count_t i(0); result && i < detail::component_count(m) - 1 ; ++i)
			for(detail::component_count_t j(i + 1); result && j < detail::component_count(m); ++j)
				result = areOrthogonal(tmp[i], tmp[j], epsilon);
		}
		return result;
	}
}//namespace glm
