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
/// @file glm/detail/func_matrix.inl
/// @date 2008-03-08 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "../geometric.hpp"
#include <limits>

namespace glm{
namespace detail
{
	template <template <class, precision> class matType, typename T, precision P>
	struct compute_transpose{};

	template <typename T, precision P>
	struct compute_transpose<tmat2x2, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat2x2<T, P> call(tmat2x2<T, P> const & m)
		{
			tmat2x2<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat2x3, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat3x2<T, P> call(tmat2x3<T, P> const & m)
		{
			tmat3x2<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[2][0] = m[0][2];
			result[2][1] = m[1][2];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat2x4, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat4x2<T, P> call(tmat2x4<T, P> const & m)
		{
			tmat4x2<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[2][0] = m[0][2];
			result[2][1] = m[1][2];
			result[3][0] = m[0][3];
			result[3][1] = m[1][3];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat3x2, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat2x3<T, P> call(tmat3x2<T, P> const & m)
		{
			tmat2x3<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[0][2] = m[2][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[1][2] = m[2][1];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat3x3, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat3x3<T, P> call(tmat3x3<T, P> const & m)
		{
			tmat3x3<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[0][2] = m[2][0];

			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[1][2] = m[2][1];

			result[2][0] = m[0][2];
			result[2][1] = m[1][2];
			result[2][2] = m[2][2];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat3x4, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat4x3<T, P> call(tmat3x4<T, P> const & m)
		{
			tmat4x3<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[0][2] = m[2][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[1][2] = m[2][1];
			result[2][0] = m[0][2];
			result[2][1] = m[1][2];
			result[2][2] = m[2][2];
			result[3][0] = m[0][3];
			result[3][1] = m[1][3];
			result[3][2] = m[2][3];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat4x2, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat2x4<T, P> call(tmat4x2<T, P> const & m)
		{
			tmat2x4<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[0][2] = m[2][0];
			result[0][3] = m[3][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[1][2] = m[2][1];
			result[1][3] = m[3][1];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat4x3, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat3x4<T, P> call(tmat4x3<T, P> const & m)
		{
			tmat3x4<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[0][2] = m[2][0];
			result[0][3] = m[3][0];
			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[1][2] = m[2][1];
			result[1][3] = m[3][1];
			result[2][0] = m[0][2];
			result[2][1] = m[1][2];
			result[2][2] = m[2][2];
			result[2][3] = m[3][2];
			return result;
		}
	};

	template <typename T, precision P>
	struct compute_transpose<tmat4x4, T, P>
	{
		GLM_FUNC_QUALIFIER static tmat4x4<T, P> call(tmat4x4<T, P> const & m)
		{
			tmat4x4<T, P> result(uninitialize);
			result[0][0] = m[0][0];
			result[0][1] = m[1][0];
			result[0][2] = m[2][0];
			result[0][3] = m[3][0];

			result[1][0] = m[0][1];
			result[1][1] = m[1][1];
			result[1][2] = m[2][1];
			result[1][3] = m[3][1];

			result[2][0] = m[0][2];
			result[2][1] = m[1][2];
			result[2][2] = m[2][2];
			result[2][3] = m[3][2];

			result[3][0] = m[0][3];
			result[3][1] = m[1][3];
			result[3][2] = m[2][3];
			result[3][3] = m[3][3];
			return result;
		}
	};

	template <template <class, precision> class matType, typename T, precision P>
	struct compute_determinant{};

	template <typename T, precision P>
	struct compute_determinant<tmat2x2, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tmat2x2<T, P> const & m)
		{
			return m[0][0] * m[1][1] - m[1][0] * m[0][1];
		}
	};

	template <typename T, precision P>
	struct compute_determinant<tmat3x3, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tmat3x3<T, P> const & m)
		{
			return
				+ m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
				- m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
				+ m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
		}
	};

	template <typename T, precision P>
	struct compute_determinant<tmat4x4, T, P>
	{
		GLM_FUNC_QUALIFIER static T call(tmat4x4<T, P> const & m)
		{
			T SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
			T SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
			T SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
			T SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
			T SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
			T SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

			tvec4<T, P> DetCof(
				+ (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02),
				- (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04),
				+ (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05),
				- (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05));

			return
				m[0][0] * DetCof[0] + m[0][1] * DetCof[1] +
				m[0][2] * DetCof[2] + m[0][3] * DetCof[3];
		}
	};
}//namespace detail

	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_QUALIFIER matType<T, P> matrixCompMult(matType<T, P> const & x, matType<T, P> const & y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'matrixCompMult' only accept floating-point inputs");

		matType<T, P> result(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(result); ++i)
			result[i] = x[i] * y[i];
		return result;
	}

	template<typename T, precision P, template <typename, precision> class vecTypeA, template <typename, precision> class vecTypeB>
	GLM_FUNC_QUALIFIER typename detail::outerProduct_trait<T, P, vecTypeA, vecTypeB>::type outerProduct(vecTypeA<T, P> const & c, vecTypeB<T, P> const & r)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'outerProduct' only accept floating-point inputs");

		typename detail::outerProduct_trait<T, P, vecTypeA, vecTypeB>::type m(uninitialize);
		for(detail::component_count_t i = 0; i < detail::component_count(m); ++i)
			m[i] = c * r[i];
		return m;
	}

	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_QUALIFIER typename matType<T, P>::transpose_type transpose(matType<T, P> const & m)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'transpose' only accept floating-point inputs");
		return detail::compute_transpose<matType, T, P>::call(m);
	}

	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_QUALIFIER T determinant(matType<T, P> const & m)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'determinant' only accept floating-point inputs");
		return detail::compute_determinant<matType, T, P>::call(m);
	}

	template <typename T, precision P, template <typename, precision> class matType>
	GLM_FUNC_QUALIFIER matType<T, P> inverse(matType<T, P> const & m)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'inverse' only accept floating-point inputs");
		return detail::compute_inverse(m);
	}

}//namespace glm
