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
/// @ref gtx_common
/// @file glm/gtx/common.inl
/// @date 2014-09-08 / 2014-09-08
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <cmath>

namespace glm
{
	template <typename T> 
	GLM_FUNC_QUALIFIER bool isdenormal(T const & x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isdenormal' only accept floating-point inputs");

#		if GLM_HAS_CXX11_STL
			return std::fpclassify(x) == FP_SUBNORMAL;
#		else
			return x != static_cast<T>(0) && std::fabs(x) < std::numeric_limits<T>::min();
#		endif
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tvec1<T, P>::bool_type isdenormal
	(
		tvec1<T, P> const & x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isdenormal' only accept floating-point inputs");

		return typename tvec1<T, P>::bool_type(
			isdenormal(x.x));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tvec2<T, P>::bool_type isdenormal
	(
		tvec2<T, P> const & x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isdenormal' only accept floating-point inputs");

		return typename tvec2<T, P>::bool_type(
			isdenormal(x.x),
			isdenormal(x.y));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tvec3<T, P>::bool_type isdenormal
	(
		tvec3<T, P> const & x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isdenormal' only accept floating-point inputs");

		return typename tvec3<T, P>::bool_type(
			isdenormal(x.x),
			isdenormal(x.y),
			isdenormal(x.z));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tvec4<T, P>::bool_type isdenormal
	(
		tvec4<T, P> const & x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'isdenormal' only accept floating-point inputs");

		return typename tvec4<T, P>::bool_type(
			isdenormal(x.x),
			isdenormal(x.y),
			isdenormal(x.z),
			isdenormal(x.w));
	}
}//namespace glm
