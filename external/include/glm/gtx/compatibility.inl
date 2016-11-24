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
/// @ref gtx_compatibility
/// @file glm/gtx/compatibility.inl
/// @date 2007-01-24 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <limits>

namespace glm
{
	// isfinite
	template <typename genType>
	GLM_FUNC_QUALIFIER bool isfinite(
		genType const & x)
	{
#		if GLM_HAS_CXX11_STL
			return std::isfinite(x) != 0;
#		elif GLM_COMPILER & GLM_COMPILER_VC
			return _finite(x);
#		elif GLM_COMPILER & GLM_COMPILER_GCC && GLM_PLATFORM & GLM_PLATFORM_ANDROID
			return _isfinite(x) != 0;
#		else
			return x >= std::numeric_limits<genType>::min() && x <= std::numeric_limits<genType>::max();
#		endif
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<bool, P> isfinite(
		tvec2<T, P> const & x)
	{
		return tvec2<bool, P>(
			isfinite(x.x),
			isfinite(x.y));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<bool, P> isfinite(
		tvec3<T, P> const & x)
	{
		return tvec3<bool, P>(
			isfinite(x.x),
			isfinite(x.y),
			isfinite(x.z));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<bool, P> isfinite(
		tvec4<T, P> const & x)
	{
		return tvec4<bool, P>(
			isfinite(x.x),
			isfinite(x.y),
			isfinite(x.z),
			isfinite(x.w));
	}

}//namespace glm
