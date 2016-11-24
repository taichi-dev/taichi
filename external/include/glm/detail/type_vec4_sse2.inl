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
/// @file glm/detail/type_tvec4_sse2.inl
/// @date 2014-12-01 / 2014-12-01
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail
{

}//namespace detail

	template <>
	GLM_FUNC_QUALIFIER tvec4<float, lowp>::tvec4()
#		ifndef GLM_FORCE_NO_CTOR_INIT
			: data(_mm_setzero_ps())
#		endif
	{}
	
	template <>
	GLM_FUNC_QUALIFIER tvec4<float, mediump>::tvec4()
#		ifndef GLM_FORCE_NO_CTOR_INIT
			: data(_mm_setzero_ps())
#		endif
	{}

	template <>
	GLM_FUNC_QUALIFIER tvec4<float, lowp>::tvec4(float s) :
		data(_mm_set1_ps(s))
	{}
	
	template <>
	GLM_FUNC_QUALIFIER tvec4<float, mediump>::tvec4(float s) :
		data(_mm_set1_ps(s))
	{}

	template <>
	GLM_FUNC_QUALIFIER tvec4<float, lowp>::tvec4(float a, float b, float c, float d) :
		data(_mm_set_ps(d, c, b, a))
	{}
	
	template <>
	GLM_FUNC_QUALIFIER tvec4<float, mediump>::tvec4(float a, float b, float c, float d) :
		data(_mm_set_ps(d, c, b, a))
	{}

	template <>
	template <typename U>
	GLM_FUNC_QUALIFIER tvec4<float, lowp> & tvec4<float, lowp>::operator+=(U scalar)
	{
		this->data = _mm_add_ps(this->data, _mm_set_ps1(static_cast<float>(scalar)));
		return *this;
	}

	template <>
	template <>
	GLM_FUNC_QUALIFIER tvec4<float, lowp> & tvec4<float, lowp>::operator+=<float>(float scalar)
	{
		this->data = _mm_add_ps(this->data, _mm_set_ps1(scalar));
		return *this;
	}

	template <>
	template <typename U>
	GLM_FUNC_QUALIFIER tvec4<float, mediump> & tvec4<float, mediump>::operator+=(U scalar)
	{
		this->data = _mm_add_ps(this->data, _mm_set_ps1(static_cast<float>(scalar)));
		return *this;
	}

	template <>
	template <>
	GLM_FUNC_QUALIFIER tvec4<float, mediump> & tvec4<float, mediump>::operator+=<float>(float scalar)
	{
		this->data = _mm_add_ps(this->data, _mm_set_ps1(scalar));
		return *this;
	}

	template <>
	template <typename U>
	GLM_FUNC_QUALIFIER tvec4<float, lowp> & tvec4<float, lowp>::operator+=(tvec1<U, lowp> const & v)
	{
		this->data = _mm_add_ps(this->data, _mm_set_ps1(static_cast<float>(v.x)));
		return *this;
	}

	template <>
	template <typename U>
	GLM_FUNC_QUALIFIER tvec4<float, mediump> & tvec4<float, mediump>::operator+=(tvec1<U, mediump> const & v)
	{
		this->data = _mm_add_ps(this->data, _mm_set_ps1(static_cast<float>(v.x)));
		return *this;
	}
}//namespace glm
