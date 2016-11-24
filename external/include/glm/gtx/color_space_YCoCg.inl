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
/// @ref gtx_color_space_YCoCg
/// @file glm/gtx/color_space_YCoCg.inl
/// @date 2008-10-28 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> rgb2YCoCg
	(
		tvec3<T, P> const & rgbColor
	)
	{
		tvec3<T, P> result;
		result.x/*Y */ =   rgbColor.r / T(4) + rgbColor.g / T(2) + rgbColor.b / T(4);
		result.y/*Co*/ =   rgbColor.r / T(2) + rgbColor.g * T(0) - rgbColor.b / T(2);
		result.z/*Cg*/ = - rgbColor.r / T(4) + rgbColor.g / T(2) - rgbColor.b / T(4);
		return result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> rgb2YCoCgR
	(
		tvec3<T, P> const & rgbColor
	)
	{
		tvec3<T, P> result;
		result.x/*Y */ = rgbColor.g / T(2) + (rgbColor.r + rgbColor.b) / T(4);
		result.y/*Co*/ = rgbColor.r - rgbColor.b;
		result.z/*Cg*/ = rgbColor.g - (rgbColor.r + rgbColor.b) / T(2);
		return result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> YCoCg2rgb
	(
		tvec3<T, P> const & YCoCgColor
	)
	{
		tvec3<T, P> result;
		result.r = YCoCgColor.x + YCoCgColor.y - YCoCgColor.z;
		result.g = YCoCgColor.x                + YCoCgColor.z;
		result.b = YCoCgColor.x - YCoCgColor.y - YCoCgColor.z;
		return result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> YCoCgR2rgb
	(
		tvec3<T, P> const & YCoCgRColor
	)
	{
		tvec3<T, P> result;
		T tmp = YCoCgRColor.x - (YCoCgRColor.z / T(2));
		result.g = YCoCgRColor.z + tmp;
		result.b = tmp - (YCoCgRColor.y / T(2));
		result.r = result.b + YCoCgRColor.y;
		return result;
	}
}//namespace glm
