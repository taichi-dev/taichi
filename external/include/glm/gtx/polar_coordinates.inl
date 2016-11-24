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
/// @ref gtx_polar_coordinates
/// @file glm/gtx/polar_coordinates.inl
/// @date 2007-03-06 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> polar
	(
		tvec3<T, P> const & euclidean
	)
	{
		T const Length(length(euclidean));
		tvec3<T, P> const tmp(euclidean / Length);
		T const xz_dist(sqrt(tmp.x * tmp.x + tmp.z * tmp.z));

		return tvec3<T, P>(
			atan(xz_dist, tmp.y),	// latitude
			atan(tmp.x, tmp.z),		// longitude
			xz_dist);				// xz distance
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> euclidean
	(
		tvec2<T, P> const & polar
	)
	{
		T const latitude(polar.x);
		T const longitude(polar.y);

		return tvec3<T, P>(
			cos(latitude) * sin(longitude),
			sin(latitude),
			cos(latitude) * cos(longitude));
	}

}//namespace glm
