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
/// @ref gtx_gradient_paint
/// @file glm/gtx/gradient_paint.inl
/// @date 2009-03-06 / 2013-04-09
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T radialGradient
	(
		tvec2<T, P> const & Center,
		T const & Radius,
		tvec2<T, P> const & Focal,
		tvec2<T, P> const & Position
	)
	{
		tvec2<T, P> F = Focal - Center;
		tvec2<T, P> D = Position - Focal;
		T Radius2 = pow2(Radius);
		T Fx2 = pow2(F.x);
		T Fy2 = pow2(F.y);

		T Numerator = (D.x * F.x + D.y * F.y) + sqrt(Radius2 * (pow2(D.x) + pow2(D.y)) - pow2(D.x * F.y - D.y * F.x));
		T Denominator = Radius2 - (Fx2 + Fy2);
		return Numerator / Denominator;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T linearGradient
	(
		tvec2<T, P> const & Point0,
		tvec2<T, P> const & Point1,
		tvec2<T, P> const & Position
	)
	{
		tvec2<T, P> Dist = Point1 - Point0;
		return (Dist.x * (Position.x - Point0.x) + Dist.y * (Position.y - Point0.y)) / glm::dot(Dist, Dist);
	}
}//namespace glm
