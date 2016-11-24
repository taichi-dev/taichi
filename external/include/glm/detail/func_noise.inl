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
/// @file glm/detail/func_noise.inl
/// @date 2008-08-01 / 2011-09-27
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include "../detail/_noise.hpp"
#include "./func_common.hpp"

namespace glm{
namespace detail
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> grad4(T const & j, tvec4<T, P> const & ip)
	{
		tvec3<T, P> pXYZ = floor(fract(tvec3<T, P>(j) * tvec3<T, P>(ip)) * T(7)) * ip[2] - T(1);
		T pW = static_cast<T>(1.5) - dot(abs(pXYZ), tvec3<T, P>(1));
		tvec4<T, P> s = tvec4<T, P>(lessThan(tvec4<T, P>(pXYZ, pW), tvec4<T, P>(0.0)));
		pXYZ = pXYZ + (tvec3<T, P>(s) * T(2) - T(1)) * s.w; 
		return tvec4<T, P>(pXYZ, pW);
	}
}//namespace detail

	template <typename T>
	GLM_FUNC_QUALIFIER T noise1(T const & x)
	{
		return noise1(tvec2<T, defaultp>(x, T(0)));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER tvec2<T, defaultp> noise2(T const & x)
	{
		return tvec2<T, defaultp>(
			noise1(x + T(0.0)),
			noise1(x + T(1.0)));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER tvec3<T, defaultp> noise3(T const & x)
	{
		return tvec3<T, defaultp>(
			noise1(x - T(1.0)),
			noise1(x + T(0.0)),
			noise1(x + T(1.0)));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER tvec4<T, defaultp> noise4(T const & x)
	{
		return tvec4<T, defaultp>(
			noise1(x - T(1.0)),
			noise1(x + T(0.0)),
			noise1(x + T(1.0)),
			noise1(x + T(2.0)));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T noise1(tvec2<T, P> const & v)
	{
		tvec4<T, P> const C = tvec4<T, P>(
			T( 0.211324865405187),		// (3.0 -  sqrt(3.0)) / 6.0
			T( 0.366025403784439),		//  0.5 * (sqrt(3.0)  - 1.0)
			T(-0.577350269189626),		// -1.0 + 2.0 * C.x
			T( 0.024390243902439));		//  1.0 / 41.0
		
		// First corner
		tvec2<T, P> i  = floor(v + dot(v, tvec2<T, P>(C[1])));
		tvec2<T, P> x0 = v -   i + dot(i, tvec2<T, P>(C[0]));
		
		// Other corners
		//i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
		//i1.y = 1.0 - i1.x;
		tvec2<T, P> i1 = (x0.x > x0.y) ? tvec2<T, P>(1, 0) : tvec2<T, P>(0, 1);

		// x0 = x0 - 0.0 + 0.0 * C.xx ;
		// x1 = x0 - i1 + 1.0 * C.xx ;
		// x2 = x0 - 1.0 + 2.0 * C.xx ;
		tvec4<T, P> x12 = tvec4<T, P>(x0.x, x0.y, x0.x, x0.y) + tvec4<T, P>(C.x, C.x, C.z, C.z);
		x12 = tvec4<T, P>(tvec2<T, P>(x12) - i1, x12.z, x12.w);
		
		// Permutations
		i = mod(i, T(289)); // Avoid truncation effects in permutation
		tvec3<T, P> p = detail::permute(
			detail::permute(i.y + tvec3<T, P>(T(0), i1.y, T(1))) + i.x + tvec3<T, P>(T(0), i1.x, T(1)));
		
		tvec3<T, P> m = max(T(0.5) - tvec3<T, P>(
			dot(x0, x0),
			dot(tvec2<T, P>(x12.x, x12.y), tvec2<T, P>(x12.x, x12.y)),
			dot(tvec2<T, P>(x12.z, x12.w), tvec2<T, P>(x12.z, x12.w))), T(0));
		
		m = m * m;
		m = m * m;
		
		// Gradients: 41 points uniformly over a line, mapped onto a diamond.
		// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
		
		tvec3<T, P> x = static_cast<T>(2) * fract(p * C.w) - T(1);
		tvec3<T, P> h = abs(x) - T(0.5);
		tvec3<T, P> ox = floor(x + T(0.5));
		tvec3<T, P> a0 = x - ox;
		
		// Normalise gradients implicitly by scaling m
		// Inlined for speed: m *= taylorInvSqrt( a0*a0 + h*h );
		m *= static_cast<T>(1.79284291400159) - T(0.85373472095314) * (a0 * a0 + h * h);
		
		// Compute final noise value at P
		tvec3<T, P> g;
		g.x  = a0.x  * x0.x  + h.x  * x0.y;
		//g.yz = a0.yz * x12.xz + h.yz * x12.yw;
		g.y = a0.y * x12.x + h.y * x12.y;
		g.z = a0.z * x12.z + h.z * x12.w;
		return T(130) * dot(m, g);
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T noise1(tvec3<T, P> const & v)
	{
		tvec2<T, P> const C(1.0 / 6.0, 1.0 / 3.0);
		tvec4<T, P> const D(0.0, 0.5, 1.0, 2.0);
		
		// First corner
		tvec3<T, P> i(floor(v + dot(v, tvec3<T, P>(C.y))));
		tvec3<T, P> x0(v - i + dot(i, tvec3<T, P>(C.x)));
		
		// Other corners
		tvec3<T, P> g(step(tvec3<T, P>(x0.y, x0.z, x0.x), x0));
		tvec3<T, P> l(T(1) - g);
		tvec3<T, P> i1(min(g, tvec3<T, P>(l.z, l.x, l.y)));
		tvec3<T, P> i2(max(g, tvec3<T, P>(l.z, l.x, l.y)));
		
		// x0 = x0 - 0.0 + 0.0 * C.xxx;
		// x1 = x0 - i1  + 1.0 * C.xxx;
		// x2 = x0 - i2  + 2.0 * C.xxx;
		// x3 = x0 - 1.0 + 3.0 * C.xxx;
		tvec3<T, P> x1(x0 - i1 + C.x);
		tvec3<T, P> x2(x0 - i2 + C.y);		// 2.0*C.x = 1/3 = C.y
		tvec3<T, P> x3(x0 - D.y);			// -1.0+3.0*C.x = -0.5 = -D.y
		
		// Permutations
		i = mod289(i); 
		tvec4<T, P> p(detail::permute(detail::permute(detail::permute(
			i.z + tvec4<T, P>(T(0), i1.z, i2.z, T(1))) +
			i.y + tvec4<T, P>(T(0), i1.y, i2.y, T(1))) +
			i.x + tvec4<T, P>(T(0), i1.x, i2.x, T(1))));
		
		// Gradients: 7x7 points over a square, mapped onto an octahedron.
		// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
		T n_ = static_cast<T>(0.142857142857); // 1.0/7.0
		tvec3<T, P> ns(n_ * tvec3<T, P>(D.w, D.y, D.z) - tvec3<T, P>(D.x, D.z, D.x));
		
		tvec4<T, P> j(p - T(49) * floor(p * ns.z * ns.z));	// mod(p,7*7)
		
		tvec4<T, P> x_(floor(j * ns.z));
		tvec4<T, P> y_(floor(j - T(7) * x_));				// mod(j,N)
		
		tvec4<T, P> x(x_ * ns.x + ns.y);
		tvec4<T, P> y(y_ * ns.x + ns.y);
		tvec4<T, P> h(T(1) - abs(x) - abs(y));
		
		tvec4<T, P> b0(x.x, x.y, y.x, y.y);
		tvec4<T, P> b1(x.z, x.w, y.z, y.w);
		
		// vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
		// vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
		tvec4<T, P> s0(floor(b0) * T(2) + T(1));
		tvec4<T, P> s1(floor(b1) * T(2) + T(1));
		tvec4<T, P> sh(-step(h, tvec4<T, P>(0.0)));
		
		tvec4<T, P> a0 = tvec4<T, P>(b0.x, b0.z, b0.y, b0.w) + tvec4<T, P>(s0.x, s0.z, s0.y, s0.w) * tvec4<T, P>(sh.x, sh.x, sh.y, sh.y);
		tvec4<T, P> a1 = tvec4<T, P>(b1.x, b1.z, b1.y, b1.w) + tvec4<T, P>(s1.x, s1.z, s1.y, s1.w) * tvec4<T, P>(sh.z, sh.z, sh.w, sh.w);
		
		tvec3<T, P> p0(a0.x, a0.y, h.x);
		tvec3<T, P> p1(a0.z, a0.w, h.y);
		tvec3<T, P> p2(a1.x, a1.y, h.z);
		tvec3<T, P> p3(a1.z, a1.w, h.w);
		
		// Normalise gradients
		tvec4<T, P> norm = taylorInvSqrt(tvec4<T, P>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
		p0 *= norm.x;
		p1 *= norm.y;
		p2 *= norm.z;
		p3 *= norm.w;
		
		// Mix final noise value
		tvec4<T, P> m = max(T(0.6) - tvec4<T, P>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), T(0));
		m = m * m;
		return T(42) * dot(m * m, tvec4<T, P>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T noise1(tvec4<T, P> const & v)
	{
		tvec4<T, P> const C(
			0.138196601125011,		// (5 - sqrt(5))/20  G4
			0.276393202250021,		// 2 * G4
			0.414589803375032,		// 3 * G4
			-0.447213595499958);	// -1 + 4 * G4
		
		// (sqrt(5) - 1)/4 = F4, used once below
		T const F4 = static_cast<T>(0.309016994374947451);
		
		// First corner
		tvec4<T, P> i  = floor(v + dot(v, tvec4<T, P>(F4)));
		tvec4<T, P> x0 = v -   i + dot(i, tvec4<T, P>(C.x));
		
		// Other corners
		
		// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
		tvec4<T, P> i0;
		tvec3<T, P> isX = step(tvec3<T, P>(x0.y, x0.z, x0.w), tvec3<T, P>(x0.x));
		tvec3<T, P> isYZ = step(tvec3<T, P>(x0.z, x0.w, x0.w), tvec3<T, P>(x0.y, x0.y, x0.z));
		
		//  i0.x = dot(isX, vec3(1.0));
		//i0.x = isX.x + isX.y + isX.z;
		//i0.yzw = static_cast<T>(1) - isX;
		i0 = tvec4<T, P>(isX.x + isX.y + isX.z, T(1) - isX);
		
		//  i0.y += dot(isYZ.xy, vec2(1.0));
		i0.y += isYZ.x + isYZ.y;
		
		//i0.zw += 1.0 - tvec2<T, P>(isYZ.x, isYZ.y);
		i0.z += static_cast<T>(1) - isYZ.x;
		i0.w += static_cast<T>(1) - isYZ.y;
		i0.z += isYZ.z;
		i0.w += static_cast<T>(1) - isYZ.z;
		
		// i0 now contains the unique values 0,1,2,3 in each channel
		tvec4<T, P> i3 = clamp(i0, T(0), T(1));
		tvec4<T, P> i2 = clamp(i0 - T(1), T(0), T(1));
		tvec4<T, P> i1 = clamp(i0 - T(2), T(0), T(1));
		
		//  x0 = x0 - 0.0 + 0.0 * C.xxxx
		//  x1 = x0 - i1  + 0.0 * C.xxxx
		//  x2 = x0 - i2  + 0.0 * C.xxxx
		//  x3 = x0 - i3  + 0.0 * C.xxxx
		//  x4 = x0 - 1.0 + 4.0 * C.xxxx
		tvec4<T, P> x1 = x0 - i1 + C.x;
		tvec4<T, P> x2 = x0 - i2 + C.y;
		tvec4<T, P> x3 = x0 - i3 + C.z;
		tvec4<T, P> x4 = x0 + C.w;
		
		// Permutations
		i = mod(i, T(289));
		T j0 = detail::permute(detail::permute(detail::permute(detail::permute(i.w) + i.z) + i.y) + i.x);
		tvec4<T, P> j1 = detail::permute(detail::permute(detail::permute(detail::permute(
			i.w + tvec4<T, P>(i1.w, i2.w, i3.w, T(1))) +
			i.z + tvec4<T, P>(i1.z, i2.z, i3.z, T(1))) +
			i.y + tvec4<T, P>(i1.y, i2.y, i3.y, T(1))) +
			i.x + tvec4<T, P>(i1.x, i2.x, i3.x, T(1)));
		
		// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
		// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
		tvec4<T, P> ip = tvec4<T, P>(T(1) / T(294), T(1) / T(49), T(1) / T(7), T(0));
		
		tvec4<T, P> p0 = detail::grad4(j0,   ip);
		tvec4<T, P> p1 = detail::grad4(j1.x, ip);
		tvec4<T, P> p2 = detail::grad4(j1.y, ip);
		tvec4<T, P> p3 = detail::grad4(j1.z, ip);
		tvec4<T, P> p4 = detail::grad4(j1.w, ip);
		
		// Normalise gradients
		tvec4<T, P> norm = detail::taylorInvSqrt(tvec4<T, P>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
		p0 *= norm.x;
		p1 *= norm.y;
		p2 *= norm.z;
		p3 *= norm.w;
		p4 *= taylorInvSqrt(dot(p4, p4));
		
		// Mix contributions from the five corners
		tvec3<T, P> m0 = max(T(0.6) - tvec3<T, P>(dot(x0, x0), dot(x1, x1), dot(x2, x2)), T(0));
		tvec2<T, P> m1 = max(T(0.6) - tvec2<T, P>(dot(x3, x3), dot(x4, x4)             ), T(0));
		m0 = m0 * m0;
		m1 = m1 * m1;
		
		return T(49) * (
			dot(m0 * m0, tvec3<T, P>(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
			dot(m1 * m1, tvec2<T, P>(dot(p3, x3), dot(p4, x4))));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> noise2(tvec2<T, P> const & x)
	{
		return tvec2<T, P>(
			noise1(x + tvec2<T, P>(0.0)),
			noise1(tvec2<T, P>(0.0) - x));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> noise2(tvec3<T, P> const & x)
	{
		return tvec2<T, P>(
			noise1(x + tvec3<T, P>(0.0)),
			noise1(tvec3<T, P>(0.0) - x));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec2<T, P> noise2(tvec4<T, P> const & x)
	{
		return tvec2<T, P>(
			noise1(x + tvec4<T, P>(0)),
			noise1(tvec4<T, P>(0) - x));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> noise3(tvec2<T, P> const & x)
	{
		return tvec3<T, P>(
			noise1(x - tvec2<T, P>(1.0)),
			noise1(x + tvec2<T, P>(0.0)),
			noise1(x + tvec2<T, P>(1.0)));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> noise3(tvec3<T, P> const & x)
	{
		return tvec3<T, P>(
			noise1(x - tvec3<T, P>(1.0)),
			noise1(x + tvec3<T, P>(0.0)),
			noise1(x + tvec3<T, P>(1.0)));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> noise3(tvec4<T, P> const & x)
	{
		return tvec3<T, P>(
			noise1(x - tvec4<T, P>(1)),
			noise1(x + tvec4<T, P>(0)),
			noise1(x + tvec4<T, P>(1)));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> noise4(tvec2<T, P> const & x)
	{
		return tvec4<T, P>(
			noise1(x - tvec2<T, P>(1)),
			noise1(x + tvec2<T, P>(0)),
			noise1(x + tvec2<T, P>(1)),
			noise1(x + tvec2<T, P>(2)));
	}

	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> noise4(tvec3<T, P> const & x)
	{
		return tvec4<T, P>(
			noise1(x - tvec3<T, P>(1)),
			noise1(x + tvec3<T, P>(0)),
			noise1(x + tvec3<T, P>(1)),
			noise1(x + tvec3<T, P>(2)));
	}
	
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> noise4(tvec4<T, P> const & x)
	{
		return tvec4<T, P>(
			noise1(x - tvec4<T, P>(1)),
			noise1(x + tvec4<T, P>(0)),
			noise1(x + tvec4<T, P>(1)),
			noise1(x + tvec4<T, P>(2)));
	}
	
}//namespace glm
