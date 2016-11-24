/////////////////////////////////////////////////////////////////////////////////////////
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
/// @ref gtx_color_space
/// @file glm/gtx/color_space.inl
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> rgbColor(const tvec3<T, P>& hsvColor)
	{
		tvec3<T, P> hsv = hsvColor;
		tvec3<T, P> rgbColor;

		if(hsv.y == static_cast<T>(0))
			// achromatic (grey)
			rgbColor = tvec3<T, P>(hsv.z);
		else
		{
			T sector = floor(hsv.x / T(60));
			T frac = (hsv.x / T(60)) - sector;
			// factorial part of h
			T o = hsv.z * (T(1) - hsv.y);
			T p = hsv.z * (T(1) - hsv.y * frac);
			T q = hsv.z * (T(1) - hsv.y * (T(1) - frac));

			switch(int(sector))
			{
			default:
			case 0:
				rgbColor.r = hsv.z;
				rgbColor.g = q;
				rgbColor.b = o;
				break;
			case 1:
				rgbColor.r = p;
				rgbColor.g = hsv.z;
				rgbColor.b = o;
				break;
			case 2:
				rgbColor.r = o;
				rgbColor.g = hsv.z;
				rgbColor.b = q;
				break;
			case 3:
				rgbColor.r = o;
				rgbColor.g = p;
				rgbColor.b = hsv.z;
				break;
			case 4:
				rgbColor.r = q; 
				rgbColor.g = o; 
				rgbColor.b = hsv.z;
				break;
			case 5:
				rgbColor.r = hsv.z; 
				rgbColor.g = o; 
				rgbColor.b = p;
				break;
			}
		}

		return rgbColor;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> hsvColor(const tvec3<T, P>& rgbColor)
	{
		tvec3<T, P> hsv = rgbColor;
		float Min   = min(min(rgbColor.r, rgbColor.g), rgbColor.b);
		float Max   = max(max(rgbColor.r, rgbColor.g), rgbColor.b);
		float Delta = Max - Min;

		hsv.z = Max;                               

		if(Max != static_cast<T>(0))
		{
			hsv.y = Delta / hsv.z;    
			T h = static_cast<T>(0);

			if(rgbColor.r == Max)
				// between yellow & magenta
				h = static_cast<T>(0) + T(60) * (rgbColor.g - rgbColor.b) / Delta;
			else if(rgbColor.g == Max)
				// between cyan & yellow
				h = static_cast<T>(120) + T(60) * (rgbColor.b - rgbColor.r) / Delta;
			else
				// between magenta & cyan
				h = static_cast<T>(240) + T(60) * (rgbColor.r - rgbColor.g) / Delta;

			if(h < T(0)) 
				hsv.x = h + T(360);
			else
				hsv.x = h;
		}
		else
		{
			// If r = g = b = 0 then s = 0, h is undefined
			hsv.y = static_cast<T>(0);
			hsv.x = static_cast<T>(0);
		}

		return hsv;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> saturation(T const s)
	{
		tvec3<T, defaultp> rgbw = tvec3<T, defaultp>(T(0.2126), T(0.7152), T(0.0722));

		T col0 = (T(1) - s) * rgbw.r;
		T col1 = (T(1) - s) * rgbw.g;
		T col2 = (T(1) - s) * rgbw.b;

		tmat4x4<T, defaultp> result(T(1));
		result[0][0] = col0 + s;
		result[0][1] = col0;
		result[0][2] = col0;
		result[1][0] = col1;
		result[1][1] = col1 + s;
		result[1][2] = col1;
		result[2][0] = col2;
		result[2][1] = col2;
		result[2][2] = col2 + s;
		return result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> saturation(const T s, const tvec3<T, P>& color)
	{
		return tvec3<T, P>(saturation(s) * tvec4<T, P>(color, T(0)));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> saturation(const T s, const tvec4<T, P>& color)
	{
		return saturation(s) * color;
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER T luminosity(const tvec3<T, P>& color)
	{
		const tvec3<T, P> tmp = tvec3<T, P>(0.33, 0.59, 0.11);
		return dot(color, tmp);
	}
}//namespace glm
