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
/// @file glm/gtx/compatibility.hpp
/// @date 2007-01-24 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtc_half_float (dependence)
///
/// @defgroup gtx_compatibility GLM_GTX_compatibility
/// @ingroup gtx
/// 
/// @brief Provide functions to increase the compatibility with Cg and HLSL languages
/// 
/// <glm/gtx/compatibility.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_compatibility extension included")
#endif

#if(GLM_COMPILER & GLM_COMPILER_VC)
#	include <cfloat>
#elif(GLM_COMPILER & GLM_COMPILER_GCC)
#	include <cmath>
#	if(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
#		undef isfinite
#	endif
#endif//GLM_COMPILER

namespace glm
{
	/// @addtogroup gtx_compatibility
	/// @{

	template <typename T> GLM_FUNC_QUALIFIER T lerp(T x, T y, T a){return mix(x, y, a);}																					//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> lerp(const tvec2<T, P>& x, const tvec2<T, P>& y, T a){return mix(x, y, a);}							//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)

	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> lerp(const tvec3<T, P>& x, const tvec3<T, P>& y, T a){return mix(x, y, a);}							//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> lerp(const tvec4<T, P>& x, const tvec4<T, P>& y, T a){return mix(x, y, a);}							//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> lerp(const tvec2<T, P>& x, const tvec2<T, P>& y, const tvec2<T, P>& a){return mix(x, y, a);}	//!< \brief Returns the component-wise result of x * (1.0 - a) + y * a, i.e., the linear blend of x and y using vector a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> lerp(const tvec3<T, P>& x, const tvec3<T, P>& y, const tvec3<T, P>& a){return mix(x, y, a);}	//!< \brief Returns the component-wise result of x * (1.0 - a) + y * a, i.e., the linear blend of x and y using vector a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> lerp(const tvec4<T, P>& x, const tvec4<T, P>& y, const tvec4<T, P>& a){return mix(x, y, a);}	//!< \brief Returns the component-wise result of x * (1.0 - a) + y * a, i.e., the linear blend of x and y using vector a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)

	template <typename T, precision P> GLM_FUNC_QUALIFIER T saturate(T x){return clamp(x, T(0), T(1));}														//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> saturate(const tvec2<T, P>& x){return clamp(x, T(0), T(1));}					//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> saturate(const tvec3<T, P>& x){return clamp(x, T(0), T(1));}					//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> saturate(const tvec4<T, P>& x){return clamp(x, T(0), T(1));}					//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)

	template <typename T, precision P> GLM_FUNC_QUALIFIER T atan2(T x, T y){return atan(x, y);}																//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> atan2(const tvec2<T, P>& x, const tvec2<T, P>& y){return atan(x, y);}	//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> atan2(const tvec3<T, P>& x, const tvec3<T, P>& y){return atan(x, y);}	//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> atan2(const tvec4<T, P>& x, const tvec4<T, P>& y){return atan(x, y);}	//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)

	template <typename genType> GLM_FUNC_DECL bool isfinite(genType const & x);											//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_DECL tvec2<bool, P> isfinite(const tvec2<T, P>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_DECL tvec3<bool, P> isfinite(const tvec3<T, P>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template <typename T, precision P> GLM_FUNC_DECL tvec4<bool, P> isfinite(const tvec4<T, P>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)

	typedef bool						bool1;			//!< \brief boolean type with 1 component. (From GLM_GTX_compatibility extension)
	typedef tvec2<bool, highp>			bool2;			//!< \brief boolean type with 2 components. (From GLM_GTX_compatibility extension)
	typedef tvec3<bool, highp>			bool3;			//!< \brief boolean type with 3 components. (From GLM_GTX_compatibility extension)
	typedef tvec4<bool, highp>			bool4;			//!< \brief boolean type with 4 components. (From GLM_GTX_compatibility extension)

	typedef bool						bool1x1;		//!< \brief boolean matrix with 1 x 1 component. (From GLM_GTX_compatibility extension)
	typedef tmat2x2<bool, highp>		bool2x2;		//!< \brief boolean matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x3<bool, highp>		bool2x3;		//!< \brief boolean matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x4<bool, highp>		bool2x4;		//!< \brief boolean matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x2<bool, highp>		bool3x2;		//!< \brief boolean matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x3<bool, highp>		bool3x3;		//!< \brief boolean matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x4<bool, highp>		bool3x4;		//!< \brief boolean matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x2<bool, highp>		bool4x2;		//!< \brief boolean matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x3<bool, highp>		bool4x3;		//!< \brief boolean matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x4<bool, highp>		bool4x4;		//!< \brief boolean matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	typedef int							int1;			//!< \brief integer vector with 1 component. (From GLM_GTX_compatibility extension)
	typedef tvec2<int, highp>			int2;			//!< \brief integer vector with 2 components. (From GLM_GTX_compatibility extension)
	typedef tvec3<int, highp>			int3;			//!< \brief integer vector with 3 components. (From GLM_GTX_compatibility extension)
	typedef tvec4<int, highp>			int4;			//!< \brief integer vector with 4 components. (From GLM_GTX_compatibility extension)

	typedef int							int1x1;			//!< \brief integer matrix with 1 component. (From GLM_GTX_compatibility extension)
	typedef tmat2x2<int, highp>		int2x2;			//!< \brief integer matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x3<int, highp>		int2x3;			//!< \brief integer matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x4<int, highp>		int2x4;			//!< \brief integer matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x2<int, highp>		int3x2;			//!< \brief integer matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x3<int, highp>		int3x3;			//!< \brief integer matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x4<int, highp>		int3x4;			//!< \brief integer matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x2<int, highp>		int4x2;			//!< \brief integer matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x3<int, highp>		int4x3;			//!< \brief integer matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x4<int, highp>		int4x4;			//!< \brief integer matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	typedef float						float1;			//!< \brief single-precision floating-point vector with 1 component. (From GLM_GTX_compatibility extension)
	typedef tvec2<float, highp>		float2;			//!< \brief single-precision floating-point vector with 2 components. (From GLM_GTX_compatibility extension)
	typedef tvec3<float, highp>		float3;			//!< \brief single-precision floating-point vector with 3 components. (From GLM_GTX_compatibility extension)
	typedef tvec4<float, highp>		float4;			//!< \brief single-precision floating-point vector with 4 components. (From GLM_GTX_compatibility extension)

	typedef float						float1x1;		//!< \brief single-precision floating-point matrix with 1 component. (From GLM_GTX_compatibility extension)
	typedef tmat2x2<float, highp>		float2x2;		//!< \brief single-precision floating-point matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x3<float, highp>		float2x3;		//!< \brief single-precision floating-point matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x4<float, highp>		float2x4;		//!< \brief single-precision floating-point matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x2<float, highp>		float3x2;		//!< \brief single-precision floating-point matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x3<float, highp>		float3x3;		//!< \brief single-precision floating-point matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x4<float, highp>		float3x4;		//!< \brief single-precision floating-point matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x2<float, highp>		float4x2;		//!< \brief single-precision floating-point matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x3<float, highp>		float4x3;		//!< \brief single-precision floating-point matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x4<float, highp>		float4x4;		//!< \brief single-precision floating-point matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	typedef double						double1;		//!< \brief double-precision floating-point vector with 1 component. (From GLM_GTX_compatibility extension)
	typedef tvec2<double, highp>		double2;		//!< \brief double-precision floating-point vector with 2 components. (From GLM_GTX_compatibility extension)
	typedef tvec3<double, highp>		double3;		//!< \brief double-precision floating-point vector with 3 components. (From GLM_GTX_compatibility extension)
	typedef tvec4<double, highp>		double4;		//!< \brief double-precision floating-point vector with 4 components. (From GLM_GTX_compatibility extension)

	typedef double						double1x1;		//!< \brief double-precision floating-point matrix with 1 component. (From GLM_GTX_compatibility extension)
	typedef tmat2x2<double, highp>		double2x2;		//!< \brief double-precision floating-point matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x3<double, highp>		double2x3;		//!< \brief double-precision floating-point matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat2x4<double, highp>		double2x4;		//!< \brief double-precision floating-point matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x2<double, highp>		double3x2;		//!< \brief double-precision floating-point matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x3<double, highp>		double3x3;		//!< \brief double-precision floating-point matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat3x4<double, highp>		double3x4;		//!< \brief double-precision floating-point matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x2<double, highp>		double4x2;		//!< \brief double-precision floating-point matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x3<double, highp>		double4x3;		//!< \brief double-precision floating-point matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef tmat4x4<double, highp>		double4x4;		//!< \brief double-precision floating-point matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	/// @}
}//namespace glm

#include "compatibility.inl"
