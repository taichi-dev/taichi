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
/// @ref gtx_simd_quat
/// @file glm/gtx/simd_quat.hpp
/// @date 2013-04-22 / 2014-11-25
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtx_simd_quat GLM_GTX_simd_quat
/// @ingroup gtx
/// 
/// @brief SIMD implementation of quat type.
/// 
/// <glm/gtx/simd_quat.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtx/fast_trigonometry.hpp"

#if(GLM_ARCH != GLM_ARCH_PURE)

#if(GLM_ARCH & GLM_ARCH_SSE2)
#   include "../gtx/simd_mat4.hpp"
#else
#	error "GLM: GLM_GTX_simd_quat requires compiler support of SSE2 through intrinsics"
#endif

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_simd_quat extension included")
#endif

// Warning silencer for nameless struct/union.
#if (GLM_COMPILER & GLM_COMPILER_VC)
#   pragma warning(push)
#   pragma warning(disable:4201)   // warning C4201: nonstandard extension used : nameless struct/union
#endif

namespace glm{
namespace detail
{
	GLM_ALIGNED_STRUCT(16) fquatSIMD
	{
		typedef __m128 value_type;
		typedef std::size_t size_type;
		static size_type value_size();

		typedef fquatSIMD type;
		typedef tquat<bool, defaultp> bool_type;

#ifdef GLM_SIMD_ENABLE_XYZW_UNION
		union
		{
			__m128 Data;
			struct {float x, y, z, w;};
		};
#else
		__m128 Data;
#endif

		//////////////////////////////////////
		// Implicit basic constructors

		fquatSIMD();
		fquatSIMD(__m128 const & Data);
		fquatSIMD(fquatSIMD const & q);

		//////////////////////////////////////
		// Explicit basic constructors

		explicit fquatSIMD(
			ctor);
		explicit fquatSIMD(
			float const & w, 
			float const & x, 
			float const & y, 
			float const & z);
		explicit fquatSIMD(
			quat const & v);
		explicit fquatSIMD(
			vec3 const & eulerAngles);
		

		//////////////////////////////////////
		// Unary arithmetic operators

		fquatSIMD& operator =(fquatSIMD const & q);
		fquatSIMD& operator*=(float const & s);
		fquatSIMD& operator/=(float const & s);
	};


	//////////////////////////////////////
	// Arithmetic operators

	detail::fquatSIMD operator- (
		detail::fquatSIMD const & q);

	detail::fquatSIMD operator+ ( 
		detail::fquatSIMD const & q, 
		detail::fquatSIMD const & p); 

	detail::fquatSIMD operator* ( 
		detail::fquatSIMD const & q, 
		detail::fquatSIMD const & p); 

	detail::fvec4SIMD operator* (
		detail::fquatSIMD const & q, 
		detail::fvec4SIMD const & v);

	detail::fvec4SIMD operator* (
		detail::fvec4SIMD const & v,
		detail::fquatSIMD const & q);

	detail::fquatSIMD operator* (
		detail::fquatSIMD const & q, 
		float s);

	detail::fquatSIMD operator* (
		float s,
		detail::fquatSIMD const & q);

	detail::fquatSIMD operator/ (
		detail::fquatSIMD const & q, 
		float s);

}//namespace detail

	/// @addtogroup gtx_simd_quat
	/// @{

	typedef glm::detail::fquatSIMD simdQuat;

	//! Convert a simdQuat to a quat.
	/// @see gtx_simd_quat
	quat quat_cast(
		detail::fquatSIMD const & x);

	//! Convert a simdMat4 to a simdQuat.
	/// @see gtx_simd_quat
	detail::fquatSIMD quatSIMD_cast(
		detail::fmat4x4SIMD const & m);

	//! Converts a mat4 to a simdQuat.
	/// @see gtx_simd_quat
	template <typename T, precision P>
	detail::fquatSIMD quatSIMD_cast(
		tmat4x4<T, P> const & m);

	//! Converts a mat3 to a simdQuat.
	/// @see gtx_simd_quat
	template <typename T, precision P>
	detail::fquatSIMD quatSIMD_cast(
		tmat3x3<T, P> const & m);

	//! Convert a simdQuat to a simdMat4
	/// @see gtx_simd_quat
	detail::fmat4x4SIMD mat4SIMD_cast(
		detail::fquatSIMD const & q);

	//! Converts a simdQuat to a standard mat4.
	/// @see gtx_simd_quat
	mat4 mat4_cast(
		detail::fquatSIMD const & q);


	/// Returns the length of the quaternion. 
	/// 
	/// @see gtx_simd_quat
	float length(
		detail::fquatSIMD const & x);

	/// Returns the normalized quaternion. 
	/// 
	/// @see gtx_simd_quat
	detail::fquatSIMD normalize(
		detail::fquatSIMD const & x);

	/// Returns dot product of q1 and q2, i.e., q1[0] * q2[0] + q1[1] * q2[1] + ... 
	/// 
	/// @see gtx_simd_quat
	float dot(
		detail::fquatSIMD const & q1, 
		detail::fquatSIMD const & q2);

	/// Spherical linear interpolation of two quaternions.
	/// The interpolation is oriented and the rotation is performed at constant speed.
	/// For short path spherical linear interpolation, use the slerp function.
	/// 
	/// @param x A quaternion
	/// @param y A quaternion
	/// @param a Interpolation factor. The interpolation is defined beyond the range [0, 1].
	/// @tparam T Value type used to build the quaternion. Supported: half, float or double.
	/// @see gtx_simd_quat
	/// @see - slerp(detail::fquatSIMD const & x, detail::fquatSIMD const & y, T const & a) 
	detail::fquatSIMD mix(
		detail::fquatSIMD const & x, 
		detail::fquatSIMD const & y, 
		float const & a);

	/// Linear interpolation of two quaternions. 
	/// The interpolation is oriented.
	/// 
	/// @param x A quaternion
	/// @param y A quaternion
	/// @param a Interpolation factor. The interpolation is defined in the range [0, 1].
	/// @tparam T Value type used to build the quaternion. Supported: half, float or double.
	/// @see gtx_simd_quat
	detail::fquatSIMD lerp(
		detail::fquatSIMD const & x, 
		detail::fquatSIMD const & y, 
		float const & a);

	/// Spherical linear interpolation of two quaternions.
	/// The interpolation always take the short path and the rotation is performed at constant speed.
	/// 
	/// @param x A quaternion
	/// @param y A quaternion
	/// @param a Interpolation factor. The interpolation is defined beyond the range [0, 1].
	/// @tparam T Value type used to build the quaternion. Supported: half, float or double.
	/// @see gtx_simd_quat
	detail::fquatSIMD slerp(
		detail::fquatSIMD const & x, 
		detail::fquatSIMD const & y, 
		float const & a);


	/// Faster spherical linear interpolation of two unit length quaternions.
	///
	/// This is the same as mix(), except for two rules:
	///   1) The two quaternions must be unit length.
	///   2) The interpolation factor (a) must be in the range [0, 1].
	///
	/// This will use the equivalent to fastAcos() and fastSin().
	///
	/// @see gtx_simd_quat
	/// @see - mix(detail::fquatSIMD const & x, detail::fquatSIMD const & y, T const & a) 
	detail::fquatSIMD fastMix(
		detail::fquatSIMD const & x, 
		detail::fquatSIMD const & y, 
		float const & a);

	/// Identical to fastMix() except takes the shortest path.
	///
	/// The same rules apply here as those in fastMix(). Both quaternions must be unit length and 'a' must be
	/// in the range [0, 1].
	///
	/// @see - fastMix(detail::fquatSIMD const & x, detail::fquatSIMD const & y, T const & a) 
	/// @see - slerp(detail::fquatSIMD const & x, detail::fquatSIMD const & y, T const & a) 
	detail::fquatSIMD fastSlerp(
		detail::fquatSIMD const & x, 
		detail::fquatSIMD const & y, 
		float const & a);


	/// Returns the q conjugate. 
	/// 
	/// @see gtx_simd_quat
	detail::fquatSIMD conjugate(
		detail::fquatSIMD const & q);

	/// Returns the q inverse. 
	/// 
	/// @see gtx_simd_quat
	detail::fquatSIMD inverse(
		detail::fquatSIMD const & q);

	/// Build a quaternion from an angle and a normalized axis.
	///
	/// @param angle Angle expressed in radians.
	/// @param axis Axis of the quaternion, must be normalized. 
	///
	/// @see gtx_simd_quat
	detail::fquatSIMD angleAxisSIMD(
		float const & angle, 
		vec3 const & axis);

	/// Build a quaternion from an angle and a normalized axis. 
	///
	/// @param angle Angle expressed in radians.
	/// @param x x component of the x-axis, x, y, z must be a normalized axis
	/// @param y y component of the y-axis, x, y, z must be a normalized axis
	/// @param z z component of the z-axis, x, y, z must be a normalized axis
	///
	/// @see gtx_simd_quat
	detail::fquatSIMD angleAxisSIMD(
		float const & angle, 
		float const & x, 
		float const & y, 
		float const & z);

	// TODO: Move this to somewhere more appropriate. Used with fastMix() and fastSlerp().
	/// Performs the equivalent of glm::fastSin() on each component of the given __m128.
	__m128 fastSin(__m128 x);

	/// @}
}//namespace glm

#include "simd_quat.inl"


#if (GLM_COMPILER & GLM_COMPILER_VC)
#   pragma warning(pop)
#endif


#endif//(GLM_ARCH != GLM_ARCH_PURE)
