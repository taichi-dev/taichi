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
/// @ref gtx_quaternion
/// @file glm/gtx/quaternion.inl
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <limits>
#include "../gtc/constants.hpp"

namespace glm
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> cross
	(
		tvec3<T, P> const & v,
		tquat<T, P> const & q
	)
	{
		return inverse(q) * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> cross
	(
		tquat<T, P> const & q,
		tvec3<T, P> const & v
	)
	{
		return q * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> squad
	(
		tquat<T, P> const & q1,
		tquat<T, P> const & q2,
		tquat<T, P> const & s1,
		tquat<T, P> const & s2,
		T const & h)
	{
		return mix(mix(q1, q2, h), mix(s1, s2, h), static_cast<T>(2) * (static_cast<T>(1) - h) * h);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> intermediate
	(
		tquat<T, P> const & prev,
		tquat<T, P> const & curr,
		tquat<T, P> const & next
	)
	{
		tquat<T, P> invQuat = inverse(curr);
		return exp((log(next + invQuat) + log(prev + invQuat)) / static_cast<T>(-4)) * curr;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> exp
	(
		tquat<T, P> const & q
	)
	{
		tvec3<T, P> u(q.x, q.y, q.z);
		T Angle = glm::length(u);
		if (Angle < epsilon<T>())
			return tquat<T, P>();

		tvec3<T, P> v(u / Angle);
		return tquat<T, P>(cos(Angle), sin(Angle) * v);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> log
	(
		tquat<T, P> const & q
	)
	{
		tvec3<T, P> u(q.x, q.y, q.z);
		T Vec3Len = length(u);

		if (Vec3Len < epsilon<T>())
		{
			if(q.w > static_cast<T>(0))
				return tquat<T, P>(log(q.w), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
			else if(q.w < static_cast<T>(0))
				return tquat<T, P>(log(-q.w), pi<T>(), static_cast<T>(0), static_cast<T>(0));
			else
				return tquat<T, P>(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
		}
		else
		{
			T QuatLen = sqrt(Vec3Len * Vec3Len + q.w * q.w);
			T t = atan(Vec3Len, T(q.w)) / Vec3Len;
			return tquat<T, P>(log(QuatLen), t * q.x, t * q.y, t * q.z);
		}
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> pow
	(
		tquat<T, P> const & x,
		T const & y
	)
	{
		if(abs(x.w) > (static_cast<T>(1) - epsilon<T>()))
			return x;
		T Angle = acos(y);
		T NewAngle = Angle * y;
		T Div = sin(NewAngle) / sin(Angle);
		return tquat<T, P>(
			cos(NewAngle),
			x.x * Div,
			x.y * Div,
			x.z * Div);
	}

	//template <typename T, precision P>
	//GLM_FUNC_QUALIFIER tquat<T, P> sqrt
	//(
	//	tquat<T, P> const & q
	//)
	//{
	//	T q0 = static_cast<T>(1) - dot(q, q);
	//	return T(2) * (T(1) + q0) * q;
	//}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> rotate
	(
		tquat<T, P> const & q,
		tvec3<T, P> const & v
	)
	{
		return q * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> rotate
	(
		tquat<T, P> const & q,
		tvec4<T, P> const & v
	)
	{
		return q * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T extractRealComponent
	(
		tquat<T, P> const & q
	)
	{
		T w = static_cast<T>(1) - q.x * q.x - q.y * q.y - q.z * q.z;
		if(w < T(0))
			return T(0);
		else
			return -sqrt(w);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER T length2
	(
		tquat<T, P> const & q
	)
	{
		return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> shortMix
	(
		tquat<T, P> const & x,
		tquat<T, P> const & y,
		T const & a
	)
	{
		if(a <= static_cast<T>(0)) return x;
		if(a >= static_cast<T>(1)) return y;

		T fCos = dot(x, y);
		tquat<T, P> y2(y); //BUG!!! tquat<T> y2;
		if(fCos < static_cast<T>(0))
		{
			y2 = -y;
			fCos = -fCos;
		}

		//if(fCos > 1.0f) // problem
		T k0, k1;
		if(fCos > (static_cast<T>(1) - epsilon<T>()))
		{
			k0 = static_cast<T>(1) - a;
			k1 = static_cast<T>(0) + a; //BUG!!! 1.0f + a;
		}
		else
		{
			T fSin = sqrt(T(1) - fCos * fCos);
			T fAngle = atan(fSin, fCos);
			T fOneOverSin = static_cast<T>(1) / fSin;
			k0 = sin((static_cast<T>(1) - a) * fAngle) * fOneOverSin;
			k1 = sin((static_cast<T>(0) + a) * fAngle) * fOneOverSin;
		}

		return tquat<T, P>(
			k0 * x.w + k1 * y2.w,
			k0 * x.x + k1 * y2.x,
			k0 * x.y + k1 * y2.y,
			k0 * x.z + k1 * y2.z);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> fastMix
	(
		tquat<T, P> const & x,
		tquat<T, P> const & y,
		T const & a
	)
	{
		return glm::normalize(x * (static_cast<T>(1) - a) + (y * a));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tquat<T, P> rotation
	(
		tvec3<T, P> const & orig,
		tvec3<T, P> const & dest
	)
	{
		T cosTheta = dot(orig, dest);
		tvec3<T, P> rotationAxis;

		if(cosTheta < static_cast<T>(-1) + epsilon<T>())
		{
			// special case when vectors in opposite directions :
			// there is no "ideal" rotation axis
			// So guess one; any will do as long as it's perpendicular to start
			// This implementation favors a rotation around the Up axis (Y),
			// since it's often what you want to do.
			rotationAxis = cross(tvec3<T, P>(0, 0, 1), orig);
			if(length2(rotationAxis) < epsilon<T>()) // bad luck, they were parallel, try again!
				rotationAxis = cross(tvec3<T, P>(1, 0, 0), orig);

			rotationAxis = normalize(rotationAxis);
			return angleAxis(pi<T>(), rotationAxis);
		}

		// Implementation from Stan Melax's Game Programming Gems 1 article
		rotationAxis = cross(orig, dest);

		T s = sqrt((T(1) + cosTheta) * static_cast<T>(2));
		T invs = static_cast<T>(1) / s;

		return tquat<T, P>(
			s * static_cast<T>(0.5f), 
			rotationAxis.x * invs,
			rotationAxis.y * invs,
			rotationAxis.z * invs);
	}

}//namespace glm
