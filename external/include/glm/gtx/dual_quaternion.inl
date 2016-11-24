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
/// @ref gtx_dual_quaternion
/// @file glm/gtx/dual_quaternion.inl
/// @date 2013-02-10 / 2013-02-13
/// @author Maksim Vorobiev (msomeone@gmail.com)
///////////////////////////////////////////////////////////////////////////////////

#include "../geometric.hpp"
#include <limits>

namespace glm
{
	//////////////////////////////////////
	// Component accesses

#	ifdef GLM_FORCE_SIZE_FUNC
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tdualquat<T, P>::size_type tdualquat<T, P>::size() const
		{
			return 2;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tdualquat<T, P>::part_type & tdualquat<T, P>::operator[](typename tdualquat<T, P>::size_type i)
		{
			assert(i >= 0 && static_cast<detail::component_count_t>(i) < detail::component_count(*this));
			return (&real)[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tdualquat<T, P>::part_type const & tdualquat<T, P>::operator[](typename tdualquat<T, P>::size_type i) const
		{
			assert(i >= 0 && static_cast<detail::component_count_t>(i) < detail::component_count(*this));
			return (&real)[i];
		}
#	else
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tdualquat<T, P>::length_type tdualquat<T, P>::length() const
		{
			return 2;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tdualquat<T, P>::part_type & tdualquat<T, P>::operator[](typename tdualquat<T, P>::length_type i)
		{
			assert(i >= 0 && static_cast<detail::component_count_t>(i) < detail::component_count(*this));
			return (&real)[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tdualquat<T, P>::part_type const & tdualquat<T, P>::operator[](typename tdualquat<T, P>::length_type i) const
		{
			assert(i >= 0 && static_cast<detail::component_count_t>(i) < detail::component_count(*this));
			return (&real)[i];
		}
#	endif//GLM_FORCE_SIZE_FUNC

	//////////////////////////////////////
	// Implicit basic constructors

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat()
#		ifndef GLM_FORCE_NO_CTOR_INIT 
			: real(tquat<T, P>())
			, dual(tquat<T, P>(0, 0, 0, 0))
#		endif
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tdualquat<T, P> const & d)
		: real(d.real)
		, dual(d.dual)
	{}

	template <typename T, precision P>
	template <precision Q>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tdualquat<T, Q> const & d)
		: real(d.real)
		, dual(d.dual)
	{}

	//////////////////////////////////////
	// Explicit basic constructors

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(ctor)
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tquat<T, P> const & r)
		: real(r), dual(tquat<T, P>(0, 0, 0, 0))
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tquat<T, P> const & q, tvec3<T, P> const& p)
		: real(q), dual(
			T(-0.5) * ( p.x*q.x + p.y*q.y + p.z*q.z),
			T(+0.5) * ( p.x*q.w + p.y*q.z - p.z*q.y),
			T(+0.5) * (-p.x*q.z + p.y*q.w + p.z*q.x),
			T(+0.5) * ( p.x*q.y - p.y*q.x + p.z*q.w))
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tquat<T, P> const & r, tquat<T, P> const & d)
		: real(r), dual(d)
	{}

	//////////////////////////////////////////////////////////////
	// tdualquat conversions

	template <typename T, precision P>
	template <typename U, precision Q>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tdualquat<U, Q> const & q)
		: real(q.real)
		, dual(q.dual)
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tmat2x4<T, P> const & m)
	{
		*this = dualquat_cast(m);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P>::tdualquat(tmat3x4<T, P> const & m)
	{
		*this = dualquat_cast(m);
	}

	//////////////////////////////////////////////////////////////
	// tdualquat operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> & tdualquat<T, P>::operator=(tdualquat<T, P> const & q)
	{
		this->real = q.real;
		this->dual = q.dual;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tdualquat<T, P> & tdualquat<T, P>::operator=(tdualquat<U, P> const & q)
	{
		this->real = q.real;
		this->dual = q.dual;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tdualquat<T, P> & tdualquat<T, P>::operator*=(U s)
	{
		this->real *= static_cast<T>(s);
		this->dual *= static_cast<T>(s);
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tdualquat<T, P> & tdualquat<T, P>::operator/=(U s)
	{
		this->real /= static_cast<T>(s);
		this->dual /= static_cast<T>(s);
		return *this;
	}

	//////////////////////////////////////////////////////////////
	// tquat<valType> external operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> operator-(tdualquat<T, P> const & q)
	{
		return tdualquat<T, P>(-q.real,-q.dual);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> operator+(tdualquat<T, P> const & q, tdualquat<T, P> const & p)
	{
		return tdualquat<T, P>(q.real + p.real,q.dual + p.dual);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> operator*(tdualquat<T, P> const & p, tdualquat<T, P> const & o)
	{
		return tdualquat<T, P>(p.real * o.real,p.real * o.dual + p.dual * o.real);
	}

	// Transformation
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> operator*(tdualquat<T, P> const & q, tvec3<T, P> const & v)
	{
		tvec3<T, P> const real_v3(q.real.x,q.real.y,q.real.z);
		tvec3<T, P> const dual_v3(q.dual.x,q.dual.y,q.dual.z);
		return (cross(real_v3, cross(real_v3,v) + v * q.real.w + dual_v3) + dual_v3 * q.real.w - real_v3 * q.dual.w) * T(2) + v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec3<T, P> operator*(tvec3<T, P> const & v,	tdualquat<T, P> const & q)
	{
		return glm::inverse(q) * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> operator*(tdualquat<T, P> const & q, tvec4<T, P> const & v)
	{
		return tvec4<T, P>(q * tvec3<T, P>(v), v.w);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tvec4<T, P> operator*(tvec4<T, P> const & v,	tdualquat<T, P> const & q)
	{
		return glm::inverse(q) * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> operator*(tdualquat<T, P> const & q, T const & s)
	{
		return tdualquat<T, P>(q.real * s, q.dual * s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> operator*(T const & s, tdualquat<T, P> const & q)
	{
		return q * s;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> operator/(tdualquat<T, P> const & q,	T const & s)
	{
		return tdualquat<T, P>(q.real / s, q.dual / s);
	}

	//////////////////////////////////////
	// Boolean operators
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator==(tdualquat<T, P> const & q1, tdualquat<T, P> const & q2)
	{
		return (q1.real == q2.real) && (q1.dual == q2.dual);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator!=(tdualquat<T, P> const & q1, tdualquat<T, P> const & q2)
	{
		return (q1.real != q2.dual) || (q1.real != q2.dual);
	}

	////////////////////////////////////////////////////////
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> normalize(tdualquat<T, P> const & q)
	{
		return q / length(q.real);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> lerp(tdualquat<T, P> const & x, tdualquat<T, P> const & y, T const & a)
	{
		// Dual Quaternion Linear blend aka DLB:
		// Lerp is only defined in [0, 1]
		assert(a >= static_cast<T>(0));
		assert(a <= static_cast<T>(1));
		T const k = dot(x.real,y.real) < static_cast<T>(0) ? -a : a;
		T const one(1);
		return tdualquat<T, P>(x * (one - a) + y * k);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> inverse(tdualquat<T, P> const & q)
	{
		const glm::tquat<T, P> real = conjugate(q.real);
		const glm::tquat<T, P> dual = conjugate(q.dual);
		return tdualquat<T, P>(real, dual + (real * (-2.0f * dot(real,dual))));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> mat2x4_cast(tdualquat<T, P> const & x)
	{
		return tmat2x4<T, P>( x[0].x, x[0].y, x[0].z, x[0].w, x[1].x, x[1].y, x[1].z, x[1].w );
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x4<T, P> mat3x4_cast(tdualquat<T, P> const & x)
	{
		tquat<T, P> r = x.real / length2(x.real);
		
		tquat<T, P> const rr(r.w * x.real.w, r.x * x.real.x, r.y * x.real.y, r.z * x.real.z);
		r *= static_cast<T>(2);
		
		T const xy = r.x * x.real.y;
		T const xz = r.x * x.real.z;
		T const yz = r.y * x.real.z;
		T const wx = r.w * x.real.x;
		T const wy = r.w * x.real.y;
		T const wz = r.w * x.real.z;
		
		tvec4<T, P> const a(
			rr.w + rr.x - rr.y - rr.z,
			xy - wz,
			xz + wy,
			-(x.dual.w * r.x - x.dual.x * r.w + x.dual.y * r.z - x.dual.z * r.y));
		
		tvec4<T, P> const b(
			xy + wz,
			rr.w + rr.y - rr.x - rr.z,
			yz - wx,
			-(x.dual.w * r.y - x.dual.x * r.z - x.dual.y * r.w + x.dual.z * r.x));
		
		tvec4<T, P> const c(
			xz - wy,
			yz + wx,
			rr.w + rr.z - rr.x - rr.y,
			-(x.dual.w * r.z + x.dual.x * r.y - x.dual.y * r.x - x.dual.z * r.w));
		
		return tmat3x4<T, P>(a, b, c);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> dualquat_cast(tmat2x4<T, P> const & x)
	{
		return tdualquat<T, P>(
			tquat<T, P>( x[0].w, x[0].x, x[0].y, x[0].z ),
			tquat<T, P>( x[1].w, x[1].x, x[1].y, x[1].z ));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tdualquat<T, P> dualquat_cast(tmat3x4<T, P> const & x)
	{
		tquat<T, P> real(uninitialize);
		
		T const trace = x[0].x + x[1].y + x[2].z;
		if(trace > static_cast<T>(0))
		{
			T const r = sqrt(T(1) + trace);
			T const invr = static_cast<T>(0.5) / r;
			real.w = static_cast<T>(0.5) * r;
			real.x = (x[2].y - x[1].z) * invr;
			real.y = (x[0].z - x[2].x) * invr;
			real.z = (x[1].x - x[0].y) * invr;
		}
		else if(x[0].x > x[1].y && x[0].x > x[2].z)
		{
			T const r = sqrt(T(1) + x[0].x - x[1].y - x[2].z);
			T const invr = static_cast<T>(0.5) / r;
			real.x = static_cast<T>(0.5)*r;
			real.y = (x[1].x + x[0].y) * invr;
			real.z = (x[0].z + x[2].x) * invr;
			real.w = (x[2].y - x[1].z) * invr;
		}
		else if(x[1].y > x[2].z)
		{
			T const r = sqrt(T(1) + x[1].y - x[0].x - x[2].z);
			T const invr = static_cast<T>(0.5) / r;
			real.x = (x[1].x + x[0].y) * invr;
			real.y = static_cast<T>(0.5) * r;
			real.z = (x[2].y + x[1].z) * invr;
			real.w = (x[0].z - x[2].x) * invr;
		}
		else
		{
			T const r = sqrt(T(1) + x[2].z - x[0].x - x[1].y);
			T const invr = static_cast<T>(0.5) / r;
			real.x = (x[0].z + x[2].x) * invr;
			real.y = (x[2].y + x[1].z) * invr;
			real.z = static_cast<T>(0.5) * r;
			real.w = (x[1].x - x[0].y) * invr;
		}
		
		tquat<T, P> dual(uninitialize);
		dual.x =  static_cast<T>(0.5) * ( x[0].w * real.w + x[1].w * real.z - x[2].w * real.y);
		dual.y =  static_cast<T>(0.5) * (-x[0].w * real.z + x[1].w * real.w + x[2].w * real.x);
		dual.z =  static_cast<T>(0.5) * ( x[0].w * real.y - x[1].w * real.x + x[2].w * real.w);
		dual.w = -static_cast<T>(0.5) * ( x[0].w * real.x + x[1].w * real.y + x[2].w * real.z);
		return tdualquat<T, P>(real, dual);
	}
}//namespace glm
