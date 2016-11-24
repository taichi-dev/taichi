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
/// @file glm/detail/type_mat4x3.inl
/// @date 2006-04-17 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	//////////////////////////////////////////////////////////////
	// Constructors

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3()
	{
#		ifndef GLM_FORCE_NO_CTOR_INIT 
			this->value[0] = col_type(1, 0, 0);
			this->value[1] = col_type(0, 1, 0);
			this->value[2] = col_type(0, 0, 1);
			this->value[3] = col_type(0, 0, 0);
#		endif
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat4x3<T, P> const & m)
	{
		this->value[0] = m.value[0];
		this->value[1] = m.value[1];
		this->value[2] = m.value[2];
		this->value[3] = m.value[3];
	}

	template <typename T, precision P>
	template <precision Q>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat4x3<T, Q> const & m)
	{
		this->value[0] = m.value[0];
		this->value[1] = m.value[1];
		this->value[2] = m.value[2];
		this->value[3] = m.value[3];
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(ctor)
	{}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(T const & s)
	{
		this->value[0] = col_type(s, 0, 0);
		this->value[1] = col_type(0, s, 0);
		this->value[2] = col_type(0, 0, s);
		this->value[3] = col_type(0, 0, 0);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3
	(
		T const & x0, T const & y0, T const & z0,
		T const & x1, T const & y1, T const & z1,
		T const & x2, T const & y2, T const & z2,
		T const & x3, T const & y3, T const & z3
	)
	{
		this->value[0] = col_type(x0, y0, z0);
		this->value[1] = col_type(x1, y1, z1);
		this->value[2] = col_type(x2, y2, z2);
		this->value[3] = col_type(x3, y3, z3);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3
	(
		col_type const & v0, 
		col_type const & v1, 
		col_type const & v2,
		col_type const & v3
	)
	{
		this->value[0] = v0;
		this->value[1] = v1;
		this->value[2] = v2;
		this->value[3] = v3;
	}

	//////////////////////////////////////
	// Conversion constructors

	template <typename T, precision P> 
	template <
		typename X1, typename Y1, typename Z1,
		typename X2, typename Y2, typename Z2,
		typename X3, typename Y3, typename Z3,
		typename X4, typename Y4, typename Z4>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3
	(
		X1 const & x1, Y1 const & y1, Z1 const & z1,
		X2 const & x2, Y2 const & y2, Z2 const & z2,
		X3 const & x3, Y3 const & y3, Z3 const & z3,
		X4 const & x4, Y4 const & y4, Z4 const & z4
	)
	{
		this->value[0] = col_type(static_cast<T>(x1), value_type(y1), value_type(z1));
		this->value[1] = col_type(static_cast<T>(x2), value_type(y2), value_type(z2));
		this->value[2] = col_type(static_cast<T>(x3), value_type(y3), value_type(z3));
		this->value[3] = col_type(static_cast<T>(x4), value_type(y4), value_type(z4));
	}
	
	template <typename T, precision P>
	template <typename V1, typename V2, typename V3, typename V4>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3
	(
		tvec3<V1, P> const & v1,
		tvec3<V2, P> const & v2,
		tvec3<V3, P> const & v3,
		tvec3<V4, P> const & v4
	)
	{
		this->value[0] = col_type(v1);
		this->value[1] = col_type(v2);
		this->value[2] = col_type(v3);
		this->value[3] = col_type(v4);
	}

	//////////////////////////////////////////////////////////////
	// Matrix conversions

	template <typename T, precision P>
	template <typename U, precision Q>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat4x3<U, Q> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
		this->value[2] = col_type(m[2]);
		this->value[3] = col_type(m[3]);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat2x2<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0);
		this->value[1] = col_type(m[1], 0);
		this->value[2] = col_type(m[2], 1);
		this->value[3] = col_type(0);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat3x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
		this->value[2] = col_type(m[2]);
		this->value[3] = col_type(0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat4x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
		this->value[2] = col_type(m[2]);
		this->value[3] = col_type(m[3]);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat2x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
		this->value[2] = col_type(0, 0, 1);
		this->value[3] = col_type(0);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat3x2<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0);
		this->value[1] = col_type(m[1], 0);
		this->value[2] = col_type(m[2], 1);
		this->value[3] = col_type(0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat2x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
		this->value[2] = col_type(0, 0, 1);
		this->value[3] = col_type(0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat4x2<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0);
		this->value[1] = col_type(m[1], 0);
		this->value[2] = col_type(m[2], 1);
		this->value[3] = col_type(m[3], 0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>::tmat4x3(tmat3x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
		this->value[2] = col_type(m[2]);
		this->value[3] = col_type(0);
	}

	//////////////////////////////////////
	// Accesses

#	ifdef GLM_FORCE_SIZE_FUNC
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tmat4x3<T, P>::size_type tmat4x3<T, P>::size() const
		{
			return 4;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat4x3<T, P>::col_type & tmat4x3<T, P>::operator[](typename tmat4x3<T, P>::size_type i)
		{
			assert(i < this->size());
			return this->value[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat4x3<T, P>::col_type const & tmat4x3<T, P>::operator[](typename tmat4x3<T, P>::size_type i) const
		{
			assert(i < this->size());
			return this->value[i];
		}
#	else
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tmat4x3<T, P>::length_type tmat4x3<T, P>::length() const
		{
			return 4;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat4x3<T, P>::col_type & tmat4x3<T, P>::operator[](typename tmat4x3<T, P>::length_type i)
		{
			assert(i < this->length());
			return this->value[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat4x3<T, P>::col_type const & tmat4x3<T, P>::operator[](typename tmat4x3<T, P>::length_type i) const
		{
			assert(i < this->length());
			return this->value[i];
		}
#	endif//GLM_FORCE_SIZE_FUNC

	//////////////////////////////////////////////////////////////
	// Unary updatable operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>& tmat4x3<T, P>::operator=(tmat4x3<T, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		this->value[2] = m[2];
		this->value[3] = m[3];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P>& tmat4x3<T, P>::operator=(tmat4x3<U, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		this->value[2] = m[2];
		this->value[3] = m[3];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator+=(U s)
	{
		this->value[0] += s;
		this->value[1] += s;
		this->value[2] += s;
		this->value[3] += s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator+=(tmat4x3<U, P> const & m)
	{
		this->value[0] += m[0];
		this->value[1] += m[1];
		this->value[2] += m[2];
		this->value[3] += m[3];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator-=(U s)
	{
		this->value[0] -= s;
		this->value[1] -= s;
		this->value[2] -= s;
		this->value[3] -= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator-=(tmat4x3<U, P> const & m)
	{
		this->value[0] -= m[0];
		this->value[1] -= m[1];
		this->value[2] -= m[2];
		this->value[3] -= m[3];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator*=(U s)
	{
		this->value[0] *= s;
		this->value[1] *= s;
		this->value[2] *= s;
		this->value[3] *= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator/=(U s)
	{
		this->value[0] /= s;
		this->value[1] /= s;
		this->value[2] /= s;
		this->value[3] /= s;
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator++()
	{
		++this->value[0];
		++this->value[1];
		++this->value[2];
		++this->value[3];
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> & tmat4x3<T, P>::operator--()
	{
		--this->value[0];
		--this->value[1];
		--this->value[2];
		--this->value[3];
		return *this;
	}

	//////////////////////////////////////////////////////////////
	// Binary operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator+(tmat4x3<T, P> const & m, T const & s)
	{
		return tmat4x3<T, P>(
			m[0] + s,
			m[1] + s,
			m[2] + s,
			m[3] + s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator+(tmat4x3<T, P> const & m1, tmat4x3<T, P> const & m2)
	{
		return tmat4x3<T, P>(
			m1[0] + m2[0],
			m1[1] + m2[1],
			m1[2] + m2[2],
			m1[3] + m2[3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator-(tmat4x3<T, P> const & m, T const & s)
	{
		return tmat4x3<T, P>(
			m[0] - s,
			m[1] - s,
			m[2] - s,
			m[3] - s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator-(tmat4x3<T, P> const & m1, tmat4x3<T, P> const & m2)
	{
		return tmat4x3<T, P>(
			m1[0] - m2[0],
			m1[1] - m2[1],
			m1[2] - m2[2],
			m1[3] - m2[3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator*(tmat4x3<T, P> const & m, T const & s)
	{
		return tmat4x3<T, P>(
			m[0] * s,
			m[1] * s,
			m[2] * s,
			m[3] * s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator*(T const & s, tmat4x3<T, P> const & m)
	{
		return tmat4x3<T, P>(
			m[0] * s,
			m[1] * s,
			m[2] * s,
			m[3] * s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat4x3<T, P>::col_type operator*
	(
		tmat4x3<T, P> const & m,
		typename tmat4x3<T, P>::row_type const & v)
	{
		return typename tmat4x3<T, P>::col_type(
			m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
			m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
			m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat4x3<T, P>::row_type operator*
	(
		typename tmat4x3<T, P>::col_type const & v,
		tmat4x3<T, P> const & m)
	{
		return typename tmat4x3<T, P>::row_type(
			v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2],
			v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2],
			v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2],
			v.x * m[3][0] + v.y * m[3][1] + v.z * m[3][2]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x3<T, P> operator*(tmat4x3<T, P> const & m1, tmat2x4<T, P> const & m2)
	{
		return tmat2x3<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2] + m1[3][0] * m2[0][3],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2] + m1[3][1] * m2[0][3],
			m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2] + m1[3][2] * m2[0][3],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2] + m1[3][0] * m2[1][3],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2] + m1[3][1] * m2[1][3],
			m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2] + m1[3][2] * m2[1][3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x3<T, P> operator*(tmat4x3<T, P> const & m1, tmat3x4<T, P> const & m2)
	{
		T const SrcA00 = m1[0][0];
		T const SrcA01 = m1[0][1];
		T const SrcA02 = m1[0][2];
		T const SrcA10 = m1[1][0];
		T const SrcA11 = m1[1][1];
		T const SrcA12 = m1[1][2];
		T const SrcA20 = m1[2][0];
		T const SrcA21 = m1[2][1];
		T const SrcA22 = m1[2][2];
		T const SrcA30 = m1[3][0];
		T const SrcA31 = m1[3][1];
		T const SrcA32 = m1[3][2];

		T const SrcB00 = m2[0][0];
		T const SrcB01 = m2[0][1];
		T const SrcB02 = m2[0][2];
		T const SrcB03 = m2[0][3];
		T const SrcB10 = m2[1][0];
		T const SrcB11 = m2[1][1];
		T const SrcB12 = m2[1][2];
		T const SrcB13 = m2[1][3];
		T const SrcB20 = m2[2][0];
		T const SrcB21 = m2[2][1];
		T const SrcB22 = m2[2][2];
		T const SrcB23 = m2[2][3];

		tmat3x3<T, P> Result(uninitialize);
		Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01 + SrcA20 * SrcB02 + SrcA30 * SrcB03;
		Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02 + SrcA31 * SrcB03;
		Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01 + SrcA22 * SrcB02 + SrcA32 * SrcB03;
		Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12 + SrcA30 * SrcB13;
		Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12 + SrcA31 * SrcB13;
		Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11 + SrcA22 * SrcB12 + SrcA32 * SrcB13;
		Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21 + SrcA20 * SrcB22 + SrcA30 * SrcB23;
		Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21 + SrcA21 * SrcB22 + SrcA31 * SrcB23;
		Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21 + SrcA22 * SrcB22 + SrcA32 * SrcB23;
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator*(tmat4x3<T, P> const & m1, tmat4x4<T, P> const & m2)
	{
		return tmat4x3<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2] + m1[3][0] * m2[0][3],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2] + m1[3][1] * m2[0][3],
			m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2] + m1[3][2] * m2[0][3],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2] + m1[3][0] * m2[1][3],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2] + m1[3][1] * m2[1][3],
			m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2] + m1[3][2] * m2[1][3],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2] + m1[3][0] * m2[2][3],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2] + m1[3][1] * m2[2][3],
			m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2] + m1[3][2] * m2[2][3],
			m1[0][0] * m2[3][0] + m1[1][0] * m2[3][1] + m1[2][0] * m2[3][2] + m1[3][0] * m2[3][3],
			m1[0][1] * m2[3][0] + m1[1][1] * m2[3][1] + m1[2][1] * m2[3][2] + m1[3][1] * m2[3][3],
			m1[0][2] * m2[3][0] + m1[1][2] * m2[3][1] + m1[2][2] * m2[3][2] + m1[3][2] * m2[3][3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator/(tmat4x3<T, P> const & m, T const & s)
	{
		return tmat4x3<T, P>(
			m[0] / s,
			m[1] / s,
			m[2] / s,
			m[3] / s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> operator/(T const & s, tmat4x3<T, P> const & m)
	{
		return tmat4x3<T, P>(
			s / m[0],
			s / m[1],
			s / m[2],
			s / m[3]);
	}

	// Unary constant operators
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> const operator-(tmat4x3<T, P> const & m)
	{
		return tmat4x3<T, P>(
			-m[0],
			-m[1],
			-m[2],
			-m[3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> const operator++(tmat4x3<T, P> const & m, int)
	{
		return tmat4x3<T, P>(
			m[0] + T(1),
			m[1] + T(1),
			m[2] + T(1),
			m[3] + T(1));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> const operator--(tmat4x3<T, P> const & m, int)
	{
		return tmat4x3<T, P>(
			m[0] - T(1),
			m[1] - T(1),
			m[2] - T(1),
			m[3] - T(1));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> tmat4x3<T, P>::operator++(int)
	{
		tmat4x3<T, P> Result(*this);
		++*this;
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x3<T, P> tmat4x3<T, P>::operator--(int)
	{
		tmat4x3<T, P> Result(*this);
		--*this;
		return Result;
	}

	//////////////////////////////////////
	// Boolean operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator==(tmat4x3<T, P> const & m1, tmat4x3<T, P> const & m2)
	{
		return (m1[0] == m2[0]) && (m1[1] == m2[1]) && (m1[2] == m2[2]) && (m1[3] == m2[3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator!=(tmat4x3<T, P> const & m1, tmat4x3<T, P> const & m2)
	{
		return (m1[0] != m2[0]) || (m1[1] != m2[1]) || (m1[2] != m2[2]) || (m1[3] != m2[3]);
	}
} //namespace glm
