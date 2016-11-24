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
/// @file glm/detail/type_mat2x4.inl
/// @date 2006-08-05 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	//////////////////////////////////////////////////////////////
	// Constructors

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4()
	{
#		ifndef GLM_FORCE_NO_CTOR_INIT 
			this->value[0] = col_type(1, 0, 0, 0);
			this->value[1] = col_type(0, 1, 0, 0);
#		endif
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat2x4<T, P> const & m)
	{
		this->value[0] = m.value[0];
		this->value[1] = m.value[1];
	}

	template <typename T, precision P>
	template <precision Q>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat2x4<T, Q> const & m)
	{
		this->value[0] = m.value[0];
		this->value[1] = m.value[1];
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(ctor)
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(T const & s)
	{
		value_type const Zero(0);
		this->value[0] = col_type(s, Zero, Zero, Zero);
		this->value[1] = col_type(Zero, s, Zero, Zero);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4
	(
		T const & x0, T const & y0, T const & z0, T const & w0,
		T const & x1, T const & y1, T const & z1, T const & w1
	)
	{
		this->value[0] = col_type(x0, y0, z0, w0);
		this->value[1] = col_type(x1, y1, z1, w1);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(col_type const & v0, col_type const & v1)
	{
		this->value[0] = v0;
		this->value[1] = v1;
	}

	//////////////////////////////////////
	// Conversion constructors
	template <typename T, precision P>
	template <
		typename X1, typename Y1, typename Z1, typename W1,
		typename X2, typename Y2, typename Z2, typename W2>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4
	(
		X1 const & x1, Y1 const & y1, Z1 const & z1, W1 const & w1,
		X2 const & x2, Y2 const & y2, Z2 const & z2, W2 const & w2
	)
	{
		this->value[0] = col_type(static_cast<T>(x1), value_type(y1), value_type(z1), value_type(w1));
		this->value[1] = col_type(static_cast<T>(x2), value_type(y2), value_type(z2), value_type(w2));
	}
	
	template <typename T, precision P>
	template <typename V1, typename V2>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tvec4<V1, P> const & v1, tvec4<V2, P> const & v2)
	{
		this->value[0] = col_type(v1);
		this->value[1] = col_type(v2);
	}

	//////////////////////////////////////
	// Matrix conversions

	template <typename T, precision P>
	template <typename U, precision Q>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat2x4<U, Q> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat2x2<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0, 0);
		this->value[1] = col_type(m[1], 0, 0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat3x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0);
		this->value[1] = col_type(m[1], 0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat4x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat2x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0);
		this->value[1] = col_type(m[1], 0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat3x2<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0, 0);
		this->value[1] = col_type(m[1], 0, 0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat3x4<T, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat4x2<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0, 0);
		this->value[1] = col_type(m[1], 0, 0);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>::tmat2x4(tmat4x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0], 0);
		this->value[1] = col_type(m[1], 0);
	}

	//////////////////////////////////////
	// Accesses

#	ifdef GLM_FORCE_SIZE_FUNC
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tmat2x4<T, P>::size_type tmat2x4<T, P>::size() const
		{
			return 2;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x4<T, P>::col_type & tmat2x4<T, P>::operator[](typename tmat2x4<T, P>::size_type i)
		{
			assert(i < this->size());
			return this->value[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x4<T, P>::col_type const & tmat2x4<T, P>::operator[](typename tmat2x4<T, P>::size_type i) const
		{
			assert(i < this->size());
			return this->value[i];
		}
#	else
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tmat2x4<T, P>::length_type tmat2x4<T, P>::length() const
		{
			return 2;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x4<T, P>::col_type & tmat2x4<T, P>::operator[](typename tmat2x4<T, P>::length_type i)
		{
			assert(i < this->length());
			return this->value[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x4<T, P>::col_type const & tmat2x4<T, P>::operator[](typename tmat2x4<T, P>::length_type i) const
		{
			assert(i < this->length());
			return this->value[i];
		}
#	endif//GLM_FORCE_SIZE_FUNC

	//////////////////////////////////////////////////////////////
	// Unary updatable operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator=(tmat2x4<T, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator=(tmat2x4<U, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator+=(U s)
	{
		this->value[0] += s;
		this->value[1] += s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator+=(tmat2x4<U, P> const & m)
	{
		this->value[0] += m[0];
		this->value[1] += m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator-=(U s)
	{
		this->value[0] -= s;
		this->value[1] -= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator-=(tmat2x4<U, P> const & m)
	{
		this->value[0] -= m[0];
		this->value[1] -= m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator*=(U s)
	{
		this->value[0] *= s;
		this->value[1] *= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> & tmat2x4<T, P>::operator/=(U s)
	{
		this->value[0] /= s;
		this->value[1] /= s;
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator++()
	{
		++this->value[0];
		++this->value[1];
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P>& tmat2x4<T, P>::operator--()
	{
		--this->value[0];
		--this->value[1];
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> tmat2x4<T, P>::operator++(int)
	{
		tmat2x4<T, P> Result(*this);
		++*this;
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> tmat2x4<T, P>::operator--(int)
	{
		tmat2x4<T, P> Result(*this);
		--*this;
		return Result;
	}

	//////////////////////////////////////////////////////////////
	// Binary operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator+(tmat2x4<T, P> const & m, T const & s)
	{
		return tmat2x4<T, P>(
			m[0] + s,
			m[1] + s);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator+(tmat2x4<T, P> const & m1, tmat2x4<T, P> const & m2)
	{
		return tmat2x4<T, P>(
			m1[0] + m2[0],
			m1[1] + m2[1]);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator-(tmat2x4<T, P> const & m, T const & s)
	{
		return tmat2x4<T, P>(
			m[0] - s,
			m[1] - s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator-(tmat2x4<T, P> const & m1, tmat2x4<T, P> const & m2)
	{
		return tmat2x4<T, P>(
			m1[0] - m2[0],
			m1[1] - m2[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator*(tmat2x4<T, P> const & m, T const & s)
	{
		return tmat2x4<T, P>(
			m[0] * s,
			m[1] * s);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator*(T const & s, tmat2x4<T, P> const & m)
	{
		return tmat2x4<T, P>(
			m[0] * s,
			m[1] * s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat2x4<T, P>::col_type operator*(tmat2x4<T, P> const & m, typename tmat2x4<T, P>::row_type const & v)
	{
		return typename tmat2x4<T, P>::col_type(
			m[0][0] * v.x + m[1][0] * v.y,
			m[0][1] * v.x + m[1][1] * v.y,
			m[0][2] * v.x + m[1][2] * v.y,
			m[0][3] * v.x + m[1][3] * v.y);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat2x4<T, P>::row_type operator*(typename tmat2x4<T, P>::col_type const & v, tmat2x4<T, P> const & m)
	{
		return typename tmat2x4<T, P>::row_type(
			v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2] + v.w * m[0][3],
			v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2] + v.w * m[1][3]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x4<T, P> operator*(tmat2x4<T, P> const & m1, tmat4x2<T, P> const & m2)
	{
		T SrcA00 = m1[0][0];
		T SrcA01 = m1[0][1];
		T SrcA02 = m1[0][2];
		T SrcA03 = m1[0][3];
		T SrcA10 = m1[1][0];
		T SrcA11 = m1[1][1];
		T SrcA12 = m1[1][2];
		T SrcA13 = m1[1][3];

		T SrcB00 = m2[0][0];
		T SrcB01 = m2[0][1];
		T SrcB10 = m2[1][0];
		T SrcB11 = m2[1][1];
		T SrcB20 = m2[2][0];
		T SrcB21 = m2[2][1];
		T SrcB30 = m2[3][0];
		T SrcB31 = m2[3][1];

		tmat4x4<T, P> Result(uninitialize);
		Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01;
		Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01;
		Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01;
		Result[0][3] = SrcA03 * SrcB00 + SrcA13 * SrcB01;
		Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11;
		Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11;
		Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11;
		Result[1][3] = SrcA03 * SrcB10 + SrcA13 * SrcB11;
		Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21;
		Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21;
		Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21;
		Result[2][3] = SrcA03 * SrcB20 + SrcA13 * SrcB21;
		Result[3][0] = SrcA00 * SrcB30 + SrcA10 * SrcB31;
		Result[3][1] = SrcA01 * SrcB30 + SrcA11 * SrcB31;
		Result[3][2] = SrcA02 * SrcB30 + SrcA12 * SrcB31;
		Result[3][3] = SrcA03 * SrcB30 + SrcA13 * SrcB31;
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator*(tmat2x4<T, P> const & m1, tmat2x2<T, P> const & m2)
	{
		return tmat2x4<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1],
			m1[0][3] * m2[0][0] + m1[1][3] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
			m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1],
			m1[0][3] * m2[1][0] + m1[1][3] * m2[1][1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x4<T, P> operator*(tmat2x4<T, P> const & m1, tmat3x2<T, P> const & m2)
	{
		return tmat3x4<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1],
			m1[0][3] * m2[0][0] + m1[1][3] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
			m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1],
			m1[0][3] * m2[1][0] + m1[1][3] * m2[1][1],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1],
			m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1],
			m1[0][3] * m2[2][0] + m1[1][3] * m2[2][1]);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator/(tmat2x4<T, P> const & m, T s)
	{
		return tmat2x4<T, P>(
			m[0] / s,
			m[1] / s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> operator/(T s, tmat2x4<T, P> const & m)
	{
		return tmat2x4<T, P>(
			s / m[0],
			s / m[1]);
	}

	// Unary constant operators
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x4<T, P> const operator-(tmat2x4<T, P> const & m)
	{
		return tmat2x4<T, P>(
			-m[0], 
			-m[1]);
	}

	//////////////////////////////////////
	// Boolean operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator==(tmat2x4<T, P> const & m1, tmat2x4<T, P> const & m2)
	{
		return (m1[0] == m2[0]) && (m1[1] == m2[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator!=(tmat2x4<T, P> const & m1, tmat2x4<T, P> const & m2)
	{
		return (m1[0] != m2[0]) || (m1[1] != m2[1]);
	}
} //namespace glm
