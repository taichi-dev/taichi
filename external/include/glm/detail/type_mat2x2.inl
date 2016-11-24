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
/// @file glm/detail/type_mat2x2.inl
/// @date 2005-01-16 / 2011-06-15
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail
{
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> compute_inverse(tmat2x2<T, P> const & m)
	{
		T OneOverDeterminant = static_cast<T>(1) / (
			+ m[0][0] * m[1][1]
			- m[1][0] * m[0][1]);

		tmat2x2<T, P> Inverse(
			+ m[1][1] * OneOverDeterminant,
			- m[0][1] * OneOverDeterminant,
			- m[1][0] * OneOverDeterminant,
			+ m[0][0] * OneOverDeterminant);

		return Inverse;
	}
}//namespace detail

	//////////////////////////////////////////////////////////////
	// Constructors

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2()
	{
#		ifndef GLM_FORCE_NO_CTOR_INIT 
			this->value[0] = col_type(1, 0);
			this->value[1] = col_type(0, 1);
#		endif
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat2x2<T, P> const & m)
	{
		this->value[0] = m.value[0];
		this->value[1] = m.value[1];
	}

	template <typename T, precision P>
	template <precision Q>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat2x2<T, Q> const & m)
	{
		this->value[0] = m.value[0];
		this->value[1] = m.value[1];
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(ctor)
	{}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(T const & s)
	{
		this->value[0] = col_type(s, 0);
		this->value[1] = col_type(0, s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2
	(
		T const & x0, T const & y0,
		T const & x1, T const & y1
	)
	{
		this->value[0] = col_type(x0, y0);
		this->value[1] = col_type(x1, y1);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(col_type const & v0, col_type const & v1)
	{
		this->value[0] = v0;
		this->value[1] = v1;
	}

	//////////////////////////////////////
	// Conversion constructors
	template <typename T, precision P>
	template <typename X1, typename Y1, typename X2, typename Y2>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2
	(
		X1 const & x1, Y1 const & y1,
		X2 const & x2, Y2 const & y2
	)
	{
		this->value[0] = col_type(static_cast<T>(x1), value_type(y1));
		this->value[1] = col_type(static_cast<T>(x2), value_type(y2));
	}
	
	template <typename T, precision P>
	template <typename V1, typename V2>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tvec2<V1, P> const & v1, tvec2<V2, P> const & v2)
	{
		this->value[0] = col_type(v1);
		this->value[1] = col_type(v2);
	}

	//////////////////////////////////////////////////////////////
	// mat2x2 matrix conversions

	template <typename T, precision P>
	template <typename U, precision Q>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat2x2<U, Q> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat3x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat4x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat2x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat3x2<T, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat2x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat4x2<T, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat3x4<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>::tmat2x2(tmat4x3<T, P> const & m)
	{
		this->value[0] = col_type(m[0]);
		this->value[1] = col_type(m[1]);
	}

	//////////////////////////////////////
	// Accesses

#	ifdef GLM_FORCE_SIZE_FUNC
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tmat2x2<T, P>::size_type tmat2x2<T, P>::size() const
		{
			return 2;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::col_type & tmat2x2<T, P>::operator[](typename tmat2x2<T, P>::size_type i)
		{
			assert(i < this->size());
			return this->value[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::col_type const & tmat2x2<T, P>::operator[](typename tmat2x2<T, P>::size_type i) const
		{
			assert(i < this->size());
			return this->value[i];
		}
#	else
		template <typename T, precision P>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename tmat2x2<T, P>::length_type tmat2x2<T, P>::length() const
		{
			return 2;
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::col_type & tmat2x2<T, P>::operator[](typename tmat2x2<T, P>::length_type i)
		{
			assert(i < this->length());
			return this->value[i];
		}

		template <typename T, precision P>
		GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::col_type const & tmat2x2<T, P>::operator[](typename tmat2x2<T, P>::length_type i) const
		{
			assert(i < this->length());
			return this->value[i];
		}
#	endif//GLM_FORCE_SIZE_FUNC

	//////////////////////////////////////////////////////////////
	// Unary updatable operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator=(tmat2x2<T, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator=(tmat2x2<U, P> const & m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator+=(U s)
	{
		this->value[0] += s;
		this->value[1] += s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator+=(tmat2x2<U, P> const & m)
	{
		this->value[0] += m[0];
		this->value[1] += m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator-=(U s)
	{
		this->value[0] -= s;
		this->value[1] -= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator-=(tmat2x2<U, P> const & m)
	{
		this->value[0] -= m[0];
		this->value[1] -= m[1];
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator*=(U s)
	{
		this->value[0] *= s;
		this->value[1] *= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator*=(tmat2x2<U, P> const & m)
	{
		return (*this = *this * m);
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator/=(U s)
	{
		this->value[0] /= s;
		this->value[1] /= s;
		return *this;
	}

	template <typename T, precision P>
	template <typename U>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator/=(tmat2x2<U, P> const & m)
	{
		return (*this = *this * detail::compute_inverse<T, P>(m));
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator++()
	{
		++this->value[0];
		++this->value[1];
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P>& tmat2x2<T, P>::operator--()
	{
		--this->value[0];
		--this->value[1];
		return *this;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> tmat2x2<T, P>::operator++(int)
	{
		tmat2x2<T, P> Result(*this);
		++*this;
		return Result;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> tmat2x2<T, P>::operator--(int)
	{
		tmat2x2<T, P> Result(*this);
		--*this;
		return Result;
	}

	//////////////////////////////////////////////////////////////
	// Binary operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator+(tmat2x2<T, P> const & m, T const & s)
	{
		return tmat2x2<T, P>(
			m[0] + s,
			m[1] + s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator+(T const & s, tmat2x2<T, P> const & m)
	{
		return tmat2x2<T, P>(
			m[0] + s,
			m[1] + s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator+(tmat2x2<T, P> const & m1, tmat2x2<T, P> const & m2)
	{
		return tmat2x2<T, P>(
			m1[0] + m2[0],
			m1[1] + m2[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator-(tmat2x2<T, P> const & m, T const & s)
	{
		return tmat2x2<T, P>(
			m[0] - s,
			m[1] - s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator-(T const & s, tmat2x2<T, P> const & m)
	{
		return tmat2x2<T, P>(
			s - m[0],
			s - m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator-(tmat2x2<T, P> const & m1, tmat2x2<T, P> const & m2)
	{
		return tmat2x2<T, P>(
			m1[0] - m2[0],
			m1[1] - m2[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator*(tmat2x2<T, P> const & m,	T const & s)
	{
		return tmat2x2<T, P>(
			m[0] * s,
			m[1] * s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator*(T const & s, tmat2x2<T, P> const & m)
	{
		return tmat2x2<T, P>(
			m[0] * s,
			m[1] * s);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::col_type operator*
	(
		tmat2x2<T, P> const & m,
		typename tmat2x2<T, P>::row_type const & v
	)
	{
		return tvec2<T, P>(
			m[0][0] * v.x + m[1][0] * v.y,
			m[0][1] * v.x + m[1][1] * v.y);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::row_type operator*
	(
		typename tmat2x2<T, P>::col_type const & v,
		tmat2x2<T, P> const & m
	)
	{
		return tvec2<T, P>(
			v.x * m[0][0] + v.y * m[0][1],
			v.x * m[1][0] + v.y * m[1][1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator*(tmat2x2<T, P> const & m1, tmat2x2<T, P> const & m2)
	{
		return tmat2x2<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat3x2<T, P> operator*(tmat2x2<T, P> const & m1, tmat3x2<T, P> const & m2)
	{
		return tmat3x2<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat4x2<T, P> operator*(tmat2x2<T, P> const & m1, tmat4x2<T, P> const & m2)
	{
		return tmat4x2<T, P>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1],
			m1[0][0] * m2[3][0] + m1[1][0] * m2[3][1],
			m1[0][1] * m2[3][0] + m1[1][1] * m2[3][1]);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator/(tmat2x2<T, P> const & m,	T const & s)
	{
		return tmat2x2<T, P>(
			m[0] / s,
			m[1] / s);
	}

	template <typename T, precision P> 
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator/(T const & s, tmat2x2<T, P> const & m)
	{
		return tmat2x2<T, P>(
			s / m[0],
			s / m[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::col_type operator/(tmat2x2<T, P> const & m, typename tmat2x2<T, P>::row_type const & v)
	{
		return detail::compute_inverse<T, P>(m) * v;
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER typename tmat2x2<T, P>::row_type operator/(typename tmat2x2<T, P>::col_type const & v, tmat2x2<T, P> const & m)
	{
		return v * detail::compute_inverse<T, P>(m);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> operator/(tmat2x2<T, P> const & m1, tmat2x2<T, P> const & m2)
	{	
		tmat2x2<T, P> m1_copy(m1);
		return m1_copy /= m2;
	}

	// Unary constant operators
	template <typename T, precision P>
	GLM_FUNC_QUALIFIER tmat2x2<T, P> const operator-(tmat2x2<T, P> const & m)
	{
		return tmat2x2<T, P>(
			-m[0], 
			-m[1]);
	}

	//////////////////////////////////////
	// Boolean operators

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator==(tmat2x2<T, P> const & m1, tmat2x2<T, P> const & m2)
	{
		return (m1[0] == m2[0]) && (m1[1] == m2[1]);
	}

	template <typename T, precision P>
	GLM_FUNC_QUALIFIER bool operator!=(tmat2x2<T, P> const & m1, tmat2x2<T, P> const & m2)
	{
		return (m1[0] != m2[0]) || (m1[1] != m2[1]);
	}
} //namespace glm
