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
/// @ref gtx_simd_mat4
/// @file glm/gtx/simd_mat4.inl
/// @date 2009-05-07 / 2011-06-07
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

GLM_FUNC_QUALIFIER length_t fmat4x4SIMD::length() const
{
	return 4;
}

//////////////////////////////////////
// Accesses

GLM_FUNC_QUALIFIER fvec4SIMD & fmat4x4SIMD::operator[]
(
	length_t i
)
{
	assert(i < this->length());

	return this->Data[i];
}

GLM_FUNC_QUALIFIER fvec4SIMD const & fmat4x4SIMD::operator[]
(
	length_t i
) const
{
	assert(i < this->length());

	return this->Data[i];
}

//////////////////////////////////////////////////////////////
// Constructors

GLM_FUNC_QUALIFIER fmat4x4SIMD::fmat4x4SIMD()
{
#	ifndef GLM_FORCE_NO_CTOR_INIT
		this->Data[0] = fvec4SIMD(1, 0, 0, 0);
		this->Data[1] = fvec4SIMD(0, 1, 0, 0);
		this->Data[2] = fvec4SIMD(0, 0, 1, 0);
		this->Data[3] = fvec4SIMD(0, 0, 0, 1);
#	endif
}

GLM_FUNC_QUALIFIER fmat4x4SIMD::fmat4x4SIMD(float const & s)
{
	this->Data[0] = fvec4SIMD(s, 0, 0, 0);
	this->Data[1] = fvec4SIMD(0, s, 0, 0);
	this->Data[2] = fvec4SIMD(0, 0, s, 0);
	this->Data[3] = fvec4SIMD(0, 0, 0, s);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD::fmat4x4SIMD
(
	float const & x0, float const & y0, float const & z0, float const & w0,
	float const & x1, float const & y1, float const & z1, float const & w1,
	float const & x2, float const & y2, float const & z2, float const & w2,
	float const & x3, float const & y3, float const & z3, float const & w3
)
{
	this->Data[0] = fvec4SIMD(x0, y0, z0, w0);
	this->Data[1] = fvec4SIMD(x1, y1, z1, w1);
	this->Data[2] = fvec4SIMD(x2, y2, z2, w2);
	this->Data[3] = fvec4SIMD(x3, y3, z3, w3);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD::fmat4x4SIMD
(
	fvec4SIMD const & v0,
	fvec4SIMD const & v1,
	fvec4SIMD const & v2,
	fvec4SIMD const & v3
)
{
	this->Data[0] = v0;
	this->Data[1] = v1;
	this->Data[2] = v2;
	this->Data[3] = v3;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD::fmat4x4SIMD
(
	mat4 const & m
)
{
	this->Data[0] = fvec4SIMD(m[0]);
	this->Data[1] = fvec4SIMD(m[1]);
	this->Data[2] = fvec4SIMD(m[2]);
	this->Data[3] = fvec4SIMD(m[3]);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD::fmat4x4SIMD
(
	__m128 const in[4]
)
{
	this->Data[0] = in[0];
	this->Data[1] = in[1];
	this->Data[2] = in[2];
	this->Data[3] = in[3];
}

//////////////////////////////////////////////////////////////
// mat4 operators

GLM_FUNC_QUALIFIER fmat4x4SIMD& fmat4x4SIMD::operator= 
(
	fmat4x4SIMD const & m
)
{
	this->Data[0] = m[0];
	this->Data[1] = m[1];
	this->Data[2] = m[2];
	this->Data[3] = m[3];
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator+= 
(
	fmat4x4SIMD const & m
)
{
	this->Data[0].Data = _mm_add_ps(this->Data[0].Data, m[0].Data);
	this->Data[1].Data = _mm_add_ps(this->Data[1].Data, m[1].Data);
	this->Data[2].Data = _mm_add_ps(this->Data[2].Data, m[2].Data);
	this->Data[3].Data = _mm_add_ps(this->Data[3].Data, m[3].Data);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator-= 
(
	fmat4x4SIMD const & m
)
{
	this->Data[0].Data = _mm_sub_ps(this->Data[0].Data, m[0].Data);
	this->Data[1].Data = _mm_sub_ps(this->Data[1].Data, m[1].Data);
	this->Data[2].Data = _mm_sub_ps(this->Data[2].Data, m[2].Data);
	this->Data[3].Data = _mm_sub_ps(this->Data[3].Data, m[3].Data);

	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator*= 
(
	fmat4x4SIMD const & m
)
{
	sse_mul_ps(&this->Data[0].Data, &m.Data[0].Data, &this->Data[0].Data);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator/= 
(
	fmat4x4SIMD const & m
)
{
	__m128 Inv[4];
	sse_inverse_ps(&m.Data[0].Data, Inv);
	sse_mul_ps(&this->Data[0].Data, Inv, &this->Data[0].Data);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator+= 
(
	float const & s
)
{
	__m128 Operand = _mm_set_ps1(s);
	this->Data[0].Data = _mm_add_ps(this->Data[0].Data, Operand);
	this->Data[1].Data = _mm_add_ps(this->Data[1].Data, Operand);
	this->Data[2].Data = _mm_add_ps(this->Data[2].Data, Operand);
	this->Data[3].Data = _mm_add_ps(this->Data[3].Data, Operand);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator-= 
(
	float const & s
)
{
	__m128 Operand = _mm_set_ps1(s);
	this->Data[0].Data = _mm_sub_ps(this->Data[0].Data, Operand);
	this->Data[1].Data = _mm_sub_ps(this->Data[1].Data, Operand);
	this->Data[2].Data = _mm_sub_ps(this->Data[2].Data, Operand);
	this->Data[3].Data = _mm_sub_ps(this->Data[3].Data, Operand);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator*= 
(
	float const & s
)
{
	__m128 Operand = _mm_set_ps1(s);
	this->Data[0].Data = _mm_mul_ps(this->Data[0].Data, Operand);
	this->Data[1].Data = _mm_mul_ps(this->Data[1].Data, Operand);
	this->Data[2].Data = _mm_mul_ps(this->Data[2].Data, Operand);
	this->Data[3].Data = _mm_mul_ps(this->Data[3].Data, Operand);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator/= 
(
	float const & s
)
{
	__m128 Operand = _mm_div_ps(one, _mm_set_ps1(s));
	this->Data[0].Data = _mm_mul_ps(this->Data[0].Data, Operand);
	this->Data[1].Data = _mm_mul_ps(this->Data[1].Data, Operand);
	this->Data[2].Data = _mm_mul_ps(this->Data[2].Data, Operand);
	this->Data[3].Data = _mm_mul_ps(this->Data[3].Data, Operand);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator++ ()
{
	this->Data[0].Data = _mm_add_ps(this->Data[0].Data, one);
	this->Data[1].Data = _mm_add_ps(this->Data[1].Data, one);
	this->Data[2].Data = _mm_add_ps(this->Data[2].Data, one);
	this->Data[3].Data = _mm_add_ps(this->Data[3].Data, one);
	return *this;
}

GLM_FUNC_QUALIFIER fmat4x4SIMD & fmat4x4SIMD::operator-- ()
{
	this->Data[0].Data = _mm_sub_ps(this->Data[0].Data, one);
	this->Data[1].Data = _mm_sub_ps(this->Data[1].Data, one);
	this->Data[2].Data = _mm_sub_ps(this->Data[2].Data, one);
	this->Data[3].Data = _mm_sub_ps(this->Data[3].Data, one);
	return *this;
}


//////////////////////////////////////////////////////////////
// Binary operators

GLM_FUNC_QUALIFIER fmat4x4SIMD operator+
(
	const fmat4x4SIMD &m,
	float const & s
)
{
	return detail::fmat4x4SIMD
	(
		m[0] + s,
		m[1] + s,
		m[2] + s,
		m[3] + s
	);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator+
(
	float const & s,
	const fmat4x4SIMD &m
)
{
	return detail::fmat4x4SIMD
	(
		m[0] + s,
		m[1] + s,
		m[2] + s,
		m[3] + s
	);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator+
(
    const fmat4x4SIMD &m1,
    const fmat4x4SIMD &m2
)
{
    return detail::fmat4x4SIMD
    (
        m1[0] + m2[0],
        m1[1] + m2[1],
        m1[2] + m2[2],
        m1[3] + m2[3]
    );
}


GLM_FUNC_QUALIFIER fmat4x4SIMD operator-
(
    const fmat4x4SIMD &m,
    float const & s
)
{
    return detail::fmat4x4SIMD
    (
        m[0] - s,
        m[1] - s,
        m[2] - s,
        m[3] - s
    );
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator-
(
    float const & s,
    const fmat4x4SIMD &m
)
{
    return detail::fmat4x4SIMD
    (
        s - m[0],
        s - m[1],
        s - m[2],
        s - m[3]
    );
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator-
(
    const fmat4x4SIMD &m1,
    const fmat4x4SIMD &m2
)
{
    return detail::fmat4x4SIMD
    (
        m1[0] - m2[0],
        m1[1] - m2[1],
        m1[2] - m2[2],
        m1[3] - m2[3]
    );
}


GLM_FUNC_QUALIFIER fmat4x4SIMD operator*
(
    const fmat4x4SIMD &m,
    float const & s
)
{
    return detail::fmat4x4SIMD
    (
        m[0] * s,
        m[1] * s,
        m[2] * s,
        m[3] * s
    );
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator*
(
    float const & s,
    const fmat4x4SIMD &m
)
{
    return detail::fmat4x4SIMD
    (
        m[0] * s,
        m[1] * s,
        m[2] * s,
        m[3] * s
    );
}

GLM_FUNC_QUALIFIER fvec4SIMD operator*
(
    const fmat4x4SIMD &m,
    fvec4SIMD const & v
)
{
    return sse_mul_ps(&m.Data[0].Data, v.Data);
}

GLM_FUNC_QUALIFIER fvec4SIMD operator*
(
    fvec4SIMD const & v,
    const fmat4x4SIMD &m
)
{
    return sse_mul_ps(v.Data, &m.Data[0].Data);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator*
(
    const fmat4x4SIMD &m1,
    const fmat4x4SIMD &m2
)
{
    fmat4x4SIMD result;
    sse_mul_ps(&m1.Data[0].Data, &m2.Data[0].Data, &result.Data[0].Data);
    
    return result;
}
    


GLM_FUNC_QUALIFIER fmat4x4SIMD operator/
(
    const fmat4x4SIMD &m,
    float const & s
)
{
    return detail::fmat4x4SIMD
    (
        m[0] / s,
        m[1] / s,
        m[2] / s,
        m[3] / s
    );
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator/
(
    float const & s,
    const fmat4x4SIMD &m
)
{
    return detail::fmat4x4SIMD
    (
        s / m[0],
        s / m[1],
        s / m[2],
        s / m[3]
    );
}

GLM_FUNC_QUALIFIER detail::fmat4x4SIMD inverse(detail::fmat4x4SIMD const & m)
{
	detail::fmat4x4SIMD result;
	detail::sse_inverse_ps(&m[0].Data, &result[0].Data);
	return result;
}

GLM_FUNC_QUALIFIER fvec4SIMD operator/
(
	const fmat4x4SIMD & m,
	fvec4SIMD const & v
)
{
	return inverse(m) * v;
}

GLM_FUNC_QUALIFIER fvec4SIMD operator/
(
	fvec4SIMD const & v,
	const fmat4x4SIMD &m
)
{
	return v * inverse(m);
}

GLM_FUNC_QUALIFIER fmat4x4SIMD operator/
(
	const fmat4x4SIMD &m1,
	const fmat4x4SIMD &m2
)
{
	__m128 result[4];
	__m128 inv[4];

	sse_inverse_ps(&m2.Data[0].Data, inv);
	sse_mul_ps(&m1.Data[0].Data, inv, result);

	return fmat4x4SIMD(result);
}


//////////////////////////////////////////////////////////////
// Unary constant operators
GLM_FUNC_QUALIFIER fmat4x4SIMD const operator-
(
    fmat4x4SIMD const & m
)
{
    return detail::fmat4x4SIMD
    (
        -m[0],
        -m[1],
        -m[2],
        -m[3]
    );
}

GLM_FUNC_QUALIFIER fmat4x4SIMD const operator--
(
    fmat4x4SIMD const & m,
    int
)
{
    return detail::fmat4x4SIMD
    (
        m[0] - 1.0f,
        m[1] - 1.0f,
        m[2] - 1.0f,
        m[3] - 1.0f
    );
}

GLM_FUNC_QUALIFIER fmat4x4SIMD const operator++
(
    fmat4x4SIMD const & m,
    int
)
{
    return detail::fmat4x4SIMD
    (
        m[0] + 1.0f,
        m[1] + 1.0f,
        m[2] + 1.0f,
        m[3] + 1.0f
    );
}

}//namespace detail

GLM_FUNC_QUALIFIER mat4 mat4_cast
(
	detail::fmat4x4SIMD const & x
)
{
	GLM_ALIGN(16) mat4 Result;
	_mm_store_ps(&Result[0][0], x.Data[0].Data);
	_mm_store_ps(&Result[1][0], x.Data[1].Data);
	_mm_store_ps(&Result[2][0], x.Data[2].Data);
	_mm_store_ps(&Result[3][0], x.Data[3].Data);
	return Result;
}

GLM_FUNC_QUALIFIER detail::fmat4x4SIMD matrixCompMult
(
	detail::fmat4x4SIMD const & x,
	detail::fmat4x4SIMD const & y
)
{
	detail::fmat4x4SIMD result;
	result[0] = x[0] * y[0];
	result[1] = x[1] * y[1];
	result[2] = x[2] * y[2];
	result[3] = x[3] * y[3];
	return result;
}

GLM_FUNC_QUALIFIER detail::fmat4x4SIMD outerProduct
(
	detail::fvec4SIMD const & c,
	detail::fvec4SIMD const & r
)
{
	__m128 Shu0 = _mm_shuffle_ps(r.Data, r.Data, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Shu1 = _mm_shuffle_ps(r.Data, r.Data, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 Shu2 = _mm_shuffle_ps(r.Data, r.Data, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 Shu3 = _mm_shuffle_ps(r.Data, r.Data, _MM_SHUFFLE(3, 3, 3, 3));

	detail::fmat4x4SIMD result(uninitialize);
	result[0].Data = _mm_mul_ps(c.Data, Shu0);
	result[1].Data = _mm_mul_ps(c.Data, Shu1);
	result[2].Data = _mm_mul_ps(c.Data, Shu2);
	result[3].Data = _mm_mul_ps(c.Data, Shu3);
	return result;
}

GLM_FUNC_QUALIFIER detail::fmat4x4SIMD transpose(detail::fmat4x4SIMD const & m)
{
	detail::fmat4x4SIMD result;
	detail::sse_transpose_ps(&m[0].Data, &result[0].Data);
	return result;
}

GLM_FUNC_QUALIFIER float determinant(detail::fmat4x4SIMD const & m)
{
	float Result;
	_mm_store_ss(&Result, detail::sse_det_ps(&m[0].Data));
	return Result;
}

}//namespace glm
