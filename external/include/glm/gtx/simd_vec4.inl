///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2014 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-07
// Updated : 2009-05-07
// Licence : This source is under MIT License
// File    : glm/gtx/simd_vec4.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{

template <int Value>
struct shuffle_mask
{
	enum{value = Value};
};

//////////////////////////////////////
// Implicit basic constructors

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD()
#	ifdef GLM_FORCE_NO_CTOR_INIT
		: Data(_mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f))
#	endif
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(__m128 const & Data) :
	Data(Data)
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(fvec4SIMD const & v) :
	Data(v.Data)
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(vec4 const & v) :
	Data(_mm_set_ps(v.w, v.z, v.y, v.x))
{}

//////////////////////////////////////
// Explicit basic constructors

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(float const & s) :
	Data(_mm_set1_ps(s))
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(float const & x, float const & y, float const & z, float const & w) :
//		Data(_mm_setr_ps(x, y, z, w))
	Data(_mm_set_ps(w, z, y, x))
{}
/*
GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(float const v[4]) :
	Data(_mm_load_ps(v))
{}
*/
//////////////////////////////////////
// Swizzle constructors

//fvec4SIMD(ref4<float> const & r);

//////////////////////////////////////
// Conversion vector constructors

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(vec2 const & v, float const & s1, float const & s2) :
	Data(_mm_set_ps(s2, s1, v.y, v.x))
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(float const & s1, vec2 const & v, float const & s2) :
	Data(_mm_set_ps(s2, v.y, v.x, s1))
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(float const & s1, float const & s2, vec2 const & v) :
	Data(_mm_set_ps(v.y, v.x, s2, s1))
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(vec3 const & v, float const & s) :
	Data(_mm_set_ps(s, v.z, v.y, v.x))
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(float const & s, vec3 const & v) :
	Data(_mm_set_ps(v.z, v.y, v.x, s))
{}

GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(vec2 const & v1, vec2 const & v2) :
	Data(_mm_set_ps(v2.y, v2.x, v1.y, v1.x))
{}

//GLM_FUNC_QUALIFIER fvec4SIMD::fvec4SIMD(ivec4SIMD const & v) :
//	Data(_mm_cvtepi32_ps(v.Data))
//{}

//////////////////////////////////////
// Unary arithmetic operators

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator=(fvec4SIMD const & v)
{
	this->Data = v.Data;
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator+=(float const & s)
{
	this->Data = _mm_add_ps(Data, _mm_set_ps1(s));
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator+=(fvec4SIMD const & v)
{
	this->Data = _mm_add_ps(this->Data , v.Data);
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator-=(float const & s)
{
	this->Data = _mm_sub_ps(Data, _mm_set_ps1(s));
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator-=(fvec4SIMD const & v)
{
	this->Data = _mm_sub_ps(this->Data , v.Data);
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator*=(float const & s)
{
	this->Data = _mm_mul_ps(this->Data, _mm_set_ps1(s));
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator*=(fvec4SIMD const & v)
{
	this->Data = _mm_mul_ps(this->Data , v.Data);
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator/=(float const & s)
{
	this->Data = _mm_div_ps(Data, _mm_set1_ps(s));
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator/=(fvec4SIMD const & v)
{
	this->Data = _mm_div_ps(this->Data , v.Data);
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator++()
{
	this->Data = _mm_add_ps(this->Data , glm::detail::one);
	return *this;
}

GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::operator--()
{
	this->Data = _mm_sub_ps(this->Data, glm::detail::one);
	return *this;
}

//////////////////////////////////////
// Swizzle operators

template <comp X, comp Y, comp Z, comp W>
GLM_FUNC_QUALIFIER fvec4SIMD fvec4SIMD::swizzle() const
{
	__m128 Data = _mm_shuffle_ps(
		this->Data, this->Data, 
		shuffle_mask<(W << 6) | (Z << 4) | (Y << 2) | (X << 0)>::value);
	return fvec4SIMD(Data);
}

template <comp X, comp Y, comp Z, comp W>
GLM_FUNC_QUALIFIER fvec4SIMD& fvec4SIMD::swizzle()
{
	this->Data = _mm_shuffle_ps(
		this->Data, this->Data, 
		shuffle_mask<(W << 6) | (Z << 4) | (Y << 2) | (X << 0)>::value);
	return *this;
}

// operator+
GLM_FUNC_QUALIFIER fvec4SIMD operator+ (fvec4SIMD const & v, float s)
{
	return fvec4SIMD(_mm_add_ps(v.Data, _mm_set1_ps(s)));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator+ (float s, fvec4SIMD const & v)
{
	return fvec4SIMD(_mm_add_ps(_mm_set1_ps(s), v.Data));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator+ (fvec4SIMD const & v1, fvec4SIMD const & v2)
{
	return fvec4SIMD(_mm_add_ps(v1.Data, v2.Data));
}

//operator-
GLM_FUNC_QUALIFIER fvec4SIMD operator- (fvec4SIMD const & v, float s)
{
	return fvec4SIMD(_mm_sub_ps(v.Data, _mm_set1_ps(s)));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator- (float s, fvec4SIMD const & v)
{
	return fvec4SIMD(_mm_sub_ps(_mm_set1_ps(s), v.Data));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator- (fvec4SIMD const & v1, fvec4SIMD const & v2)
{
	return fvec4SIMD(_mm_sub_ps(v1.Data, v2.Data));
}

//operator*
GLM_FUNC_QUALIFIER fvec4SIMD operator* (fvec4SIMD const & v, float s)
{
	__m128 par0 = v.Data;
	__m128 par1 = _mm_set1_ps(s);
	return fvec4SIMD(_mm_mul_ps(par0, par1));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator* (float s, fvec4SIMD const & v)
{
	__m128 par0 = _mm_set1_ps(s);
	__m128 par1 = v.Data;
	return fvec4SIMD(_mm_mul_ps(par0, par1));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator* (fvec4SIMD const & v1, fvec4SIMD const & v2)
{
	return fvec4SIMD(_mm_mul_ps(v1.Data, v2.Data));
}

//operator/
GLM_FUNC_QUALIFIER fvec4SIMD operator/ (fvec4SIMD const & v, float s)
{
	__m128 par0 = v.Data;
	__m128 par1 = _mm_set1_ps(s);
	return fvec4SIMD(_mm_div_ps(par0, par1));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator/ (float s, fvec4SIMD const & v)
{
	__m128 par0 = _mm_set1_ps(s);
	__m128 par1 = v.Data;
	return fvec4SIMD(_mm_div_ps(par0, par1));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator/ (fvec4SIMD const & v1, fvec4SIMD const & v2)
{
	return fvec4SIMD(_mm_div_ps(v1.Data, v2.Data));
}

// Unary constant operators
GLM_FUNC_QUALIFIER fvec4SIMD operator- (fvec4SIMD const & v)
{
	return fvec4SIMD(_mm_sub_ps(_mm_setzero_ps(), v.Data));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator++ (fvec4SIMD const & v, int)
{
	return fvec4SIMD(_mm_add_ps(v.Data, glm::detail::one));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator-- (fvec4SIMD const & v, int)
{
	return fvec4SIMD(_mm_sub_ps(v.Data, glm::detail::one));
}

}//namespace detail

GLM_FUNC_QUALIFIER vec4 vec4_cast
(
	detail::fvec4SIMD const & x
)
{
	GLM_ALIGN(16) vec4 Result;
	_mm_store_ps(&Result[0], x.Data);
	return Result;
}

// Other possible implementation
//float abs(float a)
//{
//  return max(-a, a);
//}
GLM_FUNC_QUALIFIER detail::fvec4SIMD abs
(
	detail::fvec4SIMD const & x
)
{
	return detail::sse_abs_ps(x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD sign
(
	detail::fvec4SIMD const & x
)
{
	return detail::sse_sgn_ps(x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD floor
(
	detail::fvec4SIMD const & x
)
{
	return detail::sse_flr_ps(x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD trunc
(
	detail::fvec4SIMD const & x
)
{
    //return x < 0 ? -floor(-x) : floor(x);

	__m128 Flr0 = detail::sse_flr_ps(_mm_sub_ps(_mm_setzero_ps(), x.Data));
	__m128 Sub0 = _mm_sub_ps(Flr0, x.Data);
	__m128 Flr1 = detail::sse_flr_ps(x.Data);

	__m128 Cmp0 = _mm_cmplt_ps(x.Data, glm::detail::zero);
	__m128 Cmp1 = _mm_cmpnlt_ps(x.Data, glm::detail::zero);

	__m128 And0 = _mm_and_ps(Sub0, Cmp0);
	__m128 And1 = _mm_and_ps(Flr1, Cmp1);

	return _mm_or_ps(And0, And1);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD round
(
	detail::fvec4SIMD const & x
)
{
	return detail::sse_rnd_ps(x.Data);
}

//GLM_FUNC_QUALIFIER detail::fvec4SIMD roundEven
//(
//	detail::fvec4SIMD const & x
//)
//{

//}

GLM_FUNC_QUALIFIER detail::fvec4SIMD ceil
(
	detail::fvec4SIMD const & x
)
{
	return detail::sse_ceil_ps(x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD fract
(
	detail::fvec4SIMD const & x
)
{
	return detail::sse_frc_ps(x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD mod
(
	detail::fvec4SIMD const & x, 
	detail::fvec4SIMD const & y
)
{
	return detail::sse_mod_ps(x.Data, y.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD mod
(
	detail::fvec4SIMD const & x, 
	float const & y
)
{
	return detail::sse_mod_ps(x.Data, _mm_set1_ps(y));
}

//GLM_FUNC_QUALIFIER detail::fvec4SIMD modf
//(
//	detail::fvec4SIMD const & x, 
//	detail::fvec4SIMD & i
//)
//{

//}

GLM_FUNC_QUALIFIER detail::fvec4SIMD min
(
	detail::fvec4SIMD const & x, 
	detail::fvec4SIMD const & y
)
{
	return _mm_min_ps(x.Data, y.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD min
(
	detail::fvec4SIMD const & x, 
	float const & y
)
{
	return _mm_min_ps(x.Data, _mm_set1_ps(y));
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD max
(
	detail::fvec4SIMD const & x, 
	detail::fvec4SIMD const & y
)
{
	return _mm_max_ps(x.Data, y.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD max
(
	detail::fvec4SIMD const & x, 
	float const & y
)
{
	return _mm_max_ps(x.Data, _mm_set1_ps(y));
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD clamp
(
	detail::fvec4SIMD const & x, 
	detail::fvec4SIMD const & minVal, 
	detail::fvec4SIMD const & maxVal
)
{
	return detail::sse_clp_ps(x.Data, minVal.Data, maxVal.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD clamp
(
	detail::fvec4SIMD const & x, 
	float const & minVal, 
	float const & maxVal
) 
{
	return detail::sse_clp_ps(x.Data, _mm_set1_ps(minVal), _mm_set1_ps(maxVal));
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD mix
(
	detail::fvec4SIMD const & x, 
	detail::fvec4SIMD const & y, 
	detail::fvec4SIMD const & a
)
{
	__m128 Sub0 = _mm_sub_ps(y.Data, x.Data);
	__m128 Mul0 = _mm_mul_ps(a.Data, Sub0);
	return _mm_add_ps(x.Data, Mul0);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD step
(
	detail::fvec4SIMD const & edge, 
	detail::fvec4SIMD const & x
)
{
	__m128 cmp0 = _mm_cmpngt_ps(x.Data, edge.Data);
	return _mm_max_ps(_mm_min_ps(cmp0, _mm_setzero_ps()), detail::one);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD step
(
	float const & edge, 
	detail::fvec4SIMD const & x
)
{
	__m128 cmp0 = _mm_cmpngt_ps(x.Data, _mm_set1_ps(edge));
	return _mm_max_ps(_mm_min_ps(cmp0, _mm_setzero_ps()), detail::one);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD smoothstep
(
	detail::fvec4SIMD const & edge0, 
	detail::fvec4SIMD const & edge1, 
	detail::fvec4SIMD const & x
)
{
	return detail::sse_ssp_ps(edge0.Data, edge1.Data, x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD smoothstep
(
	float const & edge0, 
	float const & edge1, 
	detail::fvec4SIMD const & x
)
{
	return detail::sse_ssp_ps(_mm_set1_ps(edge0), _mm_set1_ps(edge1), x.Data);
}

//GLM_FUNC_QUALIFIER bvec4 isnan(detail::fvec4SIMD const & x)
//{

//}

//GLM_FUNC_QUALIFIER bvec4 isinf(detail::fvec4SIMD const & x)
//{

//}

//GLM_FUNC_QUALIFIER detail::ivec4SIMD floatBitsToInt
//(
//	detail::fvec4SIMD const & value
//)
//{

//}

//GLM_FUNC_QUALIFIER detail::fvec4SIMD intBitsToFloat
//(
//	detail::ivec4SIMD const & value
//)
//{

//}

GLM_FUNC_QUALIFIER detail::fvec4SIMD fma
(
	detail::fvec4SIMD const & a, 
	detail::fvec4SIMD const & b, 
	detail::fvec4SIMD const & c
)
{
	return _mm_add_ps(_mm_mul_ps(a.Data, b.Data), c.Data);
}

GLM_FUNC_QUALIFIER float length
(
	detail::fvec4SIMD const & x
)
{
	detail::fvec4SIMD dot0 = detail::sse_dot_ss(x.Data, x.Data);
	detail::fvec4SIMD sqt0 = sqrt(dot0);
	float Result = 0;
	_mm_store_ss(&Result, sqt0.Data);
	return Result;
}

GLM_FUNC_QUALIFIER float fastLength
(
	detail::fvec4SIMD const & x
)
{
	detail::fvec4SIMD dot0 = detail::sse_dot_ss(x.Data, x.Data);
	detail::fvec4SIMD sqt0 = fastSqrt(dot0);
	float Result = 0;
	_mm_store_ss(&Result, sqt0.Data);
	return Result;
}

GLM_FUNC_QUALIFIER float niceLength
(
	detail::fvec4SIMD const & x
)
{
	detail::fvec4SIMD dot0 = detail::sse_dot_ss(x.Data, x.Data);
	detail::fvec4SIMD sqt0 = niceSqrt(dot0);
	float Result = 0;
	_mm_store_ss(&Result, sqt0.Data);
	return Result;
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD length4
(
	detail::fvec4SIMD const & x
)
{
	return sqrt(dot4(x, x));
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD fastLength4
(
	detail::fvec4SIMD const & x
)
{
	return fastSqrt(dot4(x, x));
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD niceLength4
(
	detail::fvec4SIMD const & x
)
{
	return niceSqrt(dot4(x, x));
}

GLM_FUNC_QUALIFIER float distance
(
	detail::fvec4SIMD const & p0,
	detail::fvec4SIMD const & p1
)
{
	float Result = 0;
	_mm_store_ss(&Result, detail::sse_dst_ps(p0.Data, p1.Data));
	return Result;
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD distance4
(
	detail::fvec4SIMD const & p0,
	detail::fvec4SIMD const & p1
)
{
	return detail::sse_dst_ps(p0.Data, p1.Data);
}

GLM_FUNC_QUALIFIER float dot
(
	detail::fvec4SIMD const & x,
	detail::fvec4SIMD const & y
)
{
	float Result = 0;
	_mm_store_ss(&Result, detail::sse_dot_ss(x.Data, y.Data));
	return Result;
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD dot4
(
	detail::fvec4SIMD const & x,
	detail::fvec4SIMD const & y
)
{
	return detail::sse_dot_ps(x.Data, y.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD cross
(
	detail::fvec4SIMD const & x,
	detail::fvec4SIMD const & y
)
{
	return detail::sse_xpd_ps(x.Data, y.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD normalize
(
	detail::fvec4SIMD const & x
)
{
	__m128 dot0 = detail::sse_dot_ps(x.Data, x.Data);
	__m128 isr0 = inversesqrt(detail::fvec4SIMD(dot0)).Data;
	__m128 mul0 = _mm_mul_ps(x.Data, isr0);
	return mul0;
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD fastNormalize
(
	detail::fvec4SIMD const & x
)
{
	__m128 dot0 = detail::sse_dot_ps(x.Data, x.Data);
	__m128 isr0 = fastInversesqrt(dot0).Data;
	__m128 mul0 = _mm_mul_ps(x.Data, isr0);
	return mul0;
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD faceforward
(
	detail::fvec4SIMD const & N,
	detail::fvec4SIMD const & I,
	detail::fvec4SIMD const & Nref
)
{
	return detail::sse_ffd_ps(N.Data, I.Data, Nref.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD reflect
(
	detail::fvec4SIMD const & I,
	detail::fvec4SIMD const & N
)
{
	return detail::sse_rfe_ps(I.Data, N.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD refract
(
	detail::fvec4SIMD const & I,
	detail::fvec4SIMD const & N,
	float const & eta
)
{
	return detail::sse_rfa_ps(I.Data, N.Data, _mm_set1_ps(eta));
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD sqrt(detail::fvec4SIMD const & x)
{
	return _mm_mul_ps(inversesqrt(x).Data, x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD niceSqrt(detail::fvec4SIMD const & x)
{
	return _mm_sqrt_ps(x.Data);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD fastSqrt(detail::fvec4SIMD const & x)
{
	return _mm_mul_ps(fastInversesqrt(x.Data).Data, x.Data);
}

// SSE scalar reciprocal sqrt using rsqrt op, plus one Newton-Rhaphson iteration
// By Elan Ruskin, http://assemblyrequired.crashworks.org/
GLM_FUNC_QUALIFIER detail::fvec4SIMD inversesqrt(detail::fvec4SIMD const & x)
{
	GLM_ALIGN(4) static const __m128 three = {3, 3, 3, 3}; // aligned consts for fast load
	GLM_ALIGN(4) static const __m128 half = {0.5,0.5,0.5,0.5};

	__m128 recip = _mm_rsqrt_ps(x.Data);  // "estimate" opcode
	__m128 halfrecip = _mm_mul_ps(half, recip);
	__m128 threeminus_xrr = _mm_sub_ps(three, _mm_mul_ps(x.Data, _mm_mul_ps(recip, recip)));
	return _mm_mul_ps(halfrecip, threeminus_xrr);
}

GLM_FUNC_QUALIFIER detail::fvec4SIMD fastInversesqrt(detail::fvec4SIMD const & x)
{
	return _mm_rsqrt_ps(x.Data);
}

}//namespace glm
