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
/// @file glm/gtx/simd_quat.inl
/// @date 2013-04-22 / 2014-11-25
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace detail{


//////////////////////////////////////
// Debugging
#if 0
void print(__m128 v)
{
    GLM_ALIGN(16) float result[4];
    _mm_store_ps(result, v);

    printf("__m128:    %f %f %f %f\n", result[0], result[1], result[2], result[3]);
}

void print(const fvec4SIMD &v)
{
    printf("fvec4SIMD: %f %f %f %f\n", v.x, v.y, v.z, v.w);
}
#endif


//////////////////////////////////////
// Implicit basic constructors

GLM_FUNC_QUALIFIER fquatSIMD::fquatSIMD()
#	ifdef GLM_FORCE_NO_CTOR_INIT
		: Data(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f))
#	endif
{}

GLM_FUNC_QUALIFIER fquatSIMD::fquatSIMD(__m128 const & Data) :
	Data(Data)
{}

GLM_FUNC_QUALIFIER fquatSIMD::fquatSIMD(fquatSIMD const & q) :
	Data(q.Data)
{}


//////////////////////////////////////
// Explicit basic constructors

GLM_FUNC_QUALIFIER fquatSIMD::fquatSIMD(float const & w, float const & x, float const & y, float const & z) :
	Data(_mm_set_ps(w, z, y, x))
{}

GLM_FUNC_QUALIFIER fquatSIMD::fquatSIMD(quat const & q) :
	Data(_mm_set_ps(q.w, q.z, q.y, q.x))
{}

GLM_FUNC_QUALIFIER fquatSIMD::fquatSIMD(vec3 const & eulerAngles)
{
    vec3 c = glm::cos(eulerAngles * 0.5f);
	vec3 s = glm::sin(eulerAngles * 0.5f);

    Data = _mm_set_ps(
        (c.x * c.y * c.z) + (s.x * s.y * s.z),
        (c.x * c.y * s.z) - (s.x * s.y * c.z),
        (c.x * s.y * c.z) + (s.x * c.y * s.z),
        (s.x * c.y * c.z) - (c.x * s.y * s.z));
}


//////////////////////////////////////
// Unary arithmetic operators

GLM_FUNC_QUALIFIER fquatSIMD& fquatSIMD::operator=(fquatSIMD const & q)
{
    this->Data = q.Data;
    return *this;
}

GLM_FUNC_QUALIFIER fquatSIMD& fquatSIMD::operator*=(float const & s)
{
	this->Data = _mm_mul_ps(this->Data, _mm_set_ps1(s));
	return *this;
}

GLM_FUNC_QUALIFIER fquatSIMD& fquatSIMD::operator/=(float const & s)
{
	this->Data = _mm_div_ps(Data, _mm_set1_ps(s));
	return *this;
}



// negate operator
GLM_FUNC_QUALIFIER fquatSIMD operator- (fquatSIMD const & q)
{
    return fquatSIMD(_mm_mul_ps(q.Data, _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f)));
}

// operator+
GLM_FUNC_QUALIFIER fquatSIMD operator+ (fquatSIMD const & q1, fquatSIMD const & q2)
{
	return fquatSIMD(_mm_add_ps(q1.Data, q2.Data));
}

//operator*
GLM_FUNC_QUALIFIER fquatSIMD operator* (fquatSIMD const & q1, fquatSIMD const & q2)
{
    // SSE2 STATS:
    //    11 shuffle
    //    8  mul
    //    8  add
    
    // SSE4 STATS:
    //    3 shuffle
    //    4 mul
    //    4 dpps

    __m128 mul0 = _mm_mul_ps(q1.Data, _mm_shuffle_ps(q2.Data, q2.Data, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 mul1 = _mm_mul_ps(q1.Data, _mm_shuffle_ps(q2.Data, q2.Data, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 mul2 = _mm_mul_ps(q1.Data, _mm_shuffle_ps(q2.Data, q2.Data, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128 mul3 = _mm_mul_ps(q1.Data, q2.Data);

#   if((GLM_ARCH & GLM_ARCH_SSE4))
    __m128 add0 = _mm_dp_ps(mul0, _mm_set_ps(1.0f, -1.0f,  1.0f,  1.0f), 0xff);
    __m128 add1 = _mm_dp_ps(mul1, _mm_set_ps(1.0f,  1.0f,  1.0f, -1.0f), 0xff);
    __m128 add2 = _mm_dp_ps(mul2, _mm_set_ps(1.0f,  1.0f, -1.0f,  1.0f), 0xff);
    __m128 add3 = _mm_dp_ps(mul3, _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f), 0xff);
#   else
           mul0 = _mm_mul_ps(mul0, _mm_set_ps(1.0f, -1.0f,  1.0f,  1.0f));
    __m128 add0 = _mm_add_ps(mul0, _mm_movehl_ps(mul0, mul0));
           add0 = _mm_add_ss(add0, _mm_shuffle_ps(add0, add0, 1));

           mul1 = _mm_mul_ps(mul1, _mm_set_ps(1.0f,  1.0f,  1.0f, -1.0f));
    __m128 add1 = _mm_add_ps(mul1, _mm_movehl_ps(mul1, mul1));
           add1 = _mm_add_ss(add1, _mm_shuffle_ps(add1, add1, 1));

           mul2 = _mm_mul_ps(mul2, _mm_set_ps(1.0f,  1.0f, -1.0f,  1.0f));
    __m128 add2 = _mm_add_ps(mul2, _mm_movehl_ps(mul2, mul2));
           add2 = _mm_add_ss(add2, _mm_shuffle_ps(add2, add2, 1));

           mul3 = _mm_mul_ps(mul3, _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f));
    __m128 add3 = _mm_add_ps(mul3, _mm_movehl_ps(mul3, mul3));
           add3 = _mm_add_ss(add3, _mm_shuffle_ps(add3, add3, 1));
#endif


    // This SIMD code is a politically correct way of doing this, but in every test I've tried it has been slower than
    // the final code below. I'll keep this here for reference - maybe somebody else can do something better...
    //
    //__m128 xxyy = _mm_shuffle_ps(add0, add1, _MM_SHUFFLE(0, 0, 0, 0));
    //__m128 zzww = _mm_shuffle_ps(add2, add3, _MM_SHUFFLE(0, 0, 0, 0));
    //
    //return _mm_shuffle_ps(xxyy, zzww, _MM_SHUFFLE(2, 0, 2, 0));
    
    float x;
    float y;
    float z;
    float w;

    _mm_store_ss(&x, add0);
    _mm_store_ss(&y, add1);
    _mm_store_ss(&z, add2);
    _mm_store_ss(&w, add3);

    return detail::fquatSIMD(w, x, y, z);
}

GLM_FUNC_QUALIFIER fvec4SIMD operator* (fquatSIMD const & q, fvec4SIMD const & v)
{
    static const __m128 two = _mm_set1_ps(2.0f);

    __m128 q_wwww  = _mm_shuffle_ps(q.Data, q.Data, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 q_swp0  = _mm_shuffle_ps(q.Data, q.Data, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 q_swp1  = _mm_shuffle_ps(q.Data, q.Data, _MM_SHUFFLE(3, 1, 0, 2));
	__m128 v_swp0  = _mm_shuffle_ps(v.Data, v.Data, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 v_swp1  = _mm_shuffle_ps(v.Data, v.Data, _MM_SHUFFLE(3, 1, 0, 2));
	
	__m128 uv      = _mm_sub_ps(_mm_mul_ps(q_swp0, v_swp1), _mm_mul_ps(q_swp1, v_swp0));
    __m128 uv_swp0 = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 uv_swp1 = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 uuv     = _mm_sub_ps(_mm_mul_ps(q_swp0, uv_swp1), _mm_mul_ps(q_swp1, uv_swp0));

    
    uv  = _mm_mul_ps(uv,  _mm_mul_ps(q_wwww, two));
    uuv = _mm_mul_ps(uuv, two);

    return _mm_add_ps(v.Data, _mm_add_ps(uv, uuv));
}

GLM_FUNC_QUALIFIER fvec4SIMD operator* (fvec4SIMD const & v, fquatSIMD const & q)
{
	return glm::inverse(q) * v;
}

GLM_FUNC_QUALIFIER fquatSIMD operator* (fquatSIMD const & q, float s)
{
	return fquatSIMD(_mm_mul_ps(q.Data, _mm_set1_ps(s)));
}

GLM_FUNC_QUALIFIER fquatSIMD operator* (float s, fquatSIMD const & q)
{
	return fquatSIMD(_mm_mul_ps(_mm_set1_ps(s), q.Data));
}


//operator/
GLM_FUNC_QUALIFIER fquatSIMD operator/ (fquatSIMD const & q, float s)
{
	return fquatSIMD(_mm_div_ps(q.Data, _mm_set1_ps(s)));
}


}//namespace detail


GLM_FUNC_QUALIFIER quat quat_cast
(
	detail::fquatSIMD const & x
)
{
	GLM_ALIGN(16) quat Result;
	_mm_store_ps(&Result[0], x.Data);

	return Result;
}

template <typename T>
GLM_FUNC_QUALIFIER detail::fquatSIMD quatSIMD_cast_impl(const T m0[], const T m1[], const T m2[])
{
    T trace = m0[0] + m1[1] + m2[2] + T(1.0);
    if (trace > T(0))
    {
        T s = static_cast<T>(0.5) / sqrt(trace);

        return _mm_set_ps(
            static_cast<float>(T(0.25) / s),
            static_cast<float>((m0[1] - m1[0]) * s),
            static_cast<float>((m2[0] - m0[2]) * s),
            static_cast<float>((m1[2] - m2[1]) * s));
    }
    else
    {
        if (m0[0] > m1[1])
        {
            if (m0[0] > m2[2])
            {
                // X is biggest.
                T s = sqrt(m0[0] - m1[1] - m2[2] + T(1.0)) * T(0.5);

                return _mm_set_ps(
                    static_cast<float>((m1[2] - m2[1]) * s),
                    static_cast<float>((m2[0] + m0[2]) * s),
                    static_cast<float>((m0[1] + m1[0]) * s),
                    static_cast<float>(T(0.5)          * s));
            }
        }
        else
        {
            if (m1[1] > m2[2])
            {
                // Y is biggest.
                T s = sqrt(m1[1] - m0[0] - m2[2] + T(1.0)) * T(0.5);

                return _mm_set_ps(
                    static_cast<float>((m2[0] - m0[2]) * s),
                    static_cast<float>((m1[2] + m2[1]) * s),
                    static_cast<float>(T(0.5)          * s),
                    static_cast<float>((m0[1] + m1[0]) * s));
            }
        }

        // Z is biggest.
        T s = sqrt(m2[2] - m0[0] - m1[1] + T(1.0)) * T(0.5);

        return _mm_set_ps(
            static_cast<float>((m0[1] - m1[0]) * s),
            static_cast<float>(T(0.5)          * s),
            static_cast<float>((m1[2] + m2[1]) * s),
            static_cast<float>((m2[0] + m0[2]) * s));
    }
}

GLM_FUNC_QUALIFIER detail::fquatSIMD quatSIMD_cast
(
	detail::fmat4x4SIMD const & m
)
{
    // Scalar implementation for now.
    GLM_ALIGN(16) float m0[4];
    GLM_ALIGN(16) float m1[4];
    GLM_ALIGN(16) float m2[4];

    _mm_store_ps(m0, m[0].Data);
    _mm_store_ps(m1, m[1].Data);
    _mm_store_ps(m2, m[2].Data);

    return quatSIMD_cast_impl(m0, m1, m2);
}

template <typename T, precision P>
GLM_FUNC_QUALIFIER detail::fquatSIMD quatSIMD_cast
(
    tmat4x4<T, P> const & m
)
{
    return quatSIMD_cast_impl(&m[0][0], &m[1][0], &m[2][0]);
}

template <typename T, precision P>
GLM_FUNC_QUALIFIER detail::fquatSIMD quatSIMD_cast
(
    tmat3x3<T, P> const & m
)
{
    return quatSIMD_cast_impl(&m[0][0], &m[1][0], &m[2][0]);
}


GLM_FUNC_QUALIFIER detail::fmat4x4SIMD mat4SIMD_cast
(
	detail::fquatSIMD const & q
)
{
    detail::fmat4x4SIMD result;

    __m128 _wwww  = _mm_shuffle_ps(q.Data, q.Data, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 _xyzw  = q.Data;
    __m128 _zxyw  = _mm_shuffle_ps(q.Data, q.Data, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 _yzxw  = _mm_shuffle_ps(q.Data, q.Data, _MM_SHUFFLE(3, 0, 2, 1));

    __m128 _xyzw2 = _mm_add_ps(_xyzw, _xyzw);
    __m128 _zxyw2 = _mm_shuffle_ps(_xyzw2, _xyzw2, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 _yzxw2 = _mm_shuffle_ps(_xyzw2, _xyzw2, _MM_SHUFFLE(3, 0, 2, 1));
    
    __m128 _tmp0  = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(_yzxw2, _yzxw));
           _tmp0  = _mm_sub_ps(_tmp0, _mm_mul_ps(_zxyw2, _zxyw));

    __m128 _tmp1  = _mm_mul_ps(_yzxw2, _xyzw);
           _tmp1  = _mm_add_ps(_tmp1, _mm_mul_ps(_zxyw2, _wwww));

    __m128 _tmp2  = _mm_mul_ps(_zxyw2, _xyzw);
           _tmp2  = _mm_sub_ps(_tmp2, _mm_mul_ps(_yzxw2, _wwww));


    // There's probably a better, more politically correct way of doing this...
    result[0].Data = _mm_set_ps(
        0.0f,
        reinterpret_cast<float*>(&_tmp2)[0],
        reinterpret_cast<float*>(&_tmp1)[0],
        reinterpret_cast<float*>(&_tmp0)[0]);

    result[1].Data = _mm_set_ps(
        0.0f,
        reinterpret_cast<float*>(&_tmp1)[1],
        reinterpret_cast<float*>(&_tmp0)[1],
        reinterpret_cast<float*>(&_tmp2)[1]);

    result[2].Data = _mm_set_ps(
        0.0f,
        reinterpret_cast<float*>(&_tmp0)[2],
        reinterpret_cast<float*>(&_tmp2)[2],
        reinterpret_cast<float*>(&_tmp1)[2]);

   result[3].Data = _mm_set_ps(
        1.0f,
        0.0f,
        0.0f,
        0.0f);


    return result;
}

GLM_FUNC_QUALIFIER mat4 mat4_cast
(
	detail::fquatSIMD const & q
)
{
    return mat4_cast(mat4SIMD_cast(q));
}



GLM_FUNC_QUALIFIER float length
(
	detail::fquatSIMD const & q
)
{
    return glm::sqrt(dot(q, q));
}

GLM_FUNC_QUALIFIER detail::fquatSIMD normalize
(
	detail::fquatSIMD const & q
)
{
    return _mm_mul_ps(q.Data, _mm_set1_ps(1.0f / length(q)));
}

GLM_FUNC_QUALIFIER float dot
(
	detail::fquatSIMD const & q1,
	detail::fquatSIMD const & q2
)
{
    float result;
    _mm_store_ss(&result, detail::sse_dot_ps(q1.Data, q2.Data));

    return result;
}

GLM_FUNC_QUALIFIER detail::fquatSIMD mix
(
	detail::fquatSIMD const & x, 
	detail::fquatSIMD const & y, 
	float const & a
)
{
	float cosTheta = dot(x, y);

    if (cosTheta > 1.0f - glm::epsilon<float>())
    {
	    return _mm_add_ps(x.Data, _mm_mul_ps(_mm_set1_ps(a), _mm_sub_ps(y.Data, x.Data)));
    }
    else
    {
        float angle = glm::acos(cosTheta);
        
        
        float s0 = glm::sin((1.0f - a) * angle);
        float s1 = glm::sin(a * angle);
        float d  = 1.0f / glm::sin(angle);

        return (s0 * x + s1 * y) * d;
    }
}

GLM_FUNC_QUALIFIER detail::fquatSIMD lerp
(
	detail::fquatSIMD const & x, 
	detail::fquatSIMD const & y, 
	float const & a
)
{
	// Lerp is only defined in [0, 1]
	assert(a >= 0.0f);
	assert(a <= 1.0f);

    return _mm_add_ps(x.Data, _mm_mul_ps(_mm_set1_ps(a), _mm_sub_ps(y.Data, x.Data)));
}

GLM_FUNC_QUALIFIER detail::fquatSIMD slerp
(
	detail::fquatSIMD const & x, 
	detail::fquatSIMD const & y, 
	float const & a
)
{
	detail::fquatSIMD z = y;

	float cosTheta = dot(x, y);

	// If cosTheta < 0, the interpolation will take the long way around the sphere. 
	// To fix this, one quat must be negated.
	if (cosTheta < 0.0f)
	{
		z        = -y;
		cosTheta = -cosTheta;
	}

	// Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
	if(cosTheta > 1.0f - epsilon<float>())
	{
		return _mm_add_ps(x.Data, _mm_mul_ps(_mm_set1_ps(a), _mm_sub_ps(y.Data, x.Data)));
	}
	else
	{
        float angle = glm::acos(cosTheta);


		float s0 = glm::sin((1.0f - a) * angle);
        float s1 = glm::sin(a * angle);
        float d  = 1.0f / glm::sin(angle);

        return (s0 * x + s1 * y) * d;
	}
}


GLM_FUNC_QUALIFIER detail::fquatSIMD fastMix
(
	detail::fquatSIMD const & x, 
	detail::fquatSIMD const & y, 
	float const & a
)
{
	float cosTheta = dot(x, y);

    if (cosTheta > 1.0f - glm::epsilon<float>())
    {
	    return _mm_add_ps(x.Data, _mm_mul_ps(_mm_set1_ps(a), _mm_sub_ps(y.Data, x.Data)));
    }
    else
    {
        float angle = glm::fastAcos(cosTheta);


        __m128 s  = glm::fastSin(_mm_set_ps((1.0f - a) * angle, a * angle, angle, 0.0f));

        __m128 s0 =                               _mm_shuffle_ps(s, s, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 s1 =                               _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 d  = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1)));
        
        return _mm_mul_ps(_mm_add_ps(_mm_mul_ps(s0, x.Data), _mm_mul_ps(s1, y.Data)), d);
    }
}

GLM_FUNC_QUALIFIER detail::fquatSIMD fastSlerp
(
	detail::fquatSIMD const & x, 
	detail::fquatSIMD const & y, 
	float const & a
)
{
	detail::fquatSIMD z = y;

	float cosTheta = dot(x, y);
	if (cosTheta < 0.0f)
	{
		z        = -y;
		cosTheta = -cosTheta;
	}


	if(cosTheta > 1.0f - epsilon<float>())
	{
		return _mm_add_ps(x.Data, _mm_mul_ps(_mm_set1_ps(a), _mm_sub_ps(y.Data, x.Data)));
	}
	else
	{
        float angle = glm::fastAcos(cosTheta);


        __m128 s  = glm::fastSin(_mm_set_ps((1.0f - a) * angle, a * angle, angle, 0.0f));

        __m128 s0 =                               _mm_shuffle_ps(s, s, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 s1 =                               _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 d  = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1)));
        
        return _mm_mul_ps(_mm_add_ps(_mm_mul_ps(s0, x.Data), _mm_mul_ps(s1, y.Data)), d);
	}
}



GLM_FUNC_QUALIFIER detail::fquatSIMD conjugate
(
	detail::fquatSIMD const & q
)
{
	return detail::fquatSIMD(_mm_mul_ps(q.Data, _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f)));
}

GLM_FUNC_QUALIFIER detail::fquatSIMD inverse
(
	detail::fquatSIMD const & q
)
{
	return conjugate(q) / dot(q, q);
}


GLM_FUNC_QUALIFIER detail::fquatSIMD angleAxisSIMD
(
	float const & angle,
	vec3 const & v
)
{
	float s = glm::sin(angle * 0.5f);

	return _mm_set_ps(
		glm::cos(angle * 0.5f),
		v.z * s,
		v.y * s,
		v.x * s);
}

GLM_FUNC_QUALIFIER detail::fquatSIMD angleAxisSIMD
(
	float const & angle, 
	float const & x, 
	float const & y, 
	float const & z
)
{
	return angleAxisSIMD(angle, vec3(x, y, z));
}


GLM_FUNC_QUALIFIER __m128 fastSin(__m128 x)
{
	static const __m128 c0 = _mm_set1_ps(0.16666666666666666666666666666667f);
	static const __m128 c1 = _mm_set1_ps(0.00833333333333333333333333333333f);
	static const __m128 c2 = _mm_set1_ps(0.00019841269841269841269841269841f);

	__m128 x3 = _mm_mul_ps(x,  _mm_mul_ps(x, x));
	__m128 x5 = _mm_mul_ps(x3, _mm_mul_ps(x, x));
	__m128 x7 = _mm_mul_ps(x5, _mm_mul_ps(x, x));

	__m128 y0 = _mm_mul_ps(x3, c0);
	__m128 y1 = _mm_mul_ps(x5, c1);
	__m128 y2 = _mm_mul_ps(x7, c2);

	return _mm_sub_ps(_mm_add_ps(_mm_sub_ps(x, y0), y1), y2);
}


}//namespace glm
