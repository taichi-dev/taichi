///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

/// @file math/FiniteDifference.h

#ifndef OPENVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include "Math.h"
#include "Coord.h"
#include "Vec3.h"
#include <string>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>

#ifdef DWA_OPENVDB
#include <simd/Simd.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////


/// @brief Different discrete schemes used in the first derivatives.
// Add new items to the *end* of this list, and update NUM_DS_SCHEMES.
enum DScheme {
    UNKNOWN_DS = -1,
    CD_2NDT =  0,   // center difference,    2nd order, but the result must be divided by 2
    CD_2ND,         // center difference,    2nd order
    CD_4TH,         // center difference,    4th order
    CD_6TH,         // center difference,    6th order
    FD_1ST,         // forward difference,   1st order
    FD_2ND,         // forward difference,   2nd order
    FD_3RD,         // forward difference,   3rd order
    BD_1ST,         // backward difference,  1st order
    BD_2ND,         // backward difference,  2nd order
    BD_3RD,         // backward difference,  3rd order
    FD_WENO5,       // forward difference,   weno5
    BD_WENO5,       // backward difference,  weno5
    FD_HJWENO5,     // forward differene,   HJ-weno5
    BD_HJWENO5      // backward difference, HJ-weno5
};

enum { NUM_DS_SCHEMES = BD_HJWENO5 + 1 };


inline std::string
dsSchemeToString(DScheme dss)
{
    std::string ret;
    switch (dss) {
        case UNKNOWN_DS:    ret = "unknown_ds"; break;
        case CD_2NDT:       ret = "cd_2ndt";    break;
        case CD_2ND:        ret = "cd_2nd";     break;
        case CD_4TH:        ret = "cd_4th";     break;
        case CD_6TH:        ret = "cd_6th";     break;
        case FD_1ST:        ret = "fd_1st";     break;
        case FD_2ND:        ret = "fd_2nd";     break;
        case FD_3RD:        ret = "fd_3rd";     break;
        case BD_1ST:        ret = "bd_1st";     break;
        case BD_2ND:        ret = "bd_2nd";     break;
        case BD_3RD:        ret = "bd_3rd";     break;
        case FD_WENO5:      ret = "fd_weno5";   break;
        case BD_WENO5:      ret = "bd_weno5";   break;
        case FD_HJWENO5:    ret = "fd_hjweno5"; break;
        case BD_HJWENO5:    ret = "bd_hjweno5"; break;
    }
    return ret;
}

inline DScheme
stringToDScheme(const std::string& s)
{
    DScheme ret = UNKNOWN_DS;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == dsSchemeToString(CD_2NDT)) {
        ret = CD_2NDT;
    } else if (str == dsSchemeToString(CD_2ND)) {
        ret = CD_2ND;
    } else if (str == dsSchemeToString(CD_4TH)) {
        ret = CD_4TH;
    } else if (str == dsSchemeToString(CD_6TH)) {
        ret = CD_6TH;
    } else if (str == dsSchemeToString(FD_1ST)) {
        ret = FD_1ST;
    } else if (str == dsSchemeToString(FD_2ND)) {
        ret = FD_2ND;
    } else if (str == dsSchemeToString(FD_3RD)) {
        ret = FD_3RD;
    } else if (str == dsSchemeToString(BD_1ST)) {
        ret = BD_1ST;
    } else if (str == dsSchemeToString(BD_2ND)) {
        ret = BD_2ND;
    } else if (str == dsSchemeToString(BD_3RD)) {
        ret = BD_3RD;
    } else if (str == dsSchemeToString(FD_WENO5)) {
        ret = FD_WENO5;
    } else if (str == dsSchemeToString(BD_WENO5)) {
        ret = BD_WENO5;
    } else if (str == dsSchemeToString(FD_HJWENO5)) {
        ret = FD_HJWENO5;
    } else if (str == dsSchemeToString(BD_HJWENO5)) {
        ret = BD_HJWENO5;
    }

    return ret;
}

inline std::string
dsSchemeToMenuName(DScheme dss)
{
    std::string ret;
    switch (dss) {
        case UNKNOWN_DS:    ret = "Unknown DS scheme";                      break;
        case CD_2NDT:       ret = "Twice 2nd-order center difference";      break;
        case CD_2ND:        ret = "2nd-order center difference";            break;
        case CD_4TH:        ret = "4th-order center difference";            break;
        case CD_6TH:        ret = "6th-order center difference";            break;
        case FD_1ST:        ret = "1st-order forward difference";           break;
        case FD_2ND:        ret = "2nd-order forward difference";           break;
        case FD_3RD:        ret = "3rd-order forward difference";           break;
        case BD_1ST:        ret = "1st-order backward difference";          break;
        case BD_2ND:        ret = "2nd-order backward difference";          break;
        case BD_3RD:        ret = "3rd-order backward difference";          break;
        case FD_WENO5:      ret = "5th-order WENO forward difference";      break;
        case BD_WENO5:      ret = "5th-order WENO backward difference";     break;
        case FD_HJWENO5:    ret = "5th-order HJ-WENO forward difference";   break;
        case BD_HJWENO5:    ret = "5th-order HJ-WENO backward difference";  break;
    }
    return ret;
}



////////////////////////////////////////


/// @brief Different discrete schemes used in the second derivatives.
// Add new items to the *end* of this list, and update NUM_DD_SCHEMES.
enum DDScheme {
    UNKNOWN_DD  = -1,
    CD_SECOND   =  0,   // center difference, 2nd order
    CD_FOURTH,          // center difference, 4th order
    CD_SIXTH            // center difference, 6th order
};

enum { NUM_DD_SCHEMES = CD_SIXTH + 1 };


////////////////////////////////////////


/// @brief Biased Gradients are limited to non-centered differences
// Add new items to the *end* of this list, and update NUM_BIAS_SCHEMES.
enum BiasedGradientScheme {
    UNKNOWN_BIAS    = -1,
    FIRST_BIAS      = 0,    // uses FD_1ST & BD_1ST
    SECOND_BIAS,            // uses FD_2ND & BD_2ND
    THIRD_BIAS,             // uses FD_3RD & BD_3RD
    WENO5_BIAS,             // uses WENO5
    HJWENO5_BIAS            // uses HJWENO5
};

enum { NUM_BIAS_SCHEMES = HJWENO5_BIAS + 1 };

inline std::string
biasedGradientSchemeToString(BiasedGradientScheme bgs)
{
    std::string ret;
    switch (bgs) {
        case UNKNOWN_BIAS:  ret = "unknown_bias";   break;
        case FIRST_BIAS:    ret = "first_bias";     break;
        case SECOND_BIAS:   ret = "second_bias";    break;
        case THIRD_BIAS:    ret = "third_bias";     break;
        case WENO5_BIAS:    ret = "weno5_bias";     break;
        case HJWENO5_BIAS:  ret = "hjweno5_bias";   break;
    }
    return ret;
}

inline BiasedGradientScheme
stringToBiasedGradientScheme(const std::string& s)
{
    BiasedGradientScheme ret = UNKNOWN_BIAS;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == biasedGradientSchemeToString(FIRST_BIAS)) {
        ret = FIRST_BIAS;
    } else if (str == biasedGradientSchemeToString(SECOND_BIAS)) {
        ret = SECOND_BIAS;
    } else if (str == biasedGradientSchemeToString(THIRD_BIAS)) {
        ret = THIRD_BIAS;
    } else if (str == biasedGradientSchemeToString(WENO5_BIAS)) {
        ret = WENO5_BIAS;
    } else if (str == biasedGradientSchemeToString(HJWENO5_BIAS)) {
        ret = HJWENO5_BIAS;
    }
    return ret;
}

inline std::string
biasedGradientSchemeToMenuName(BiasedGradientScheme bgs)
{
    std::string ret;
    switch (bgs) {
        case UNKNOWN_BIAS:  ret = "Unknown biased gradient";            break;
        case FIRST_BIAS:    ret = "1st-order biased gradient";          break;
        case SECOND_BIAS:   ret = "2nd-order biased gradient";          break;
        case THIRD_BIAS:    ret = "3rd-order biased gradient";          break;
        case WENO5_BIAS:    ret = "5th-order WENO biased gradient";     break;
        case HJWENO5_BIAS:  ret = "5th-order HJ-WENO biased gradient";  break;
    }
    return ret;
}

////////////////////////////////////////


/// @brief Temporal integration schemes
// Add new items to the *end* of this list, and update NUM_TEMPORAL_SCHEMES.
enum TemporalIntegrationScheme {
    UNKNOWN_TIS = -1,
    TVD_RK1,//same as explicit Euler integration
    TVD_RK2,
    TVD_RK3
};

enum { NUM_TEMPORAL_SCHEMES = TVD_RK3 + 1 };

inline std::string
temporalIntegrationSchemeToString(TemporalIntegrationScheme tis)
{
    std::string ret;
    switch (tis) {
        case UNKNOWN_TIS:   ret = "unknown_tis";    break;
        case TVD_RK1:       ret = "tvd_rk1";        break;
        case TVD_RK2:       ret = "tvd_rk2";        break;
        case TVD_RK3:       ret = "tvd_rk3";        break;
    }
    return ret;
}

inline TemporalIntegrationScheme
stringToTemporalIntegrationScheme(const std::string& s)
{
    TemporalIntegrationScheme ret = UNKNOWN_TIS;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == temporalIntegrationSchemeToString(TVD_RK1)) {
        ret = TVD_RK1;
    } else if (str == temporalIntegrationSchemeToString(TVD_RK2)) {
        ret = TVD_RK2;
    } else if (str == temporalIntegrationSchemeToString(TVD_RK3)) {
        ret = TVD_RK3;
    }

    return ret;
}

inline std::string
temporalIntegrationSchemeToMenuName(TemporalIntegrationScheme tis)
{
    std::string ret;
    switch (tis) {
        case UNKNOWN_TIS:   ret = "Unknown temporal integration";   break;
        case TVD_RK1:       ret = "Forward Euler";                  break;
        case TVD_RK2:       ret = "2nd-order Runge-Kutta";          break;
        case TVD_RK3:       ret = "3rd-order Runge-Kutta";          break;
    }
    return ret;
}


//@}


/// @brief Implementation of nominally fifth-order finite-difference WENO
/// @details This function returns the numerical flux.  See "High Order Finite Difference and
/// Finite Volume WENO Schemes and Discontinuous Galerkin Methods for CFD" - Chi-Wang Shu
/// ICASE Report No 2001-11 (page 6).  Also see ICASE No 97-65 for a more complete reference
/// (Shu, 1997).
/// Given v1 = f(x-2dx), v2 = f(x-dx), v3 = f(x), v4 = f(x+dx) and v5 = f(x+2dx),
/// return an interpolated value f(x+dx/2) with the special property that
/// ( f(x+dx/2) - f(x-dx/2) ) / dx  = df/dx (x) + error,
/// where the error is fifth-order in smooth regions: O(dx) <= error <=O(dx^5)
template<typename ValueType>
inline ValueType
WENO5(const ValueType& v1, const ValueType& v2, const ValueType& v3,
    const ValueType& v4, const ValueType& v5, float scale2 = 0.01f)
{
    const double C = 13.0 / 12.0;
    // WENO is formulated for non-dimensional equations, here the optional scale2
    // is a reference value (squared) for the function being interpolated.  For
    // example if 'v' is of order 1000, then scale2 = 10^6 is ok.  But in practice
    // leave scale2 = 1.
    const double eps = 1.0e-6 * static_cast<double>(scale2);
    // {\tilde \omega_k} = \gamma_k / ( \beta_k + \epsilon)^2 in Shu's ICASE report)
    const double A1=0.1/math::Pow2(C*math::Pow2(v1-2*v2+v3)+0.25*math::Pow2(v1-4*v2+3.0*v3)+eps),
                 A2=0.6/math::Pow2(C*math::Pow2(v2-2*v3+v4)+0.25*math::Pow2(v2-v4)+eps),
                 A3=0.3/math::Pow2(C*math::Pow2(v3-2*v4+v5)+0.25*math::Pow2(3.0*v3-4*v4+v5)+eps);

    return static_cast<ValueType>(static_cast<ValueType>(
        A1*(2.0*v1 - 7.0*v2 + 11.0*v3) +
        A2*(5.0*v3 -     v2 +  2.0*v4) +
        A3*(2.0*v3 + 5.0*v4 -      v5))/(6.0*(A1+A2+A3)));
}


template <typename Real>
inline Real GodunovsNormSqrd(bool isOutside,
                             Real dP_xm, Real dP_xp,
                             Real dP_ym, Real dP_yp,
                             Real dP_zm, Real dP_zp)
{
    using math::Max;
    using math::Min;
    using math::Pow2;

    const Real zero(0);
    Real dPLen2;
    if (isOutside) { // outside
        dPLen2  = Max(Pow2(Max(dP_xm, zero)), Pow2(Min(dP_xp,zero))); // (dP/dx)2
        dPLen2 += Max(Pow2(Max(dP_ym, zero)), Pow2(Min(dP_yp,zero))); // (dP/dy)2
        dPLen2 += Max(Pow2(Max(dP_zm, zero)), Pow2(Min(dP_zp,zero))); // (dP/dz)2
    } else { // inside
        dPLen2  = Max(Pow2(Min(dP_xm, zero)), Pow2(Max(dP_xp,zero))); // (dP/dx)2
        dPLen2 += Max(Pow2(Min(dP_ym, zero)), Pow2(Max(dP_yp,zero))); // (dP/dy)2
        dPLen2 += Max(Pow2(Min(dP_zm, zero)), Pow2(Max(dP_zp,zero))); // (dP/dz)2
    }
    return dPLen2; // |\nabla\phi|^2
}


template<typename Real>
inline Real
GodunovsNormSqrd(bool isOutside, const Vec3<Real>& gradient_m, const Vec3<Real>& gradient_p)
{
    return GodunovsNormSqrd<Real>(isOutside,
                                  gradient_m[0], gradient_p[0],
                                  gradient_m[1], gradient_p[1],
                                  gradient_m[2], gradient_p[2]);
}


#ifdef DWA_OPENVDB
inline simd::Float4 simdMin(const simd::Float4& a, const simd::Float4& b) {
    return simd::Float4(_mm_min_ps(a.base(), b.base()));
}
inline simd::Float4 simdMax(const simd::Float4& a, const simd::Float4& b) {
    return simd::Float4(_mm_max_ps(a.base(), b.base()));
}

inline float simdSum(const simd::Float4& v);

inline simd::Float4 Pow2(const simd::Float4& v) { return v * v; }

template<>
inline simd::Float4
WENO5<simd::Float4>(const simd::Float4& v1, const simd::Float4& v2, const simd::Float4& v3,
                    const simd::Float4& v4, const simd::Float4& v5, float scale2)
{
    using math::Pow2;
    using F4 = simd::Float4;
    const F4
        C(13.f / 12.f),
        eps(1.0e-6f * scale2),
        two(2.0), three(3.0), four(4.0), five(5.0), fourth(0.25),
        A1 = F4(0.1f) / Pow2(C*Pow2(v1-two*v2+v3) + fourth*Pow2(v1-four*v2+three*v3) + eps),
        A2 = F4(0.6f) / Pow2(C*Pow2(v2-two*v3+v4) + fourth*Pow2(v2-v4) + eps),
        A3 = F4(0.3f) / Pow2(C*Pow2(v3-two*v4+v5) + fourth*Pow2(three*v3-four*v4+v5) + eps);
    return (A1 * (two * v1 - F4(7.0) * v2 + F4(11.0) * v3) +
            A2 * (five * v3 - v2 + two * v4) +
            A3 * (two * v3 + five * v4 - v5)) / (F4(6.0) * (A1 + A2 + A3));
}


inline float
simdSum(const simd::Float4& v)
{
    // temp = { v3+v3, v2+v2, v1+v3, v0+v2 }
    __m128 temp = _mm_add_ps(v.base(), _mm_movehl_ps(v.base(), v.base()));
    // temp = { v3+v3, v2+v2, v1+v3, (v0+v2)+(v1+v3) }
    temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    return _mm_cvtss_f32(temp);
}

inline float
GodunovsNormSqrd(bool isOutside, const simd::Float4& dP_m, const simd::Float4& dP_p)
{
    const simd::Float4 zero(0.0);
    simd::Float4 v = isOutside
        ? simdMax(math::Pow2(simdMax(dP_m, zero)), math::Pow2(simdMin(dP_p, zero)))
        : simdMax(math::Pow2(simdMin(dP_m, zero)), math::Pow2(simdMax(dP_p, zero)));
    return simdSum(v);//should be v[0]+v[1]+v[2]
}
#endif


template<DScheme DiffScheme>
struct D1
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk);

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S);

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S);

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S);
};

template<>
struct D1<CD_2NDT>
{
    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp1, const ValueType& xm1) {
        return xp1 - xm1;
    }

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(1, 0, 0)),
            grid.getValue(ijk.offsetBy(-1, 0, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 1, 0)),
            grid.getValue(ijk.offsetBy( 0, -1, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 0, 1)),
            grid.getValue(ijk.offsetBy( 0, 0, -1)));
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue< 1, 0, 0>(),  S.template getValue<-1, 0, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 1, 0>(),  S.template getValue< 0,-1, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 1>(),  S.template getValue< 0, 0,-1>());
    }
};

template<>
struct D1<CD_2ND>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp1, const ValueType& xm1) {
        return (xp1 - xm1)*ValueType(0.5);
    }


    // random access
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(1, 0, 0)),
            grid.getValue(ijk.offsetBy(-1, 0, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 1, 0)),
            grid.getValue(ijk.offsetBy( 0, -1, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 0, 1)),
            grid.getValue(ijk.offsetBy( 0, 0, -1)));
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference(S.template getValue< 1, 0, 0>(), S.template getValue<-1, 0, 0>());
    }
    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 1, 0>(), S.template getValue< 0,-1, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0,-1>());
    }

};

template<>
struct D1<CD_4TH>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference( const ValueType& xp2, const ValueType& xp1,
                                 const ValueType& xm1, const ValueType& xm2 ) {
        return ValueType(2./3.)*(xp1 - xm1) + ValueType(1./12.)*(xm2 - xp2) ;
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 2,0,0)), grid.getValue(ijk.offsetBy( 1,0,0)),
            grid.getValue(ijk.offsetBy(-1,0,0)), grid.getValue(ijk.offsetBy(-2,0,0)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {

        return difference(
            grid.getValue(ijk.offsetBy( 0, 2, 0)), grid.getValue(ijk.offsetBy( 0, 1, 0)),
            grid.getValue(ijk.offsetBy( 0,-1, 0)), grid.getValue(ijk.offsetBy( 0,-2, 0)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {

        return difference(
            grid.getValue(ijk.offsetBy( 0, 0, 2)), grid.getValue(ijk.offsetBy( 0, 0, 1)),
            grid.getValue(ijk.offsetBy( 0, 0,-1)), grid.getValue(ijk.offsetBy( 0, 0,-2)) );
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue<-2, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0,-2, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0,-2>() );
    }
};

template<>
struct D1<CD_6TH>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference( const ValueType& xp3, const ValueType& xp2, const ValueType& xp1,
                                 const ValueType& xm1, const ValueType& xm2, const ValueType& xm3 )
    {
        return ValueType(3./4.)*(xp1 - xm1) - ValueType(0.15)*(xp2 - xm2)
            + ValueType(1./60.)*(xp3-xm3);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 3,0,0)), grid.getValue(ijk.offsetBy( 2,0,0)),
            grid.getValue(ijk.offsetBy( 1,0,0)), grid.getValue(ijk.offsetBy(-1,0,0)),
            grid.getValue(ijk.offsetBy(-2,0,0)), grid.getValue(ijk.offsetBy(-3,0,0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 3, 0)), grid.getValue(ijk.offsetBy( 0, 2, 0)),
            grid.getValue(ijk.offsetBy( 0, 1, 0)), grid.getValue(ijk.offsetBy( 0,-1, 0)),
            grid.getValue(ijk.offsetBy( 0,-2, 0)), grid.getValue(ijk.offsetBy( 0,-3, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 0, 3)), grid.getValue(ijk.offsetBy( 0, 0, 2)),
            grid.getValue(ijk.offsetBy( 0, 0, 1)), grid.getValue(ijk.offsetBy( 0, 0,-1)),
            grid.getValue(ijk.offsetBy( 0, 0,-2)), grid.getValue(ijk.offsetBy( 0, 0,-3)));
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return  difference(S.template getValue< 3, 0, 0>(),
                           S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue<-2, 0, 0>(),
                           S.template getValue<-3, 0, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {

        return  difference( S.template getValue< 0, 3, 0>(),
                            S.template getValue< 0, 2, 0>(),
                            S.template getValue< 0, 1, 0>(),
                            S.template getValue< 0,-1, 0>(),
                            S.template getValue< 0,-2, 0>(),
                            S.template getValue< 0,-3, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {

        return  difference( S.template getValue< 0, 0, 3>(),
                            S.template getValue< 0, 0, 2>(),
                            S.template getValue< 0, 0, 1>(),
                            S.template getValue< 0, 0,-1>(),
                            S.template getValue< 0, 0,-2>(),
                            S.template getValue< 0, 0,-3>());
    }
};


template<>
struct D1<FD_1ST>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp1, const ValueType& xp0) {
        return xp1 - xp0;
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(1, 0, 0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0, 1, 0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0, 0, 1)), grid.getValue(ijk));
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference(S.template getValue< 1, 0, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 1, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0, 0>());
    }
};


template<>
struct D1<FD_2ND>
{
    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp2, const ValueType& xp1, const ValueType& xp0)
    {
        return ValueType(2)*xp1 -(ValueType(0.5)*xp2 + ValueType(3./2.)*xp0);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(2,0,0)),
            grid.getValue(ijk.offsetBy(1,0,0)),
            grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0,2,0)),
            grid.getValue(ijk.offsetBy(0,1,0)),
            grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0,0,2)),
            grid.getValue(ijk.offsetBy(0,0,1)),
            grid.getValue(ijk));
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>() );
    }

};


template<>
struct D1<FD_3RD>
{

    // the difference opperator
    template<typename ValueType>
    static ValueType difference(const ValueType& xp3, const ValueType& xp2,
        const ValueType& xp1, const ValueType& xp0)
    {
        return static_cast<ValueType>(xp3/3.0 - 1.5*xp2 + 3.0*xp1 - 11.0*xp0/6.0);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(3,0,0)),
                           grid.getValue(ijk.offsetBy(2,0,0)),
                           grid.getValue(ijk.offsetBy(1,0,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,3,0)),
                           grid.getValue(ijk.offsetBy(0,2,0)),
                           grid.getValue(ijk.offsetBy(0,1,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,0,3)),
                           grid.getValue(ijk.offsetBy(0,0,2)),
                           grid.getValue(ijk.offsetBy(0,0,1)),
                           grid.getValue(ijk) );
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference(S.template getValue< 3, 0, 0>(),
                          S.template getValue< 2, 0, 0>(),
                          S.template getValue< 1, 0, 0>(),
                          S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 3, 0>(),
                          S.template getValue< 0, 2, 0>(),
                          S.template getValue< 0, 1, 0>(),
                          S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 3>(),
                           S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>() );
    }
};


template<>
struct D1<BD_1ST>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xm1, const ValueType& xm0) {
        return -D1<FD_1ST>::difference(xm1, xm0);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(-1,0,0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0,-1,0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0, 0,-1)), grid.getValue(ijk));
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference(S.template getValue<-1, 0, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference(S.template getValue< 0,-1, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0,-1>(), S.template getValue< 0, 0, 0>());
    }
};


template<>
struct D1<BD_2ND>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xm2, const ValueType& xm1, const ValueType& xm0)
    {
        return -D1<FD_2ND>::difference(xm2, xm1, xm0);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(-2,0,0)),
                           grid.getValue(ijk.offsetBy(-1,0,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,-2,0)),
                           grid.getValue(ijk.offsetBy(0,-1,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,0,-2)),
                           grid.getValue(ijk.offsetBy(0,0,-1)),
                           grid.getValue(ijk) );
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue<-2, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0,-2, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0,-2>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0, 0>() );
    }
};


template<>
struct D1<BD_3RD>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xm3, const ValueType& xm2,
        const ValueType& xm1, const ValueType& xm0)
    {
        return -D1<FD_3RD>::difference(xm3, xm2, xm1, xm0);
    }

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(-3,0,0)),
                           grid.getValue(ijk.offsetBy(-2,0,0)),
                           grid.getValue(ijk.offsetBy(-1,0,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 0,-3,0)),
                           grid.getValue(ijk.offsetBy( 0,-2,0)),
                           grid.getValue(ijk.offsetBy( 0,-1,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 0, 0,-3)),
                           grid.getValue(ijk.offsetBy( 0, 0,-2)),
                           grid.getValue(ijk.offsetBy( 0, 0,-1)),
                           grid.getValue(ijk) );
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue<-3, 0, 0>(),
                           S.template getValue<-2, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0,-3, 0>(),
                           S.template getValue< 0,-2, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0,-3>(),
                           S.template getValue< 0, 0,-2>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0, 0>() );
    }

};

template<>
struct D1<FD_WENO5>
{
    // the difference operator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp3, const ValueType& xp2,
                                const ValueType& xp1, const ValueType& xp0,
                                const ValueType& xm1, const ValueType& xm2) {
        return WENO5<ValueType>(xp3, xp2, xp1, xp0, xm1)
              - WENO5<ValueType>(xp2, xp1, xp0, xm1, xm2);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(-2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,-2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,-2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {

        return static_cast<typename Stencil::ValueType>(difference(
            S.template getValue< 3, 0, 0>(),
            S.template getValue< 2, 0, 0>(),
            S.template getValue< 1, 0, 0>(),
            S.template getValue< 0, 0, 0>(),
            S.template getValue<-1, 0, 0>(),
            S.template getValue<-2, 0, 0>() ));

    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return static_cast<typename Stencil::ValueType>(difference(
            S.template getValue< 0, 3, 0>(),
            S.template getValue< 0, 2, 0>(),
            S.template getValue< 0, 1, 0>(),
            S.template getValue< 0, 0, 0>(),
            S.template getValue< 0,-1, 0>(),
            S.template getValue< 0,-2, 0>() ));
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return static_cast<typename Stencil::ValueType>(difference(
            S.template getValue< 0, 0, 3>(),
            S.template getValue< 0, 0, 2>(),
            S.template getValue< 0, 0, 1>(),
            S.template getValue< 0, 0, 0>(),
            S.template getValue< 0, 0,-1>(),
            S.template getValue< 0, 0,-2>() ));
    }
};

template<>
struct D1<FD_HJWENO5>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp3, const ValueType& xp2,
                                const ValueType& xp1, const ValueType& xp0,
                                const ValueType& xm1, const ValueType& xm2) {
        return WENO5<ValueType>(xp3 - xp2, xp2 - xp1, xp1 - xp0, xp0-xm1, xm1-xm2);
    }

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(-2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);

    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,-2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,-2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {

        return difference( S.template getValue< 3, 0, 0>(),
                           S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue<-2, 0, 0>() );

    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 3, 0>(),
                           S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0,-2, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {

        return difference( S.template getValue< 0, 0, 3>(),
                           S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0,-2>() );
    }

};

template<>
struct D1<BD_WENO5>
{

    template<typename ValueType>
    static ValueType difference(const ValueType& xm3, const ValueType& xm2, const ValueType& xm1,
                                const ValueType& xm0, const ValueType& xp1, const ValueType& xp2)
    {
        return -D1<FD_WENO5>::difference(xm3, xm2, xm1, xm0, xp1, xp2);
    }


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(-3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(-2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,-3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,-2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,-3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,-2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue<-3, 0, 0>();
        V[1] = S.template getValue<-2, 0, 0>();
        V[2] = S.template getValue<-1, 0, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 1, 0, 0>();
        V[5] = S.template getValue< 2, 0, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0,-3, 0>();
        V[1] = S.template getValue< 0,-2, 0>();
        V[2] = S.template getValue< 0,-1, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 1, 0>();
        V[5] = S.template getValue< 0, 2, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0, 0,-3>();
        V[1] = S.template getValue< 0, 0,-2>();
        V[2] = S.template getValue< 0, 0,-1>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 0, 1>();
        V[5] = S.template getValue< 0, 0, 2>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }
};


template<>
struct D1<BD_HJWENO5>
{
    template<typename ValueType>
    static ValueType difference(const ValueType& xm3, const ValueType& xm2, const ValueType& xm1,
                                const ValueType& xm0, const ValueType& xp1, const ValueType& xp2)
    {
        return -D1<FD_HJWENO5>::difference(xm3, xm2, xm1, xm0, xp1, xp2);
    }

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(-3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(-2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,-3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,-2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,-3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,-2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue<-3, 0, 0>();
        V[1] = S.template getValue<-2, 0, 0>();
        V[2] = S.template getValue<-1, 0, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 1, 0, 0>();
        V[5] = S.template getValue< 2, 0, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0,-3, 0>();
        V[1] = S.template getValue< 0,-2, 0>();
        V[2] = S.template getValue< 0,-1, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 1, 0>();
        V[5] = S.template getValue< 0, 2, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0, 0,-3>();
        V[1] = S.template getValue< 0, 0,-2>();
        V[2] = S.template getValue< 0, 0,-1>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 0, 1>();
        V[5] = S.template getValue< 0, 0, 2>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }
};


template<DScheme DiffScheme>
struct D1Vec
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<DiffScheme>::inX(grid, ijk)[n];
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<DiffScheme>::inY(grid, ijk)[n];
    }
    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<DiffScheme>::inZ(grid, ijk)[n];
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType::value_type inX(const Stencil& S, int n)
    {
        return D1<DiffScheme>::inX(S)[n];
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inY(const Stencil& S, int n)
    {
        return D1<DiffScheme>::inY(S)[n];
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inZ(const Stencil& S, int n)
    {
        return D1<DiffScheme>::inZ(S)[n];
    }
};


template<>
struct D1Vec<CD_2NDT>
{

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2NDT>::difference( grid.getValue(ijk.offsetBy( 1, 0, 0))[n],
                                        grid.getValue(ijk.offsetBy(-1, 0, 0))[n] );
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2NDT>::difference( grid.getValue(ijk.offsetBy(0, 1, 0))[n],
                                        grid.getValue(ijk.offsetBy(0,-1, 0))[n] );
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2NDT>::difference( grid.getValue(ijk.offsetBy(0, 0, 1))[n],
                                        grid.getValue(ijk.offsetBy(0, 0,-1))[n] );
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType::value_type inX(const Stencil& S, int n)
    {
        return D1<CD_2NDT>::difference( S.template getValue< 1, 0, 0>()[n],
                                        S.template getValue<-1, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inY(const Stencil& S, int n)
    {
        return D1<CD_2NDT>::difference( S.template getValue< 0, 1, 0>()[n],
                                        S.template getValue< 0,-1, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inZ(const Stencil& S, int n)
    {
        return D1<CD_2NDT>::difference( S.template getValue< 0, 0, 1>()[n],
                                        S.template getValue< 0, 0,-1>()[n] );
    }
};

template<>
struct D1Vec<CD_2ND>
{

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2ND>::difference( grid.getValue(ijk.offsetBy( 1, 0, 0))[n] ,
                                       grid.getValue(ijk.offsetBy(-1, 0, 0))[n] );
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2ND>::difference( grid.getValue(ijk.offsetBy(0, 1, 0))[n] ,
                                       grid.getValue(ijk.offsetBy(0,-1, 0))[n] );
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2ND>::difference( grid.getValue(ijk.offsetBy(0, 0, 1))[n] ,
                                       grid.getValue(ijk.offsetBy(0, 0,-1))[n] );
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType::value_type inX(const Stencil& S, int n)
    {
        return D1<CD_2ND>::difference( S.template getValue< 1, 0, 0>()[n],
                                       S.template getValue<-1, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inY(const Stencil& S, int n)
    {
        return D1<CD_2ND>::difference( S.template getValue< 0, 1, 0>()[n],
                                       S.template getValue< 0,-1, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inZ(const Stencil& S, int n)
    {
        return D1<CD_2ND>::difference( S.template getValue< 0, 0, 1>()[n],
                                       S.template getValue< 0, 0,-1>()[n] );
    }
};


template<>
struct D1Vec<CD_4TH> {
    // using value_type = typename Accessor::ValueType::value_type;


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_4TH>::difference(
            grid.getValue(ijk.offsetBy(2, 0, 0))[n], grid.getValue(ijk.offsetBy( 1, 0, 0))[n],
            grid.getValue(ijk.offsetBy(-1,0, 0))[n], grid.getValue(ijk.offsetBy(-2, 0, 0))[n]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_4TH>::difference(
            grid.getValue(ijk.offsetBy( 0, 2, 0))[n], grid.getValue(ijk.offsetBy( 0, 1, 0))[n],
            grid.getValue(ijk.offsetBy( 0,-1, 0))[n], grid.getValue(ijk.offsetBy( 0,-2, 0))[n]);
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_4TH>::difference(
            grid.getValue(ijk.offsetBy(0,0, 2))[n], grid.getValue(ijk.offsetBy( 0, 0, 1))[n],
            grid.getValue(ijk.offsetBy(0,0,-1))[n], grid.getValue(ijk.offsetBy( 0, 0,-2))[n]);
    }

    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType::value_type inX(const Stencil& S, int n)
    {
        return D1<CD_4TH>::difference(
            S.template getValue< 2, 0, 0>()[n],  S.template getValue< 1, 0, 0>()[n],
            S.template getValue<-1, 0, 0>()[n],  S.template getValue<-2, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inY(const Stencil& S, int n)
    {
        return D1<CD_4TH>::difference(
            S.template getValue< 0, 2, 0>()[n],  S.template getValue< 0, 1, 0>()[n],
            S.template getValue< 0,-1, 0>()[n],  S.template getValue< 0,-2, 0>()[n]);
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inZ(const Stencil& S, int n)
    {
        return D1<CD_4TH>::difference(
            S.template getValue< 0, 0, 2>()[n],  S.template getValue< 0, 0, 1>()[n],
            S.template getValue< 0, 0,-1>()[n],  S.template getValue< 0, 0,-2>()[n]);
    }
};


template<>
struct D1Vec<CD_6TH>
{
    //using ValueType = typename Accessor::ValueType::value_type::value_type;

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_6TH>::difference(
            grid.getValue(ijk.offsetBy( 3, 0, 0))[n], grid.getValue(ijk.offsetBy( 2, 0, 0))[n],
            grid.getValue(ijk.offsetBy( 1, 0, 0))[n], grid.getValue(ijk.offsetBy(-1, 0, 0))[n],
            grid.getValue(ijk.offsetBy(-2, 0, 0))[n], grid.getValue(ijk.offsetBy(-3, 0, 0))[n] );
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_6TH>::difference(
            grid.getValue(ijk.offsetBy( 0, 3, 0))[n], grid.getValue(ijk.offsetBy( 0, 2, 0))[n],
            grid.getValue(ijk.offsetBy( 0, 1, 0))[n], grid.getValue(ijk.offsetBy( 0,-1, 0))[n],
            grid.getValue(ijk.offsetBy( 0,-2, 0))[n], grid.getValue(ijk.offsetBy( 0,-3, 0))[n] );
    }

    template<typename Accessor>
    static typename Accessor::ValueType::value_type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_6TH>::difference(
            grid.getValue(ijk.offsetBy( 0, 0, 3))[n], grid.getValue(ijk.offsetBy( 0, 0, 2))[n],
            grid.getValue(ijk.offsetBy( 0, 0, 1))[n], grid.getValue(ijk.offsetBy( 0, 0,-1))[n],
            grid.getValue(ijk.offsetBy( 0, 0,-2))[n], grid.getValue(ijk.offsetBy( 0, 0,-3))[n] );
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType::value_type inX(const Stencil& S, int n)
    {
        return D1<CD_6TH>::difference(
            S.template getValue< 3, 0, 0>()[n], S.template getValue< 2, 0, 0>()[n],
            S.template getValue< 1, 0, 0>()[n], S.template getValue<-1, 0, 0>()[n],
            S.template getValue<-2, 0, 0>()[n], S.template getValue<-3, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inY(const Stencil& S, int n)
    {
        return D1<CD_6TH>::difference(
            S.template getValue< 0, 3, 0>()[n], S.template getValue< 0, 2, 0>()[n],
            S.template getValue< 0, 1, 0>()[n], S.template getValue< 0,-1, 0>()[n],
            S.template getValue< 0,-2, 0>()[n], S.template getValue< 0,-3, 0>()[n] );
    }

    template<typename Stencil>
    static typename Stencil::ValueType::value_type inZ(const Stencil& S, int n)
    {
        return D1<CD_6TH>::difference(
            S.template getValue< 0, 0, 3>()[n], S.template getValue< 0, 0, 2>()[n],
            S.template getValue< 0, 0, 1>()[n], S.template getValue< 0, 0,-1>()[n],
            S.template getValue< 0, 0,-2>()[n], S.template getValue< 0, 0,-3>()[n] );
    }
};

template<DDScheme DiffScheme>
struct D2
{

    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk);
    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk);
    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk);

    // cross derivatives
    template<typename Accessor>
    static typename Accessor::ValueType inXandY(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename Accessor::ValueType inXandZ(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename Accessor::ValueType inYandZ(const Accessor& grid, const Coord& ijk);


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S);
    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S);
    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S);

    // cross derivatives
    template<typename Stencil>
    static typename Stencil::ValueType inXandY(const Stencil& S);

    template<typename Stencil>
    static typename Stencil::ValueType inXandZ(const Stencil& S);

    template<typename Stencil>
    static typename Stencil::ValueType inYandZ(const Stencil& S);
};

template<>
struct D2<CD_SECOND>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp1, const ValueType& xp0, const ValueType& xm1)
    {
        return xp1 + xm1 - ValueType(2)*xp0;
    }

    template <typename ValueType>
    static ValueType crossdifference(const ValueType& xpyp, const ValueType& xpym,
                                     const ValueType& xmyp, const ValueType& xmym)
    {
        return ValueType(0.25)*(xpyp + xmym - xpym - xmyp);
    }

    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 1,0,0)), grid.getValue(ijk),
                           grid.getValue(ijk.offsetBy(-1,0,0)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {

        return difference( grid.getValue(ijk.offsetBy(0, 1,0)), grid.getValue(ijk),
                           grid.getValue(ijk.offsetBy(0,-1,0)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 0,0, 1)), grid.getValue(ijk),
                           grid.getValue(ijk.offsetBy( 0,0,-1)) );
    }

    // cross derivatives
    template<typename Accessor>
    static typename Accessor::ValueType inXandY(const Accessor& grid, const Coord& ijk)
    {
        return crossdifference(
            grid.getValue(ijk.offsetBy(1, 1,0)), grid.getValue(ijk.offsetBy( 1,-1,0)),
            grid.getValue(ijk.offsetBy(-1,1,0)), grid.getValue(ijk.offsetBy(-1,-1,0)));

    }

    template<typename Accessor>
    static typename Accessor::ValueType inXandZ(const Accessor& grid, const Coord& ijk)
    {
        return crossdifference(
            grid.getValue(ijk.offsetBy(1,0, 1)), grid.getValue(ijk.offsetBy(1, 0,-1)),
            grid.getValue(ijk.offsetBy(-1,0,1)), grid.getValue(ijk.offsetBy(-1,0,-1)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inYandZ(const Accessor& grid, const Coord& ijk)
    {
        return crossdifference(
            grid.getValue(ijk.offsetBy(0, 1,1)), grid.getValue(ijk.offsetBy(0, 1,-1)),
            grid.getValue(ijk.offsetBy(0,-1,1)), grid.getValue(ijk.offsetBy(0,-1,-1)) );
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue< 1, 0, 0>(), S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 1, 0>(), S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>() );
    }

    // cross derivatives
    template<typename Stencil>
    static typename Stencil::ValueType inXandY(const Stencil& S)
    {
        return crossdifference(S.template getValue< 1, 1, 0>(),  S.template getValue< 1,-1, 0>(),
                               S.template getValue<-1, 1, 0>(),  S.template getValue<-1,-1, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inXandZ(const Stencil& S)
    {
        return crossdifference(S.template getValue< 1, 0, 1>(),  S.template getValue< 1, 0,-1>(),
                               S.template getValue<-1, 0, 1>(),  S.template getValue<-1, 0,-1>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inYandZ(const Stencil& S)
    {
        return crossdifference(S.template getValue< 0, 1, 1>(),  S.template getValue< 0, 1,-1>(),
                               S.template getValue< 0,-1, 1>(),  S.template getValue< 0,-1,-1>() );
    }
};


template<>
struct D2<CD_FOURTH>
{

    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp2, const ValueType& xp1, const ValueType& xp0,
                                const ValueType& xm1, const ValueType& xm2) {
        return ValueType(-1./12.)*(xp2 + xm2) + ValueType(4./3.)*(xp1 + xm1) -ValueType(2.5)*xp0;
    }

    template <typename ValueType>
    static ValueType crossdifference(const ValueType& xp2yp2, const ValueType& xp2yp1,
                                     const ValueType& xp2ym1, const ValueType& xp2ym2,
                                     const ValueType& xp1yp2, const ValueType& xp1yp1,
                                     const ValueType& xp1ym1, const ValueType& xp1ym2,
                                     const ValueType& xm2yp2, const ValueType& xm2yp1,
                                     const ValueType& xm2ym1, const ValueType& xm2ym2,
                                     const ValueType& xm1yp2, const ValueType& xm1yp1,
                                     const ValueType& xm1ym1, const ValueType& xm1ym2 ) {
        ValueType tmp1 =
            ValueType(2./3.0)*(xp1yp1 - xm1yp1 - xp1ym1 + xm1ym1)-
            ValueType(1./12.)*(xp2yp1 - xm2yp1 - xp2ym1 + xm2ym1);
        ValueType tmp2 =
            ValueType(2./3.0)*(xp1yp2 - xm1yp2 - xp1ym2 + xm1ym2)-
            ValueType(1./12.)*(xp2yp2 - xm2yp2 - xp2ym2 + xm2ym2);

        return ValueType(2./3.)*tmp1 - ValueType(1./12.)*tmp2;
    }



    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(2,0,0)),  grid.getValue(ijk.offsetBy( 1,0,0)),
            grid.getValue(ijk),
            grid.getValue(ijk.offsetBy(-1,0,0)), grid.getValue(ijk.offsetBy(-2, 0, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 2,0)), grid.getValue(ijk.offsetBy(0, 1,0)),
            grid.getValue(ijk),
            grid.getValue(ijk.offsetBy(0,-1,0)), grid.getValue(ijk.offsetBy(0,-2, 0)));
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {
         return difference(
             grid.getValue(ijk.offsetBy(0,0, 2)), grid.getValue(ijk.offsetBy(0, 0,1)),
             grid.getValue(ijk),
             grid.getValue(ijk.offsetBy(0,0,-1)), grid.getValue(ijk.offsetBy(0,0,-2)));
    }

    // cross derivatives
    template<typename Accessor>
    static typename Accessor::ValueType inXandY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        typename Accessor::ValueType tmp1 =
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0, 1, 0)) -
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0,-1, 0));
        typename Accessor::ValueType tmp2 =
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0, 2, 0)) -
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0,-2, 0));
        return ValueType(2./3.)*tmp1 - ValueType(1./12.)*tmp2;
    }

    template<typename Accessor>
    static typename Accessor::ValueType inXandZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        typename Accessor::ValueType tmp1 =
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0, 0,-1));
        typename Accessor::ValueType tmp2 =
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_4TH>::inX(grid, ijk.offsetBy(0, 0,-2));
        return ValueType(2./3.)*tmp1 - ValueType(1./12.)*tmp2;
    }

    template<typename Accessor>
    static typename Accessor::ValueType inYandZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        typename Accessor::ValueType tmp1 =
            D1<CD_4TH>::inY(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_4TH>::inY(grid, ijk.offsetBy(0, 0,-1));
        typename Accessor::ValueType tmp2 =
            D1<CD_4TH>::inY(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_4TH>::inY(grid, ijk.offsetBy(0, 0,-2));
        return ValueType(2./3.)*tmp1 - ValueType(1./12.)*tmp2;
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return  difference(S.template getValue< 2, 0, 0>(), S.template getValue< 1, 0, 0>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>(), S.template getValue<-2, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return  difference(S.template getValue< 0, 2, 0>(), S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>(), S.template getValue< 0,-2, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return  difference(S.template getValue< 0, 0, 2>(), S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>(), S.template getValue< 0, 0,-2>() );
    }

    // cross derivatives
    template<typename Stencil>
    static typename Stencil::ValueType inXandY(const Stencil& S)
     {
         return crossdifference(
             S.template getValue< 2, 2, 0>(), S.template getValue< 2, 1, 0>(),
             S.template getValue< 2,-1, 0>(), S.template getValue< 2,-2, 0>(),
             S.template getValue< 1, 2, 0>(), S.template getValue< 1, 1, 0>(),
             S.template getValue< 1,-1, 0>(), S.template getValue< 1,-2, 0>(),
             S.template getValue<-2, 2, 0>(), S.template getValue<-2, 1, 0>(),
             S.template getValue<-2,-1, 0>(), S.template getValue<-2,-2, 0>(),
             S.template getValue<-1, 2, 0>(), S.template getValue<-1, 1, 0>(),
             S.template getValue<-1,-1, 0>(), S.template getValue<-1,-2, 0>() );
     }

    template<typename Stencil>
    static typename Stencil::ValueType inXandZ(const Stencil& S)
    {
        return crossdifference(
            S.template getValue< 2, 0, 2>(), S.template getValue< 2, 0, 1>(),
            S.template getValue< 2, 0,-1>(), S.template getValue< 2, 0,-2>(),
            S.template getValue< 1, 0, 2>(), S.template getValue< 1, 0, 1>(),
            S.template getValue< 1, 0,-1>(), S.template getValue< 1, 0,-2>(),
            S.template getValue<-2, 0, 2>(), S.template getValue<-2, 0, 1>(),
            S.template getValue<-2, 0,-1>(), S.template getValue<-2, 0,-2>(),
            S.template getValue<-1, 0, 2>(), S.template getValue<-1, 0, 1>(),
            S.template getValue<-1, 0,-1>(), S.template getValue<-1, 0,-2>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inYandZ(const Stencil& S)
    {
        return crossdifference(
            S.template getValue< 0, 2, 2>(), S.template getValue< 0, 2, 1>(),
            S.template getValue< 0, 2,-1>(), S.template getValue< 0, 2,-2>(),
            S.template getValue< 0, 1, 2>(), S.template getValue< 0, 1, 1>(),
            S.template getValue< 0, 1,-1>(), S.template getValue< 0, 1,-2>(),
            S.template getValue< 0,-2, 2>(), S.template getValue< 0,-2, 1>(),
            S.template getValue< 0,-2,-1>(), S.template getValue< 0,-2,-2>(),
            S.template getValue< 0,-1, 2>(), S.template getValue< 0,-1, 1>(),
            S.template getValue< 0,-1,-1>(), S.template getValue< 0,-1,-2>() );
    }
};


template<>
struct D2<CD_SIXTH>
{
    // the difference opperator
    template <typename ValueType>
    static ValueType difference(const ValueType& xp3, const ValueType& xp2, const ValueType& xp1,
                                const ValueType& xp0,
                                const ValueType& xm1, const ValueType& xm2, const ValueType& xm3)
    {
        return  ValueType(1./90.)*(xp3 + xm3) - ValueType(3./20.)*(xp2 + xm2)
              + ValueType(1.5)*(xp1 + xm1) - ValueType(49./18.)*xp0;
    }

    template <typename ValueType>
    static ValueType crossdifference( const ValueType& xp1yp1,const ValueType& xm1yp1,
                                      const ValueType& xp1ym1,const ValueType& xm1ym1,
                                      const ValueType& xp2yp1,const ValueType& xm2yp1,
                                      const ValueType& xp2ym1,const ValueType& xm2ym1,
                                      const ValueType& xp3yp1,const ValueType& xm3yp1,
                                      const ValueType& xp3ym1,const ValueType& xm3ym1,
                                      const ValueType& xp1yp2,const ValueType& xm1yp2,
                                      const ValueType& xp1ym2,const ValueType& xm1ym2,
                                      const ValueType& xp2yp2,const ValueType& xm2yp2,
                                      const ValueType& xp2ym2,const ValueType& xm2ym2,
                                      const ValueType& xp3yp2,const ValueType& xm3yp2,
                                      const ValueType& xp3ym2,const ValueType& xm3ym2,
                                      const ValueType& xp1yp3,const ValueType& xm1yp3,
                                      const ValueType& xp1ym3,const ValueType& xm1ym3,
                                      const ValueType& xp2yp3,const ValueType& xm2yp3,
                                      const ValueType& xp2ym3,const ValueType& xm2ym3,
                                      const ValueType& xp3yp3,const ValueType& xm3yp3,
                                      const ValueType& xp3ym3,const ValueType& xm3ym3 )
    {
        ValueType tmp1 =
            ValueType(0.7500)*(xp1yp1 - xm1yp1 - xp1ym1 + xm1ym1) -
            ValueType(0.1500)*(xp2yp1 - xm2yp1 - xp2ym1 + xm2ym1) +
            ValueType(1./60.)*(xp3yp1 - xm3yp1 - xp3ym1 + xm3ym1);

        ValueType tmp2 =
            ValueType(0.7500)*(xp1yp2 - xm1yp2 - xp1ym2 + xm1ym2) -
            ValueType(0.1500)*(xp2yp2 - xm2yp2 - xp2ym2 + xm2ym2) +
            ValueType(1./60.)*(xp3yp2 - xm3yp2 - xp3ym2 + xm3ym2);

        ValueType tmp3 =
            ValueType(0.7500)*(xp1yp3 - xm1yp3 - xp1ym3 + xm1ym3) -
            ValueType(0.1500)*(xp2yp3 - xm2yp3 - xp2ym3 + xm2ym3) +
            ValueType(1./60.)*(xp3yp3 - xm3yp3 - xp3ym3 + xm3ym3);

        return ValueType(0.75)*tmp1 - ValueType(0.15)*tmp2 + ValueType(1./60)*tmp3;
    }

    // random access version

    template<typename Accessor>
    static typename Accessor::ValueType inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 3, 0, 0)), grid.getValue(ijk.offsetBy( 2, 0, 0)),
            grid.getValue(ijk.offsetBy( 1, 0, 0)), grid.getValue(ijk),
            grid.getValue(ijk.offsetBy(-1, 0, 0)), grid.getValue(ijk.offsetBy(-2, 0, 0)),
            grid.getValue(ijk.offsetBy(-3, 0, 0)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 3, 0)), grid.getValue(ijk.offsetBy( 0, 2, 0)),
            grid.getValue(ijk.offsetBy( 0, 1, 0)), grid.getValue(ijk),
            grid.getValue(ijk.offsetBy( 0,-1, 0)), grid.getValue(ijk.offsetBy( 0,-2, 0)),
            grid.getValue(ijk.offsetBy( 0,-3, 0)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inZ(const Accessor& grid, const Coord& ijk)
    {

        return difference(
            grid.getValue(ijk.offsetBy( 0, 0, 3)), grid.getValue(ijk.offsetBy( 0, 0, 2)),
            grid.getValue(ijk.offsetBy( 0, 0, 1)), grid.getValue(ijk),
            grid.getValue(ijk.offsetBy( 0, 0,-1)), grid.getValue(ijk.offsetBy( 0, 0,-2)),
            grid.getValue(ijk.offsetBy( 0, 0,-3)) );
    }

    template<typename Accessor>
    static typename Accessor::ValueType inXandY(const Accessor& grid, const Coord& ijk)
    {
        using ValueT = typename Accessor::ValueType;
        ValueT tmp1 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 1, 0)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0,-1, 0));
        ValueT tmp2 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 2, 0)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0,-2, 0));
        ValueT tmp3 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 3, 0)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0,-3, 0));
        return ValueT(0.75*tmp1 - 0.15*tmp2 + 1./60*tmp3);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inXandZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueT = typename Accessor::ValueType;
        ValueT tmp1 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0,-1));
        ValueT tmp2 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0,-2));
        ValueT tmp3 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0, 3)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0,-3));
        return ValueT(0.75*tmp1 - 0.15*tmp2 + 1./60*tmp3);
    }

    template<typename Accessor>
    static typename Accessor::ValueType inYandZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueT = typename Accessor::ValueType;
        ValueT tmp1 =
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0,-1));
        ValueT tmp2 =
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0,-2));
        ValueT tmp3 =
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0, 3)) -
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0,-3));
        return ValueT(0.75*tmp1 - 0.15*tmp2 + 1./60*tmp3);
    }


    // stencil access version
    template<typename Stencil>
    static typename Stencil::ValueType inX(const Stencil& S)
    {
        return difference( S.template getValue< 3, 0, 0>(),  S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),  S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),  S.template getValue<-2, 0, 0>(),
                           S.template getValue<-3, 0, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 3, 0>(),  S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),  S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>(),  S.template getValue< 0,-2, 0>(),
                           S.template getValue< 0,-3, 0>() );

    }

    template<typename Stencil>
    static typename Stencil::ValueType inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 3>(),  S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),  S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>(),  S.template getValue< 0, 0,-2>(),
                           S.template getValue< 0, 0,-3>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inXandY(const Stencil& S)
    {
        return crossdifference( S.template getValue< 1, 1, 0>(), S.template getValue<-1, 1, 0>(),
                                S.template getValue< 1,-1, 0>(), S.template getValue<-1,-1, 0>(),
                                S.template getValue< 2, 1, 0>(), S.template getValue<-2, 1, 0>(),
                                S.template getValue< 2,-1, 0>(), S.template getValue<-2,-1, 0>(),
                                S.template getValue< 3, 1, 0>(), S.template getValue<-3, 1, 0>(),
                                S.template getValue< 3,-1, 0>(), S.template getValue<-3,-1, 0>(),
                                S.template getValue< 1, 2, 0>(), S.template getValue<-1, 2, 0>(),
                                S.template getValue< 1,-2, 0>(), S.template getValue<-1,-2, 0>(),
                                S.template getValue< 2, 2, 0>(), S.template getValue<-2, 2, 0>(),
                                S.template getValue< 2,-2, 0>(), S.template getValue<-2,-2, 0>(),
                                S.template getValue< 3, 2, 0>(), S.template getValue<-3, 2, 0>(),
                                S.template getValue< 3,-2, 0>(), S.template getValue<-3,-2, 0>(),
                                S.template getValue< 1, 3, 0>(), S.template getValue<-1, 3, 0>(),
                                S.template getValue< 1,-3, 0>(), S.template getValue<-1,-3, 0>(),
                                S.template getValue< 2, 3, 0>(), S.template getValue<-2, 3, 0>(),
                                S.template getValue< 2,-3, 0>(), S.template getValue<-2,-3, 0>(),
                                S.template getValue< 3, 3, 0>(), S.template getValue<-3, 3, 0>(),
                                S.template getValue< 3,-3, 0>(), S.template getValue<-3,-3, 0>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inXandZ(const Stencil& S)
    {
        return crossdifference( S.template getValue< 1, 0, 1>(), S.template getValue<-1, 0, 1>(),
                                S.template getValue< 1, 0,-1>(), S.template getValue<-1, 0,-1>(),
                                S.template getValue< 2, 0, 1>(), S.template getValue<-2, 0, 1>(),
                                S.template getValue< 2, 0,-1>(), S.template getValue<-2, 0,-1>(),
                                S.template getValue< 3, 0, 1>(), S.template getValue<-3, 0, 1>(),
                                S.template getValue< 3, 0,-1>(), S.template getValue<-3, 0,-1>(),
                                S.template getValue< 1, 0, 2>(), S.template getValue<-1, 0, 2>(),
                                S.template getValue< 1, 0,-2>(), S.template getValue<-1, 0,-2>(),
                                S.template getValue< 2, 0, 2>(), S.template getValue<-2, 0, 2>(),
                                S.template getValue< 2, 0,-2>(), S.template getValue<-2, 0,-2>(),
                                S.template getValue< 3, 0, 2>(), S.template getValue<-3, 0, 2>(),
                                S.template getValue< 3, 0,-2>(), S.template getValue<-3, 0,-2>(),
                                S.template getValue< 1, 0, 3>(), S.template getValue<-1, 0, 3>(),
                                S.template getValue< 1, 0,-3>(), S.template getValue<-1, 0,-3>(),
                                S.template getValue< 2, 0, 3>(), S.template getValue<-2, 0, 3>(),
                                S.template getValue< 2, 0,-3>(), S.template getValue<-2, 0,-3>(),
                                S.template getValue< 3, 0, 3>(), S.template getValue<-3, 0, 3>(),
                                S.template getValue< 3, 0,-3>(), S.template getValue<-3, 0,-3>() );
    }

    template<typename Stencil>
    static typename Stencil::ValueType inYandZ(const Stencil& S)
    {
        return crossdifference( S.template getValue< 0, 1, 1>(), S.template getValue< 0,-1, 1>(),
                                S.template getValue< 0, 1,-1>(), S.template getValue< 0,-1,-1>(),
                                S.template getValue< 0, 2, 1>(), S.template getValue< 0,-2, 1>(),
                                S.template getValue< 0, 2,-1>(), S.template getValue< 0,-2,-1>(),
                                S.template getValue< 0, 3, 1>(), S.template getValue< 0,-3, 1>(),
                                S.template getValue< 0, 3,-1>(), S.template getValue< 0,-3,-1>(),
                                S.template getValue< 0, 1, 2>(), S.template getValue< 0,-1, 2>(),
                                S.template getValue< 0, 1,-2>(), S.template getValue< 0,-1,-2>(),
                                S.template getValue< 0, 2, 2>(), S.template getValue< 0,-2, 2>(),
                                S.template getValue< 0, 2,-2>(), S.template getValue< 0,-2,-2>(),
                                S.template getValue< 0, 3, 2>(), S.template getValue< 0,-3, 2>(),
                                S.template getValue< 0, 3,-2>(), S.template getValue< 0,-3,-2>(),
                                S.template getValue< 0, 1, 3>(), S.template getValue< 0,-1, 3>(),
                                S.template getValue< 0, 1,-3>(), S.template getValue< 0,-1,-3>(),
                                S.template getValue< 0, 2, 3>(), S.template getValue< 0,-2, 3>(),
                                S.template getValue< 0, 2,-3>(), S.template getValue< 0,-2,-3>(),
                                S.template getValue< 0, 3, 3>(), S.template getValue< 0,-3, 3>(),
                                S.template getValue< 0, 3,-3>(), S.template getValue< 0,-3,-3>() );
    }

};

} // end math namespace
} // namespace OPENVDB_VERSION_NAME
} // end openvdb namespace

#endif // OPENVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
