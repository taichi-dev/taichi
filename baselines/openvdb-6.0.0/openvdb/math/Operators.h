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

/// @file math/Operators.h

#ifndef OPENVDB_MATH_OPERATORS_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_OPERATORS_HAS_BEEN_INCLUDED

#include "FiniteDifference.h"
#include "Stencils.h"
#include "Maps.h"
#include "Transform.h"
#include <cmath> // for std::sqrt()


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

// Simple tools to help determine when type conversions are needed
template<typename Vec3T> struct is_vec3d { static const bool value = false; };
template<> struct is_vec3d<Vec3d>        { static const bool value = true; };

template<typename T> struct is_double    { static const bool value = false; };
template<> struct is_double<double>      { static const bool value = true; };


/// @brief Adapter to associate a map with a world-space operator,
/// giving it the same call signature as an index-space operator
/// @todo For now, the operator's result type must be specified explicitly,
/// but eventually it should be possible, via traits, to derive the result type
/// from the operator type.
template<typename MapType, typename OpType, typename ResultType>
struct MapAdapter {
    MapAdapter(const MapType& m): map(m) {}

    template<typename AccessorType>
    inline ResultType
    result(const AccessorType& grid, const Coord& ijk) { return OpType::result(map, grid, ijk); }

    template<typename StencilType>
    inline ResultType
    result(const StencilType& stencil) { return OpType::result(map, stencil); }

    const MapType map;
};


/// Adapter for vector-valued index-space operators to return the vector magnitude
template<typename OpType>
struct ISOpMagnitude {
    template<typename AccessorType>
    static inline double result(const AccessorType& grid, const Coord& ijk) {
        return double(OpType::result(grid, ijk).length());
    }

    template<typename StencilType>
    static inline double result(const StencilType& stencil) {
        return double(OpType::result(stencil).length());
    }
};

/// Adapter for vector-valued world-space operators to return the vector magnitude
template<typename OpType, typename MapT>
struct OpMagnitude {
    template<typename AccessorType>
    static inline double result(const MapT& map, const AccessorType& grid, const Coord& ijk) {
        return double(OpType::result(map, grid, ijk).length());
    }

    template<typename StencilType>
    static inline double result(const MapT& map, const StencilType& stencil) {
        return double(OpType::result(map, stencil).length());
    }
};


namespace internal {

// This additional layer is necessary for Visual C++ to compile.
template<typename T>
struct ReturnValue {
    using ValueType = typename T::ValueType;
    using Vec3Type = math::Vec3<ValueType>;
};

} // namespace internal

// ---- Operators defined in index space


//@{
/// @brief Gradient operators defined in index space of various orders
template<DScheme DiffScheme>
struct ISGradient
{
    // random access version
    template<typename Accessor> static Vec3<typename Accessor::ValueType>
    result(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = Vec3<ValueType>;
        return Vec3Type( D1<DiffScheme>::inX(grid, ijk),
                         D1<DiffScheme>::inY(grid, ijk),
                         D1<DiffScheme>::inZ(grid, ijk) );
    }

    // stencil access version
    template<typename StencilT> static Vec3<typename StencilT::ValueType>
    result(const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = Vec3<ValueType>;
        return Vec3Type( D1<DiffScheme>::inX(stencil),
                         D1<DiffScheme>::inY(stencil),
                         D1<DiffScheme>::inZ(stencil) );
    }
};
//@}

/// struct that relates the BiasedGradientScheme to the
/// forward and backward difference methods used, as well as to
/// the correct stencil type for index space use
template<BiasedGradientScheme bgs>
struct BIAS_SCHEME {
    static const DScheme FD = FD_1ST;
    static const DScheme BD = BD_1ST;

    template<typename GridType, bool IsSafe = true>
    struct ISStencil {
        using StencilType = SevenPointStencil<GridType, IsSafe>;
    };
};

template<> struct BIAS_SCHEME<FIRST_BIAS>
{
    static const DScheme FD = FD_1ST;
    static const DScheme BD = BD_1ST;

    template<typename GridType, bool IsSafe = true>
    struct ISStencil {
        using StencilType = SevenPointStencil<GridType, IsSafe>;
    };
};

template<> struct BIAS_SCHEME<SECOND_BIAS>
{
    static const DScheme FD = FD_2ND;
    static const DScheme BD = BD_2ND;

    template<typename GridType, bool IsSafe = true>
    struct ISStencil {
        using StencilType = ThirteenPointStencil<GridType, IsSafe>;
      };
};
template<> struct BIAS_SCHEME<THIRD_BIAS>
{
    static const DScheme FD = FD_3RD;
    static const DScheme BD = BD_3RD;

    template<typename GridType, bool IsSafe = true>
    struct ISStencil {
        using StencilType = NineteenPointStencil<GridType, IsSafe>;
    };
};
template<> struct BIAS_SCHEME<WENO5_BIAS>
{
    static const DScheme FD = FD_WENO5;
    static const DScheme BD = BD_WENO5;

    template<typename GridType, bool IsSafe = true>
    struct ISStencil {
        using StencilType = NineteenPointStencil<GridType, IsSafe>;
    };
};
template<> struct BIAS_SCHEME<HJWENO5_BIAS>
{
    static const DScheme FD = FD_HJWENO5;
    static const DScheme BD = BD_HJWENO5;

    template<typename GridType, bool IsSafe = true>
    struct ISStencil {
        using StencilType = NineteenPointStencil<GridType, IsSafe>;
    };
};


//@{
/// @brief Biased Gradient Operators, using upwinding defined by the @c Vec3Bias input

template<BiasedGradientScheme GradScheme, typename Vec3Bias>
struct ISGradientBiased
{
    static const DScheme FD = BIAS_SCHEME<GradScheme>::FD;
    static const DScheme BD = BIAS_SCHEME<GradScheme>::BD;

    // random access version
    template<typename Accessor>
    static Vec3<typename Accessor::ValueType>
    result(const Accessor& grid, const Coord& ijk, const Vec3Bias& V)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = Vec3<ValueType>;

        return Vec3Type(V[0]<0 ? D1<FD>::inX(grid,ijk) : D1<BD>::inX(grid,ijk),
                        V[1]<0 ? D1<FD>::inY(grid,ijk) : D1<BD>::inY(grid,ijk),
                        V[2]<0 ? D1<FD>::inZ(grid,ijk) : D1<BD>::inZ(grid,ijk) );
    }

    // stencil access version
    template<typename StencilT>
    static Vec3<typename StencilT::ValueType>
    result(const StencilT& stencil, const Vec3Bias& V)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = Vec3<ValueType>;

        return Vec3Type(V[0]<0 ? D1<FD>::inX(stencil) : D1<BD>::inX(stencil),
                        V[1]<0 ? D1<FD>::inY(stencil) : D1<BD>::inY(stencil),
                        V[2]<0 ? D1<FD>::inZ(stencil) : D1<BD>::inZ(stencil) );
    }
};


template<BiasedGradientScheme GradScheme>
struct ISGradientNormSqrd
{
    static const DScheme FD = BIAS_SCHEME<GradScheme>::FD;
    static const DScheme BD = BIAS_SCHEME<GradScheme>::BD;


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType
    result(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = math::Vec3<ValueType>;

        Vec3Type up   = ISGradient<FD>::result(grid, ijk);
        Vec3Type down = ISGradient<BD>::result(grid, ijk);
        return math::GodunovsNormSqrd(grid.getValue(ijk)>0, down, up);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType
    result(const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = math::Vec3<ValueType>;

        Vec3Type up   = ISGradient<FD>::result(stencil);
        Vec3Type down = ISGradient<BD>::result(stencil);
        return math::GodunovsNormSqrd(stencil.template getValue<0, 0, 0>()>0, down, up);
    }
};

#ifdef DWA_OPENVDB  // for SIMD - note will do the computations in float
template<>
struct ISGradientNormSqrd<HJWENO5_BIAS>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const Accessor& grid, const Coord& ijk)
    {
        struct GetValue
        {
            const Accessor& acc;
            GetValue(const Accessor& acc_): acc(acc_) {}
            // Return the grid value at ijk converted to simd::Float4::value_type (= float).
            inline simd::Float4::value_type operator()(const Coord& ijk_) {
                return static_cast<simd::Float4::value_type>(acc.getValue(ijk_));
            }
        }
        valueAt(grid);

        // SSE optimized
        const simd::Float4
            v1(valueAt(ijk.offsetBy(-2, 0, 0)) - valueAt(ijk.offsetBy(-3, 0, 0)),
               valueAt(ijk.offsetBy( 0,-2, 0)) - valueAt(ijk.offsetBy( 0,-3, 0)),
               valueAt(ijk.offsetBy( 0, 0,-2)) - valueAt(ijk.offsetBy( 0, 0,-3)), 0),
            v2(valueAt(ijk.offsetBy(-1, 0, 0)) - valueAt(ijk.offsetBy(-2, 0, 0)),
               valueAt(ijk.offsetBy( 0,-1, 0)) - valueAt(ijk.offsetBy( 0,-2, 0)),
               valueAt(ijk.offsetBy( 0, 0,-1)) - valueAt(ijk.offsetBy( 0, 0,-2)), 0),
            v3(valueAt(ijk                   ) - valueAt(ijk.offsetBy(-1, 0, 0)),
               valueAt(ijk                   ) - valueAt(ijk.offsetBy( 0,-1, 0)),
               valueAt(ijk                   ) - valueAt(ijk.offsetBy( 0, 0,-1)), 0),
            v4(valueAt(ijk.offsetBy( 1, 0, 0)) - valueAt(ijk                   ),
               valueAt(ijk.offsetBy( 0, 1, 0)) - valueAt(ijk                   ),
               valueAt(ijk.offsetBy( 0, 0, 1)) - valueAt(ijk                   ), 0),
            v5(valueAt(ijk.offsetBy( 2, 0, 0)) - valueAt(ijk.offsetBy( 1, 0, 0)),
               valueAt(ijk.offsetBy( 0, 2, 0)) - valueAt(ijk.offsetBy( 0, 1, 0)),
               valueAt(ijk.offsetBy( 0, 0, 2)) - valueAt(ijk.offsetBy( 0, 0, 1)), 0),
            v6(valueAt(ijk.offsetBy( 3, 0, 0)) - valueAt(ijk.offsetBy( 2, 0, 0)),
               valueAt(ijk.offsetBy( 0, 3, 0)) - valueAt(ijk.offsetBy( 0, 2, 0)),
               valueAt(ijk.offsetBy( 0, 0, 3)) - valueAt(ijk.offsetBy( 0, 0, 2)), 0),
            down = math::WENO5(v1, v2, v3, v4, v5),
            up   = math::WENO5(v6, v5, v4, v3, v2);

        return math::GodunovsNormSqrd(grid.getValue(ijk)>0, down, up);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const StencilT& s)
    {
        using F4Val = simd::Float4::value_type;

        // SSE optimized
        const simd::Float4
            v1(F4Val(s.template getValue<-2, 0, 0>()) - F4Val(s.template getValue<-3, 0, 0>()),
               F4Val(s.template getValue< 0,-2, 0>()) - F4Val(s.template getValue< 0,-3, 0>()),
               F4Val(s.template getValue< 0, 0,-2>()) - F4Val(s.template getValue< 0, 0,-3>()), 0),
            v2(F4Val(s.template getValue<-1, 0, 0>()) - F4Val(s.template getValue<-2, 0, 0>()),
               F4Val(s.template getValue< 0,-1, 0>()) - F4Val(s.template getValue< 0,-2, 0>()),
               F4Val(s.template getValue< 0, 0,-1>()) - F4Val(s.template getValue< 0, 0,-2>()), 0),
            v3(F4Val(s.template getValue< 0, 0, 0>()) - F4Val(s.template getValue<-1, 0, 0>()),
               F4Val(s.template getValue< 0, 0, 0>()) - F4Val(s.template getValue< 0,-1, 0>()),
               F4Val(s.template getValue< 0, 0, 0>()) - F4Val(s.template getValue< 0, 0,-1>()), 0),
            v4(F4Val(s.template getValue< 1, 0, 0>()) - F4Val(s.template getValue< 0, 0, 0>()),
               F4Val(s.template getValue< 0, 1, 0>()) - F4Val(s.template getValue< 0, 0, 0>()),
               F4Val(s.template getValue< 0, 0, 1>()) - F4Val(s.template getValue< 0, 0, 0>()), 0),
            v5(F4Val(s.template getValue< 2, 0, 0>()) - F4Val(s.template getValue< 1, 0, 0>()),
               F4Val(s.template getValue< 0, 2, 0>()) - F4Val(s.template getValue< 0, 1, 0>()),
               F4Val(s.template getValue< 0, 0, 2>()) - F4Val(s.template getValue< 0, 0, 1>()), 0),
            v6(F4Val(s.template getValue< 3, 0, 0>()) - F4Val(s.template getValue< 2, 0, 0>()),
               F4Val(s.template getValue< 0, 3, 0>()) - F4Val(s.template getValue< 0, 2, 0>()),
               F4Val(s.template getValue< 0, 0, 3>()) - F4Val(s.template getValue< 0, 0, 2>()), 0),
            down = math::WENO5(v1, v2, v3, v4, v5),
            up   = math::WENO5(v6, v5, v4, v3, v2);

        return math::GodunovsNormSqrd(s.template getValue<0, 0, 0>()>0, down, up);
    }
};
#endif //DWA_OPENVDB  // for SIMD - note will do the computations in float
//@}


//@{
/// @brief Laplacian defined in index space, using various center-difference stencils
template<DDScheme DiffScheme>
struct ISLaplacian
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const Accessor& grid, const Coord& ijk);

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const StencilT& stencil);
};


template<>
struct ISLaplacian<CD_SECOND>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const Accessor& grid, const Coord& ijk)
    {
        return  grid.getValue(ijk.offsetBy(1,0,0)) + grid.getValue(ijk.offsetBy(-1, 0, 0)) +
                grid.getValue(ijk.offsetBy(0,1,0)) + grid.getValue(ijk.offsetBy(0, -1, 0)) +
                grid.getValue(ijk.offsetBy(0,0,1)) + grid.getValue(ijk.offsetBy(0,  0,-1))
                                                   - 6*grid.getValue(ijk);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const StencilT& stencil)
    {
        return  stencil.template getValue< 1, 0, 0>() + stencil.template getValue<-1, 0, 0>() +
                stencil.template getValue< 0, 1, 0>() + stencil.template getValue< 0,-1, 0>() +
                stencil.template getValue< 0, 0, 1>() + stencil.template getValue< 0, 0,-1>()
                                                   - 6*stencil.template getValue< 0, 0, 0>();
    }
};

template<>
struct ISLaplacian<CD_FOURTH>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const Accessor& grid, const Coord& ijk)
    {
        using ValueT = typename Accessor::ValueType;
        return static_cast<ValueT>(
            (-1./12.)*(
                grid.getValue(ijk.offsetBy(2,0,0)) + grid.getValue(ijk.offsetBy(-2, 0, 0)) +
                grid.getValue(ijk.offsetBy(0,2,0)) + grid.getValue(ijk.offsetBy( 0,-2, 0)) +
                grid.getValue(ijk.offsetBy(0,0,2)) + grid.getValue(ijk.offsetBy( 0, 0,-2)) )
            + (4./3.)*(
                grid.getValue(ijk.offsetBy(1,0,0)) + grid.getValue(ijk.offsetBy(-1, 0, 0)) +
                grid.getValue(ijk.offsetBy(0,1,0)) + grid.getValue(ijk.offsetBy( 0,-1, 0)) +
                grid.getValue(ijk.offsetBy(0,0,1)) + grid.getValue(ijk.offsetBy( 0, 0,-1)) )
            - 7.5*grid.getValue(ijk));
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const StencilT& stencil)
    {
        using ValueT = typename StencilT::ValueType;
        return static_cast<ValueT>(
            (-1./12.)*(
                stencil.template getValue< 2, 0, 0>() + stencil.template getValue<-2, 0, 0>() +
                stencil.template getValue< 0, 2, 0>() + stencil.template getValue< 0,-2, 0>() +
                stencil.template getValue< 0, 0, 2>() + stencil.template getValue< 0, 0,-2>() )
            + (4./3.)*(
                stencil.template getValue< 1, 0, 0>() + stencil.template getValue<-1, 0, 0>() +
                stencil.template getValue< 0, 1, 0>() + stencil.template getValue< 0,-1, 0>() +
                stencil.template getValue< 0, 0, 1>() + stencil.template getValue< 0, 0,-1>() )
            - 7.5*stencil.template getValue< 0, 0, 0>());
    }
};

template<>
struct ISLaplacian<CD_SIXTH>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const Accessor& grid, const Coord& ijk)
    {
        using ValueT = typename Accessor::ValueType;
        return static_cast<ValueT>(
            (1./90.)*(
                grid.getValue(ijk.offsetBy(3,0,0)) + grid.getValue(ijk.offsetBy(-3, 0, 0)) +
                grid.getValue(ijk.offsetBy(0,3,0)) + grid.getValue(ijk.offsetBy( 0,-3, 0)) +
                grid.getValue(ijk.offsetBy(0,0,3)) + grid.getValue(ijk.offsetBy( 0, 0,-3)) )
            - (3./20.)*(
                grid.getValue(ijk.offsetBy(2,0,0)) + grid.getValue(ijk.offsetBy(-2, 0, 0)) +
                grid.getValue(ijk.offsetBy(0,2,0)) + grid.getValue(ijk.offsetBy( 0,-2, 0)) +
                grid.getValue(ijk.offsetBy(0,0,2)) + grid.getValue(ijk.offsetBy( 0, 0,-2)) )
            + 1.5 *(
                grid.getValue(ijk.offsetBy(1,0,0)) + grid.getValue(ijk.offsetBy(-1, 0, 0)) +
                grid.getValue(ijk.offsetBy(0,1,0)) + grid.getValue(ijk.offsetBy( 0,-1, 0)) +
                grid.getValue(ijk.offsetBy(0,0,1)) + grid.getValue(ijk.offsetBy( 0, 0,-1)) )
            - (3*49/18.)*grid.getValue(ijk));
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const StencilT& stencil)
    {
        using ValueT = typename StencilT::ValueType;
        return static_cast<ValueT>(
            (1./90.)*(
                stencil.template getValue< 3, 0, 0>() + stencil.template getValue<-3, 0, 0>() +
                stencil.template getValue< 0, 3, 0>() + stencil.template getValue< 0,-3, 0>() +
                stencil.template getValue< 0, 0, 3>() + stencil.template getValue< 0, 0,-3>() )
            - (3./20.)*(
                stencil.template getValue< 2, 0, 0>() + stencil.template getValue<-2, 0, 0>() +
                stencil.template getValue< 0, 2, 0>() + stencil.template getValue< 0,-2, 0>() +
                stencil.template getValue< 0, 0, 2>() + stencil.template getValue< 0, 0,-2>() )
            + 1.5 *(
                stencil.template getValue< 1, 0, 0>() + stencil.template getValue<-1, 0, 0>() +
                stencil.template getValue< 0, 1, 0>() + stencil.template getValue< 0,-1, 0>() +
                stencil.template getValue< 0, 0, 1>() + stencil.template getValue< 0, 0,-1>() )
            - (3*49/18.)*stencil.template getValue< 0, 0, 0>());
    }
};
//@}


//@{
/// Divergence operator defined in index space using various first derivative schemes
template<DScheme DiffScheme>
struct ISDivergence
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const Accessor& grid, const Coord& ijk)
    {
        return D1Vec<DiffScheme>::inX(grid, ijk, 0) +
               D1Vec<DiffScheme>::inY(grid, ijk, 1) +
               D1Vec<DiffScheme>::inZ(grid, ijk, 2);
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const StencilT& stencil)
    {
        return D1Vec<DiffScheme>::inX(stencil, 0) +
               D1Vec<DiffScheme>::inY(stencil, 1) +
               D1Vec<DiffScheme>::inZ(stencil, 2);
    }
};
//@}


//@{
/// Curl operator defined in index space using various first derivative schemes
template<DScheme DiffScheme>
struct ISCurl
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename Accessor::ValueType;
        return Vec3Type( D1Vec<DiffScheme>::inY(grid, ijk, 2) - //dw/dy - dv/dz
                         D1Vec<DiffScheme>::inZ(grid, ijk, 1),
                         D1Vec<DiffScheme>::inZ(grid, ijk, 0) - //du/dz - dw/dx
                         D1Vec<DiffScheme>::inX(grid, ijk, 2),
                         D1Vec<DiffScheme>::inX(grid, ijk, 1) - //dv/dx - du/dy
                         D1Vec<DiffScheme>::inY(grid, ijk, 0) );
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const StencilT& stencil)
    {
        using Vec3Type = typename StencilT::ValueType;
        return Vec3Type( D1Vec<DiffScheme>::inY(stencil, 2) - //dw/dy - dv/dz
                         D1Vec<DiffScheme>::inZ(stencil, 1),
                         D1Vec<DiffScheme>::inZ(stencil, 0) - //du/dz - dw/dx
                         D1Vec<DiffScheme>::inX(stencil, 2),
                         D1Vec<DiffScheme>::inX(stencil, 1) - //dv/dx - du/dy
                         D1Vec<DiffScheme>::inY(stencil, 0) );
    }
};
//@}


//@{
/// Compute the mean curvature in index space
template<DDScheme DiffScheme2, DScheme DiffScheme1>
struct ISMeanCurvature
{
    /// @brief Random access version
    /// @return @c true if the gradient is nonzero, in which case the mean curvature
    /// is returned in two parts, @a alpha and @a beta, where @a alpha is the numerator
    /// in &nabla; &middot; (&nabla;&Phi; / |&nabla;&Phi;|) and @a beta is |&nabla;&Phi;|.
    template<typename Accessor>
    static bool result(const Accessor& grid, const Coord& ijk,
                       typename Accessor::ValueType& alpha,
                       typename Accessor::ValueType& beta)
    {
        using ValueType = typename Accessor::ValueType;

        const ValueType Dx = D1<DiffScheme1>::inX(grid, ijk);
        const ValueType Dy = D1<DiffScheme1>::inY(grid, ijk);
        const ValueType Dz = D1<DiffScheme1>::inZ(grid, ijk);

        const ValueType Dx2 = Dx*Dx;
        const ValueType Dy2 = Dy*Dy;
        const ValueType Dz2 = Dz*Dz;
        const ValueType normGrad = Dx2 + Dy2 + Dz2;
        if (normGrad <= math::Tolerance<ValueType>::value()) {
            alpha = beta = 0;
            return false;
        }

        const ValueType Dxx = D2<DiffScheme2>::inX(grid, ijk);
        const ValueType Dyy = D2<DiffScheme2>::inY(grid, ijk);
        const ValueType Dzz = D2<DiffScheme2>::inZ(grid, ijk);

        const ValueType Dxy = D2<DiffScheme2>::inXandY(grid, ijk);
        const ValueType Dyz = D2<DiffScheme2>::inYandZ(grid, ijk);
        const ValueType Dxz = D2<DiffScheme2>::inXandZ(grid, ijk);

        // for return
        alpha = (Dx2*(Dyy+Dzz)+Dy2*(Dxx+Dzz)+Dz2*(Dxx+Dyy)-2*(Dx*(Dy*Dxy+Dz*Dxz)+Dy*Dz*Dyz));
        beta  = ValueType(std::sqrt(double(normGrad))); // * 1/dx
        return true;
    }

    /// @brief Stencil access version
    /// @return @c true if the gradient is nonzero, in which case the mean curvature
    /// is returned in two parts, @a alpha and @a beta, where @a alpha is the numerator
    /// in &nabla; &middot; (&nabla;&Phi; / |&nabla;&Phi;|) and @a beta is |&nabla;&Phi;|.
    template<typename StencilT>
    static bool result(const StencilT& stencil,
                       typename StencilT::ValueType& alpha,
                       typename StencilT::ValueType& beta)
    {
        using ValueType = typename StencilT::ValueType;
        const ValueType Dx = D1<DiffScheme1>::inX(stencil);
        const ValueType Dy = D1<DiffScheme1>::inY(stencil);
        const ValueType Dz = D1<DiffScheme1>::inZ(stencil);

        const ValueType Dx2 = Dx*Dx;
        const ValueType Dy2 = Dy*Dy;
        const ValueType Dz2 = Dz*Dz;
        const ValueType normGrad = Dx2 + Dy2 + Dz2;
        if (normGrad <= math::Tolerance<ValueType>::value()) {
            alpha = beta = 0;
            return false;
        }

        const ValueType Dxx = D2<DiffScheme2>::inX(stencil);
        const ValueType Dyy = D2<DiffScheme2>::inY(stencil);
        const ValueType Dzz = D2<DiffScheme2>::inZ(stencil);

        const ValueType Dxy = D2<DiffScheme2>::inXandY(stencil);
        const ValueType Dyz = D2<DiffScheme2>::inYandZ(stencil);
        const ValueType Dxz = D2<DiffScheme2>::inXandZ(stencil);

        // for return
        alpha = (Dx2*(Dyy+Dzz)+Dy2*(Dxx+Dzz)+Dz2*(Dxx+Dyy)-2*(Dx*(Dy*Dxy+Dz*Dxz)+Dy*Dz*Dyz));
        beta = ValueType(std::sqrt(double(normGrad))); // * 1/dx
        return true;
    }
};

////////////////////////////////////////////////////////

// --- Operators defined in the Range of a given map

//@{
/// @brief Center difference gradient operators, defined with respect to
/// the range-space of the @c map
/// @note This will need to be divided by two in the case of CD_2NDT
template<typename MapType, DScheme DiffScheme>
struct Gradient
{
    // random access version
    template<typename Accessor>
    static typename internal::ReturnValue<Accessor>::Vec3Type
    result(const MapType& map, const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename internal::ReturnValue<Accessor>::Vec3Type;

        Vec3d iGradient( ISGradient<DiffScheme>::result(grid, ijk) );
        return Vec3Type(map.applyIJT(iGradient, ijk.asVec3d()));
    }

    // stencil access version
    template<typename StencilT>
    static typename internal::ReturnValue<StencilT>::Vec3Type
    result(const MapType& map, const StencilT& stencil)
    {
        using Vec3Type = typename internal::ReturnValue<StencilT>::Vec3Type;

        Vec3d iGradient( ISGradient<DiffScheme>::result(stencil) );
        return Vec3Type(map.applyIJT(iGradient, stencil.getCenterCoord().asVec3d()));
    }
};

// Partial template specialization of Gradient
// translation, any order
template<DScheme DiffScheme>
struct Gradient<TranslationMap, DiffScheme>
{
    // random access version
    template<typename Accessor>
    static typename internal::ReturnValue<Accessor>::Vec3Type
    result(const TranslationMap&, const Accessor& grid, const Coord& ijk)
    {
        return ISGradient<DiffScheme>::result(grid, ijk);
    }

    // stencil access version
    template<typename StencilT>
    static typename internal::ReturnValue<StencilT>::Vec3Type
    result(const TranslationMap&, const StencilT& stencil)
    {
        return ISGradient<DiffScheme>::result(stencil);
    }
};

/// Full template specialization of Gradient
/// uniform scale, 2nd order
template<>
struct Gradient<UniformScaleMap, CD_2ND>
{
    // random access version
    template<typename Accessor>
    static typename internal::ReturnValue<Accessor>::Vec3Type
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename internal::ReturnValue<Accessor>::ValueType;
        using Vec3Type = typename internal::ReturnValue<Accessor>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(grid, ijk) );
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return  iGradient * inv2dx;
    }

    // stencil access version
    template<typename StencilT>
    static typename internal::ReturnValue<StencilT>::Vec3Type
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename internal::ReturnValue<StencilT>::ValueType;
        using Vec3Type = typename internal::ReturnValue<StencilT>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(stencil) );
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return  iGradient * inv2dx;
    }
};

/// Full template specialization of Gradient
/// uniform scale translate, 2nd order
template<>
struct Gradient<UniformScaleTranslateMap, CD_2ND>
{
    // random access version
    template<typename Accessor>
    static typename internal::ReturnValue<Accessor>::Vec3Type
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename internal::ReturnValue<Accessor>::ValueType;
        using Vec3Type = typename internal::ReturnValue<Accessor>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(grid, ijk) );
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return  iGradient * inv2dx;
    }

    // stencil access version
    template<typename StencilT>
    static typename internal::ReturnValue<StencilT>::Vec3Type
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename internal::ReturnValue<StencilT>::ValueType;
        using Vec3Type = typename internal::ReturnValue<StencilT>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(stencil) );
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return  iGradient * inv2dx;
    }
};

/// Full template specialization of Gradient
/// scale, 2nd order
template<>
struct Gradient<ScaleMap, CD_2ND>
{
    // random access version
    template<typename Accessor>
    static typename internal::ReturnValue<Accessor>::Vec3Type
    result(const ScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename internal::ReturnValue<Accessor>::ValueType;
        using Vec3Type = typename internal::ReturnValue<Accessor>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(grid, ijk) );
        return  Vec3Type(ValueType(iGradient[0] * map.getInvTwiceScale()[0]),
                         ValueType(iGradient[1] * map.getInvTwiceScale()[1]),
                         ValueType(iGradient[2] * map.getInvTwiceScale()[2]) );
    }

    // stencil access version
    template<typename StencilT>
    static typename internal::ReturnValue<StencilT>::Vec3Type
    result(const ScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename internal::ReturnValue<StencilT>::ValueType;
        using Vec3Type = typename internal::ReturnValue<StencilT>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(stencil) );
        return  Vec3Type(ValueType(iGradient[0] * map.getInvTwiceScale()[0]),
                         ValueType(iGradient[1] * map.getInvTwiceScale()[1]),
                         ValueType(iGradient[2] * map.getInvTwiceScale()[2]) );
    }
};

/// Full template specialization of Gradient
/// scale translate, 2nd order
template<>
struct Gradient<ScaleTranslateMap, CD_2ND>
{
    // random access version
    template<typename Accessor>
    static typename internal::ReturnValue<Accessor>::Vec3Type
    result(const ScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename internal::ReturnValue<Accessor>::ValueType;
        using Vec3Type = typename internal::ReturnValue<Accessor>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(grid, ijk) );
        return  Vec3Type(ValueType(iGradient[0] * map.getInvTwiceScale()[0]),
                         ValueType(iGradient[1] * map.getInvTwiceScale()[1]),
                         ValueType(iGradient[2] * map.getInvTwiceScale()[2]) );
    }

    // Stencil access version
    template<typename StencilT>
    static typename internal::ReturnValue<StencilT>::Vec3Type
    result(const ScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename internal::ReturnValue<StencilT>::ValueType;
        using Vec3Type = typename internal::ReturnValue<StencilT>::Vec3Type;

        Vec3Type iGradient( ISGradient<CD_2NDT>::result(stencil) );
        return  Vec3Type(ValueType(iGradient[0] * map.getInvTwiceScale()[0]),
                         ValueType(iGradient[1] * map.getInvTwiceScale()[1]),
                         ValueType(iGradient[2] * map.getInvTwiceScale()[2]) );
    }
};
//@}


//@{
/// @brief Biased gradient operators, defined with respect to the range-space of the map
/// @note This will need to be divided by two in the case of CD_2NDT
template<typename MapType, BiasedGradientScheme GradScheme>
struct GradientBiased
{
    // random access version
    template<typename Accessor> static math::Vec3<typename Accessor::ValueType>
    result(const MapType& map, const Accessor& grid, const Coord& ijk,
           const Vec3<typename Accessor::ValueType>& V)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = math::Vec3<ValueType>;

        Vec3d iGradient( ISGradientBiased<GradScheme, Vec3Type>::result(grid, ijk, V) );
        return Vec3Type(map.applyIJT(iGradient, ijk.asVec3d()));
    }

    // stencil access version
    template<typename StencilT> static math::Vec3<typename StencilT::ValueType>
    result(const MapType& map, const StencilT& stencil,
           const Vec3<typename StencilT::ValueType>& V)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = math::Vec3<ValueType>;

        Vec3d iGradient( ISGradientBiased<GradScheme, Vec3Type>::result(stencil, V) );
        return Vec3Type(map.applyIJT(iGradient, stencil.getCenterCoord().asVec3d()));
    }
};
//@}


////////////////////////////////////////////////////////

// Computes |Grad[Phi]| using upwinding
template<typename MapType, BiasedGradientScheme GradScheme>
struct GradientNormSqrd
{
    static const DScheme FD = BIAS_SCHEME<GradScheme>::FD;
    static const DScheme BD = BIAS_SCHEME<GradScheme>::BD;


    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType
    result(const MapType& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = math::Vec3<ValueType>;

        Vec3Type up   = Gradient<MapType, FD>::result(map, grid, ijk);
        Vec3Type down = Gradient<MapType, BD>::result(map, grid, ijk);
        return math::GodunovsNormSqrd(grid.getValue(ijk)>0, down, up);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType
    result(const MapType& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = math::Vec3<ValueType>;

        Vec3Type up   = Gradient<MapType, FD>::result(map, stencil);
        Vec3Type down = Gradient<MapType, BD>::result(map, stencil);
        return math::GodunovsNormSqrd(stencil.template getValue<0, 0, 0>()>0, down, up);
    }
};

/// Partial template specialization of GradientNormSqrd
template<BiasedGradientScheme GradScheme>
struct GradientNormSqrd<UniformScaleMap, GradScheme>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return invdxdx * ISGradientNormSqrd<GradScheme>::result(grid, ijk);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return invdxdx * ISGradientNormSqrd<GradScheme>::result(stencil);
    }
};

/// Partial template specialization of GradientNormSqrd
template<BiasedGradientScheme GradScheme>
struct GradientNormSqrd<UniformScaleTranslateMap, GradScheme>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return invdxdx * ISGradientNormSqrd<GradScheme>::result(grid, ijk);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return invdxdx * ISGradientNormSqrd<GradScheme>::result(stencil);
    }
};


//@{
/// @brief Compute the divergence of a vector-valued grid using differencing
/// of various orders, the result defined with respect to the range-space of the map.
template<typename MapType, DScheme DiffScheme>
struct Divergence
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const MapType& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div(0);
        for (int i=0; i < 3; i++) {
            Vec3d vec( D1Vec<DiffScheme>::inX(grid, ijk, i),
                       D1Vec<DiffScheme>::inY(grid, ijk, i),
                       D1Vec<DiffScheme>::inZ(grid, ijk, i) );
            div += ValueType(map.applyIJT(vec, ijk.asVec3d())[i]);
        }
        return div;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const MapType& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);
        for (int i=0; i < 3; i++) {
            Vec3d vec( D1Vec<DiffScheme>::inX(stencil, i),
                       D1Vec<DiffScheme>::inY(stencil, i),
                       D1Vec<DiffScheme>::inZ(stencil, i) );
            div += ValueType(map.applyIJT(vec, stencil.getCenterCoord().asVec3d())[i]);
        }
        return div;
    }
};

/// Partial template specialization of Divergence
/// translation, any scheme
template<DScheme DiffScheme>
struct Divergence<TranslationMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const TranslationMap&, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div(0);
        div =ISDivergence<DiffScheme>::result(grid, ijk);
        return div;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const TranslationMap&, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);
        div =ISDivergence<DiffScheme>::result(stencil);
        return div;
    }
};

/// Partial template specialization of Divergence
/// uniform scale, any scheme
template<DScheme DiffScheme>
struct Divergence<UniformScaleMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div(0);

        div =ISDivergence<DiffScheme>::result(grid, ijk);
        ValueType invdx = ValueType(map.getInvScale()[0]);
        return div * invdx;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);

        div =ISDivergence<DiffScheme>::result(stencil);
        ValueType invdx = ValueType(map.getInvScale()[0]);
        return div * invdx;
    }
};

/// Partial template specialization of Divergence
/// uniform scale and translation, any scheme
template<DScheme DiffScheme>
struct Divergence<UniformScaleTranslateMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div(0);

        div =ISDivergence<DiffScheme>::result(grid, ijk);
        ValueType invdx = ValueType(map.getInvScale()[0]);
        return div * invdx;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);

        div =ISDivergence<DiffScheme>::result(stencil);
        ValueType invdx = ValueType(map.getInvScale()[0]);
        return div * invdx;
    }
};

/// Full template specialization of Divergence
/// uniform scale 2nd order
template<>
struct Divergence<UniformScaleMap, CD_2ND>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div(0);
        div =ISDivergence<CD_2NDT>::result(grid, ijk);
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return div * inv2dx;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);
        div =ISDivergence<CD_2NDT>::result(stencil);
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return div * inv2dx;
    }
};

/// Full template specialization of Divergence
/// uniform scale translate 2nd order
template<>
struct Divergence<UniformScaleTranslateMap, CD_2ND>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div(0);

        div =ISDivergence<CD_2NDT>::result(grid, ijk);
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return div * inv2dx;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);

        div =ISDivergence<CD_2NDT>::result(stencil);
        ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
        return div * inv2dx;
    }
};

/// Partial template specialization of Divergence
/// scale, any scheme
template<DScheme DiffScheme>
struct Divergence<ScaleMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const ScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div = ValueType(
            D1Vec<DiffScheme>::inX(grid, ijk, 0) * (map.getInvScale()[0]) +
            D1Vec<DiffScheme>::inY(grid, ijk, 1) * (map.getInvScale()[1]) +
            D1Vec<DiffScheme>::inZ(grid, ijk, 2) * (map.getInvScale()[2]));
        return div;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const ScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);
        div = ValueType(
              D1Vec<DiffScheme>::inX(stencil, 0) * (map.getInvScale()[0]) +
              D1Vec<DiffScheme>::inY(stencil, 1) * (map.getInvScale()[1]) +
              D1Vec<DiffScheme>::inZ(stencil, 2) * (map.getInvScale()[2]) );
        return div;
    }
};

/// Partial template specialization of Divergence
/// scale translate, any scheme
template<DScheme DiffScheme>
struct Divergence<ScaleTranslateMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const ScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div = ValueType(
            D1Vec<DiffScheme>::inX(grid, ijk, 0) * (map.getInvScale()[0]) +
            D1Vec<DiffScheme>::inY(grid, ijk, 1) * (map.getInvScale()[1]) +
            D1Vec<DiffScheme>::inZ(grid, ijk, 2) * (map.getInvScale()[2]));
        return div;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const ScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div(0);
        div = ValueType(
              D1Vec<DiffScheme>::inX(stencil, 0) * (map.getInvScale()[0]) +
              D1Vec<DiffScheme>::inY(stencil, 1) * (map.getInvScale()[1]) +
              D1Vec<DiffScheme>::inZ(stencil, 2) * (map.getInvScale()[2]) );
        return div;
    }
};

/// Full template specialization Divergence
/// scale 2nd order
template<>
struct Divergence<ScaleMap, CD_2ND>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const ScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div = ValueType(
            D1Vec<CD_2NDT>::inX(grid, ijk, 0) * (map.getInvTwiceScale()[0]) +
            D1Vec<CD_2NDT>::inY(grid, ijk, 1) * (map.getInvTwiceScale()[1]) +
            D1Vec<CD_2NDT>::inZ(grid, ijk, 2) * (map.getInvTwiceScale()[2]) );
        return div;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const ScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div = ValueType(
            D1Vec<CD_2NDT>::inX(stencil, 0) * (map.getInvTwiceScale()[0]) +
            D1Vec<CD_2NDT>::inY(stencil, 1) * (map.getInvTwiceScale()[1]) +
            D1Vec<CD_2NDT>::inZ(stencil, 2) * (map.getInvTwiceScale()[2]) );
        return div;
    }
};

/// Full template specialization of Divergence
/// scale and translate, 2nd order
template<>
struct Divergence<ScaleTranslateMap, CD_2ND>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType::value_type
    result(const ScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType::value_type;

        ValueType div = ValueType(
            D1Vec<CD_2NDT>::inX(grid, ijk, 0) * (map.getInvTwiceScale()[0]) +
            D1Vec<CD_2NDT>::inY(grid, ijk, 1) * (map.getInvTwiceScale()[1]) +
            D1Vec<CD_2NDT>::inZ(grid, ijk, 2) * (map.getInvTwiceScale()[2]) );
        return div;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType::value_type
    result(const ScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType::value_type;

        ValueType div = ValueType(
            D1Vec<CD_2NDT>::inX(stencil, 0) * (map.getInvTwiceScale()[0]) +
            D1Vec<CD_2NDT>::inY(stencil, 1) * (map.getInvTwiceScale()[1]) +
            D1Vec<CD_2NDT>::inZ(stencil, 2) * (map.getInvTwiceScale()[2]) );
        return div;
    }
};
//@}


//@{
/// @brief Compute the curl of a vector-valued grid using differencing
/// of various orders in the space defined by the range of the map.
template<typename MapType, DScheme DiffScheme>
struct Curl
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const MapType& map, const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename Accessor::ValueType;
        Vec3Type mat[3];
        for (int i = 0; i < 3; i++) {
            Vec3d vec(
                D1Vec<DiffScheme>::inX(grid, ijk, i),
                D1Vec<DiffScheme>::inY(grid, ijk, i),
                D1Vec<DiffScheme>::inZ(grid, ijk, i));
            // dF_i/dx_j   (x_1 = x, x_2 = y,  x_3 = z)
            mat[i] = Vec3Type(map.applyIJT(vec, ijk.asVec3d()));
        }
        return Vec3Type(mat[2][1] - mat[1][2], // dF_3/dx_2 - dF_2/dx_3
                        mat[0][2] - mat[2][0], // dF_1/dx_3 - dF_3/dx_1
                        mat[1][0] - mat[0][1]); // dF_2/dx_1 - dF_1/dx_2
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const MapType& map, const StencilT& stencil)
    {
        using Vec3Type = typename StencilT::ValueType;
        Vec3Type mat[3];
        for (int i = 0; i < 3; i++) {
            Vec3d vec(
                D1Vec<DiffScheme>::inX(stencil, i),
                D1Vec<DiffScheme>::inY(stencil, i),
                D1Vec<DiffScheme>::inZ(stencil, i));
            // dF_i/dx_j   (x_1 = x, x_2 = y,  x_3 = z)
            mat[i] = Vec3Type(map.applyIJT(vec, stencil.getCenterCoord().asVec3d()));
        }
        return Vec3Type(mat[2][1] - mat[1][2], // dF_3/dx_2 - dF_2/dx_3
                        mat[0][2] - mat[2][0], // dF_1/dx_3 - dF_3/dx_1
                        mat[1][0] - mat[0][1]); // dF_2/dx_1 - dF_1/dx_2
    }
};

/// Partial template specialization of Curl
template<DScheme DiffScheme>
struct Curl<UniformScaleMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename Accessor::ValueType;
        using ValueType = typename Vec3Type::value_type;
        return ISCurl<DiffScheme>::result(grid, ijk) * ValueType(map.getInvScale()[0]);
    }

    // Stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
         using Vec3Type = typename StencilT::ValueType;
         using ValueType = typename Vec3Type::value_type;
         return ISCurl<DiffScheme>::result(stencil) * ValueType(map.getInvScale()[0]);
     }
};

/// Partial template specialization of Curl
template<DScheme DiffScheme>
struct Curl<UniformScaleTranslateMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename Accessor::ValueType;
        using ValueType = typename Vec3Type::value_type;

        return ISCurl<DiffScheme>::result(grid, ijk) * ValueType(map.getInvScale()[0]);
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using Vec3Type = typename StencilT::ValueType;
        using ValueType = typename Vec3Type::value_type;

        return ISCurl<DiffScheme>::result(stencil) * ValueType(map.getInvScale()[0]);
    }
};

/// Full template specialization of Curl
template<>
struct Curl<UniformScaleMap, CD_2ND>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename Accessor::ValueType;
        using ValueType = typename Vec3Type::value_type;

        return ISCurl<CD_2NDT>::result(grid, ijk) * ValueType(map.getInvTwiceScale()[0]);
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using Vec3Type = typename StencilT::ValueType;
        using ValueType = typename Vec3Type::value_type;

        return ISCurl<CD_2NDT>::result(stencil) * ValueType(map.getInvTwiceScale()[0]);
    }
};

/// Full template specialization of Curl
template<>
struct Curl<UniformScaleTranslateMap, CD_2ND>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using Vec3Type = typename Accessor::ValueType;
        using ValueType = typename Vec3Type::value_type;

        return ISCurl<CD_2NDT>::result(grid, ijk) * ValueType(map.getInvTwiceScale()[0]);
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using Vec3Type = typename StencilT::ValueType;
        using ValueType = typename Vec3Type::value_type;

        return ISCurl<CD_2NDT>::result(stencil) * ValueType(map.getInvTwiceScale()[0]);
    }
};
//@}


//@{
/// @brief Compute the Laplacian at a given location in a grid using finite differencing
/// of various orders.  The result is defined in the range of the map.
template<typename MapType, DDScheme DiffScheme>
struct Laplacian
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const MapType& map,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        // all the second derivatives in index space
        ValueType iddx  = D2<DiffScheme>::inX(grid, ijk);
        ValueType iddy  = D2<DiffScheme>::inY(grid, ijk);
        ValueType iddz  = D2<DiffScheme>::inZ(grid, ijk);

        ValueType iddxy = D2<DiffScheme>::inXandY(grid, ijk);
        ValueType iddyz = D2<DiffScheme>::inYandZ(grid, ijk);
        ValueType iddxz = D2<DiffScheme>::inXandZ(grid, ijk);

        // second derivatives in index space
        Mat3d  d2_is(iddx,  iddxy, iddxz,
                     iddxy, iddy,  iddyz,
                     iddxz, iddyz, iddz);

        Mat3d d2_rs;  // to hold the second derivative matrix in range space
        if (is_linear<MapType>::value) {
            d2_rs = map.applyIJC(d2_is);
        } else {
            // compute the first derivatives with 2nd order accuracy.
            Vec3d d1_is(static_cast<double>(D1<CD_2ND>::inX(grid, ijk)),
                        static_cast<double>(D1<CD_2ND>::inY(grid, ijk)),
                        static_cast<double>(D1<CD_2ND>::inZ(grid, ijk)));

            d2_rs = map.applyIJC(d2_is, d1_is, ijk.asVec3d());
        }

        // the trace of the second derivative (range space) matrix is laplacian
        return ValueType(d2_rs(0,0) + d2_rs(1,1) + d2_rs(2,2));
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const MapType& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        // all the second derivatives in index space
        ValueType iddx  = D2<DiffScheme>::inX(stencil);
        ValueType iddy  = D2<DiffScheme>::inY(stencil);
        ValueType iddz  = D2<DiffScheme>::inZ(stencil);

        ValueType iddxy = D2<DiffScheme>::inXandY(stencil);
        ValueType iddyz = D2<DiffScheme>::inYandZ(stencil);
        ValueType iddxz = D2<DiffScheme>::inXandZ(stencil);

        // second derivatives in index space
        Mat3d  d2_is(iddx,  iddxy, iddxz,
                     iddxy, iddy,  iddyz,
                     iddxz, iddyz, iddz);

        Mat3d d2_rs;  // to hold the second derivative matrix in range space
        if (is_linear<MapType>::value) {
            d2_rs = map.applyIJC(d2_is);
        } else {
            // compute the first derivatives with 2nd order accuracy.
            Vec3d d1_is(D1<CD_2ND>::inX(stencil),
                        D1<CD_2ND>::inY(stencil),
                        D1<CD_2ND>::inZ(stencil) );

            d2_rs = map.applyIJC(d2_is, d1_is, stencil.getCenterCoord().asVec3d());
        }

        // the trace of the second derivative (range space) matrix is laplacian
        return ValueType(d2_rs(0,0) + d2_rs(1,1) + d2_rs(2,2));
    }
};


template<DDScheme DiffScheme>
struct Laplacian<TranslationMap, DiffScheme>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const TranslationMap&,
        const Accessor& grid, const Coord& ijk)
    {
        return ISLaplacian<DiffScheme>::result(grid, ijk);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const TranslationMap&, const StencilT& stencil)
    {
        return ISLaplacian<DiffScheme>::result(stencil);
    }
};


// The Laplacian is invariant to rotation or reflection.
template<DDScheme DiffScheme>
struct Laplacian<UnitaryMap, DiffScheme>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const UnitaryMap&,
        const Accessor& grid, const Coord& ijk)
    {
        return ISLaplacian<DiffScheme>::result(grid, ijk);
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const UnitaryMap&, const StencilT& stencil)
    {
        return ISLaplacian<DiffScheme>::result(stencil);
    }
};


template<DDScheme DiffScheme>
struct Laplacian<UniformScaleMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return ISLaplacian<DiffScheme>::result(grid, ijk) * invdxdx;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return ISLaplacian<DiffScheme>::result(stencil) * invdxdx;
    }
};


template<DDScheme DiffScheme>
struct Laplacian<UniformScaleTranslateMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return ISLaplacian<DiffScheme>::result(grid, ijk) * invdxdx;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
        return ISLaplacian<DiffScheme>::result(stencil) * invdxdx;
    }
};


template<DDScheme DiffScheme>
struct Laplacian<ScaleMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const ScaleMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        // compute the second derivatives in index space
        ValueType iddx = D2<DiffScheme>::inX(grid, ijk);
        ValueType iddy = D2<DiffScheme>::inY(grid, ijk);
        ValueType iddz = D2<DiffScheme>::inZ(grid, ijk);
        const Vec3d& invScaleSqr = map.getInvScaleSqr();
        // scale them by the appropriate 1/dx^2, 1/dy^2, 1/dz^2 and sum
        return ValueType(iddx * invScaleSqr[0] + iddy * invScaleSqr[1] + iddz * invScaleSqr[2]);
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const ScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        // compute the second derivatives in index space
        ValueType iddx = D2<DiffScheme>::inX(stencil);
        ValueType iddy = D2<DiffScheme>::inY(stencil);
        ValueType iddz = D2<DiffScheme>::inZ(stencil);
        const Vec3d& invScaleSqr = map.getInvScaleSqr();
        // scale them by the appropriate 1/dx^2, 1/dy^2, 1/dz^2 and sum
        return ValueType(iddx * invScaleSqr[0] + iddy * invScaleSqr[1] + iddz * invScaleSqr[2]);
    }
};


template<DDScheme DiffScheme>
struct Laplacian<ScaleTranslateMap, DiffScheme>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const ScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        // compute the second derivatives in index space
        ValueType iddx = D2<DiffScheme>::inX(grid, ijk);
        ValueType iddy = D2<DiffScheme>::inY(grid, ijk);
        ValueType iddz = D2<DiffScheme>::inZ(grid, ijk);
        const Vec3d& invScaleSqr = map.getInvScaleSqr();
        // scale them by the appropriate 1/dx^2, 1/dy^2, 1/dz^2 and sum
        return ValueType(iddx * invScaleSqr[0] + iddy * invScaleSqr[1] + iddz * invScaleSqr[2]);
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const ScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        // compute the second derivatives in index space
        ValueType iddx = D2<DiffScheme>::inX(stencil);
        ValueType iddy = D2<DiffScheme>::inY(stencil);
        ValueType iddz = D2<DiffScheme>::inZ(stencil);
        const Vec3d& invScaleSqr = map.getInvScaleSqr();
        // scale them by the appropriate 1/dx^2, 1/dy^2, 1/dz^2 and sum
        return ValueType(iddx * invScaleSqr[0] + iddy * invScaleSqr[1] + iddz * invScaleSqr[2]);
    }
};


/// @brief Compute the closest-point transform to a level set.
/// @return the closest point to the surface from which the level set was derived,
/// in the domain space of the map (e.g., voxel space).
template<typename MapType, DScheme DiffScheme>
struct CPT
{
    // random access version
    template<typename Accessor> static math::Vec3<typename Accessor::ValueType>
    result(const MapType& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = Vec3<ValueType>;

        // current distance
        ValueType d = grid.getValue(ijk);
        // compute gradient in physical space where it is a unit normal
        // since the grid holds a distance level set.
        Vec3d vectorFromSurface(d*Gradient<MapType,DiffScheme>::result(map, grid, ijk));
        if (is_linear<MapType>::value) {
            Vec3d result = ijk.asVec3d() - map.applyInverseMap(vectorFromSurface);
            return Vec3Type(result);
        } else {
            Vec3d location = map.applyMap(ijk.asVec3d());
            Vec3d result = map.applyInverseMap(location - vectorFromSurface);
            return Vec3Type(result);
        }
    }

    // stencil access version
    template<typename StencilT> static math::Vec3<typename StencilT::ValueType>
    result(const MapType& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = Vec3<ValueType>;

        // current distance
        ValueType d = stencil.template getValue<0, 0, 0>();
        // compute gradient in physical space where it is a unit normal
        // since the grid holds a distance level set.
        Vec3d vectorFromSurface(d*Gradient<MapType, DiffScheme>::result(map, stencil));
        if (is_linear<MapType>::value) {
            Vec3d result = stencil.getCenterCoord().asVec3d()
                - map.applyInverseMap(vectorFromSurface);
            return Vec3Type(result);
        } else {
            Vec3d location = map.applyMap(stencil.getCenterCoord().asVec3d());
            Vec3d result = map.applyInverseMap(location - vectorFromSurface);
            return Vec3Type(result);
        }
    }
};


/// @brief Compute the closest-point transform to a level set.
/// @return the closest point to the surface from which the level set was derived,
/// in the range space of the map (e.g., in world space)
template<typename MapType, DScheme DiffScheme>
struct CPT_RANGE
{
    // random access version
    template<typename Accessor> static Vec3<typename Accessor::ValueType>
    result(const MapType& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        using Vec3Type = Vec3<ValueType>;
        // current distance
        ValueType d = grid.getValue(ijk);
        // compute gradient in physical space where it is a unit normal
        // since the grid holds a distance level set.
        Vec3Type vectorFromSurface =
            d*Gradient<MapType,DiffScheme>::result(map, grid, ijk);
        Vec3d result = map.applyMap(ijk.asVec3d()) - vectorFromSurface;

        return Vec3Type(result);
    }

    // stencil access version
    template<typename StencilT> static Vec3<typename StencilT::ValueType>
    result(const MapType& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;
        using Vec3Type = Vec3<ValueType>;
        // current distance
        ValueType d = stencil.template getValue<0, 0, 0>();
        // compute gradient in physical space where it is a unit normal
        // since the grid holds a distance level set.
        Vec3Type vectorFromSurface =
            d*Gradient<MapType, DiffScheme>::result(map, stencil);
        Vec3d result = map.applyMap(stencil.getCenterCoord().asVec3d()) - vectorFromSurface;

        return Vec3Type(result);
    }
};


/// @brief Compute the mean curvature.
/// @details The mean curvature is returned in two parts, @a alpha and @a beta,
/// where @a alpha is the numerator in &nabla; &middot; (&nabla;&Phi; / |&nabla;&Phi;|)
/// and @a beta is |&nabla;&Phi;|.
template<typename MapType, DDScheme DiffScheme2, DScheme DiffScheme1>
struct MeanCurvature
{
    /// @brief Random access version
    /// @return @c true if the gradient is nonzero, in which case the mean curvature
    /// is returned in two parts, @a alpha and @a beta, where @a alpha is the numerator
    /// in &nabla; &middot; (&nabla;&Phi; / |&nabla;&Phi;|) and @a beta is |&nabla;&Phi;|.
    template<typename Accessor>
    static bool compute(const MapType& map, const Accessor& grid, const Coord& ijk,
                        double& alpha, double& beta)
    {
        using ValueType = typename Accessor::ValueType;

         // compute the gradient in index and world space
         Vec3d d1_is(static_cast<double>(D1<DiffScheme1>::inX(grid, ijk)),
                     static_cast<double>(D1<DiffScheme1>::inY(grid, ijk)),
                     static_cast<double>(D1<DiffScheme1>::inZ(grid, ijk))), d1_ws;
         if (is_linear<MapType>::value) {//resolved at compiletime
             d1_ws = map.applyIJT(d1_is);
         } else {
             d1_ws = map.applyIJT(d1_is, ijk.asVec3d());
         }
         const double Dx2 = d1_ws(0)*d1_ws(0);
         const double Dy2 = d1_ws(1)*d1_ws(1);
         const double Dz2 = d1_ws(2)*d1_ws(2);
         const double normGrad = Dx2 + Dy2 + Dz2;
         if (normGrad <= math::Tolerance<double>::value()) {
             alpha = beta = 0;
             return false;
         }

         // all the second derivatives in index space
         ValueType iddx  = D2<DiffScheme2>::inX(grid, ijk);
         ValueType iddy  = D2<DiffScheme2>::inY(grid, ijk);
         ValueType iddz  = D2<DiffScheme2>::inZ(grid, ijk);

         ValueType iddxy = D2<DiffScheme2>::inXandY(grid, ijk);
         ValueType iddyz = D2<DiffScheme2>::inYandZ(grid, ijk);
         ValueType iddxz = D2<DiffScheme2>::inXandZ(grid, ijk);

         // second derivatives in index space
         Mat3d  d2_is(iddx,  iddxy, iddxz,
                      iddxy, iddy,  iddyz,
                      iddxz, iddyz, iddz);

         // convert second derivatives to world space
         Mat3d d2_ws;
         if (is_linear<MapType>::value) {//resolved at compiletime
             d2_ws = map.applyIJC(d2_is);
         } else {
             d2_ws = map.applyIJC(d2_is, d1_is, ijk.asVec3d());
         }

         // assemble the nominator and denominator for mean curvature
         alpha = (Dx2*(d2_ws(1,1)+d2_ws(2,2))+Dy2*(d2_ws(0,0)+d2_ws(2,2))
                  +Dz2*(d2_ws(0,0)+d2_ws(1,1))
                  -2*(d1_ws(0)*(d1_ws(1)*d2_ws(0,1)+d1_ws(2)*d2_ws(0,2))
                      +d1_ws(1)*d1_ws(2)*d2_ws(1,2)));
         beta = std::sqrt(normGrad); // * 1/dx
         return true;
    }

    template<typename Accessor>
    static typename Accessor::ValueType result(const MapType& map,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        double alpha, beta;
        return compute(map, grid, ijk, alpha, beta) ?
               ValueType(alpha/(2. *math::Pow3(beta))) : 0;
    }

    template<typename Accessor>
    static typename Accessor::ValueType normGrad(const MapType& map,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        double alpha, beta;
        return compute(map, grid, ijk, alpha, beta) ?
               ValueType(alpha/(2. *math::Pow2(beta))) : 0;
    }

    /// @brief Stencil access version
    /// @return @c true if the gradient is nonzero, in which case the mean curvature
    /// is returned in two parts, @a alpha and @a beta, where @a alpha is the numerator
    /// in &nabla; &middot; (&nabla;&Phi; / |&nabla;&Phi;|) and @a beta is |&nabla;&Phi;|.
    template<typename StencilT>
    static bool compute(const MapType& map, const StencilT& stencil,
                        double& alpha, double& beta)
    {
        using ValueType = typename StencilT::ValueType;

         // compute the gradient in index and world space
         Vec3d d1_is(D1<DiffScheme1>::inX(stencil),
                     D1<DiffScheme1>::inY(stencil),
                     D1<DiffScheme1>::inZ(stencil) ), d1_ws;
         if (is_linear<MapType>::value) {//resolved at compiletime
             d1_ws = map.applyIJT(d1_is);
         } else {
             d1_ws = map.applyIJT(d1_is, stencil.getCenterCoord().asVec3d());
         }
         const double Dx2 = d1_ws(0)*d1_ws(0);
         const double Dy2 = d1_ws(1)*d1_ws(1);
         const double Dz2 = d1_ws(2)*d1_ws(2);
         const double normGrad = Dx2 + Dy2 + Dz2;
         if (normGrad <= math::Tolerance<double>::value()) {
             alpha = beta = 0;
             return false;
         }

         // all the second derivatives in index space
         ValueType iddx  = D2<DiffScheme2>::inX(stencil);
         ValueType iddy  = D2<DiffScheme2>::inY(stencil);
         ValueType iddz  = D2<DiffScheme2>::inZ(stencil);

         ValueType iddxy = D2<DiffScheme2>::inXandY(stencil);
         ValueType iddyz = D2<DiffScheme2>::inYandZ(stencil);
         ValueType iddxz = D2<DiffScheme2>::inXandZ(stencil);

         // second derivatives in index space
         Mat3d  d2_is(iddx,  iddxy, iddxz,
                      iddxy, iddy,  iddyz,
                      iddxz, iddyz, iddz);

         // convert second derivatives to world space
         Mat3d d2_ws;
         if (is_linear<MapType>::value) {//resolved at compiletime
             d2_ws = map.applyIJC(d2_is);
         } else {
             d2_ws = map.applyIJC(d2_is, d1_is, stencil.getCenterCoord().asVec3d());
         }

         // for return
         alpha = (Dx2*(d2_ws(1,1)+d2_ws(2,2))+Dy2*(d2_ws(0,0)+d2_ws(2,2))
                  +Dz2*(d2_ws(0,0)+d2_ws(1,1))
                  -2*(d1_ws(0)*(d1_ws(1)*d2_ws(0,1)+d1_ws(2)*d2_ws(0,2))
                      +d1_ws(1)*d1_ws(2)*d2_ws(1,2)));
         beta  = std::sqrt(normGrad); // * 1/dx
         return true;
    }

    template<typename StencilT>
    static typename StencilT::ValueType
    result(const MapType& map, const StencilT stencil)
    {
        using ValueType = typename StencilT::ValueType;
        double alpha, beta;
        return compute(map, stencil, alpha, beta) ?
               ValueType(alpha/(2*math::Pow3(beta))) : 0;
    }

    template<typename StencilT>
    static typename StencilT::ValueType normGrad(const MapType& map, const StencilT stencil)
    {
        using ValueType = typename StencilT::ValueType;
        double alpha, beta;
        return compute(map, stencil, alpha, beta) ?
               ValueType(alpha/(2*math::Pow2(beta))) : 0;
    }
};


template<DDScheme DiffScheme2, DScheme DiffScheme1>
struct MeanCurvature<TranslationMap, DiffScheme2, DiffScheme1>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const TranslationMap&,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType alpha, beta;
        return ISMeanCurvature<DiffScheme2, DiffScheme1>::result(grid, ijk, alpha, beta) ?
               ValueType(alpha /(2*math::Pow3(beta))) : 0;
    }

    template<typename Accessor>
    static typename Accessor::ValueType normGrad(const TranslationMap&,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType alpha, beta;
        return ISMeanCurvature<DiffScheme2, DiffScheme1>::result(grid, ijk, alpha, beta) ?
               ValueType(alpha/(2*math::Pow2(beta))) : 0;
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const TranslationMap&, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType alpha, beta;
        return ISMeanCurvature<DiffScheme2, DiffScheme1>::result(stencil, alpha, beta) ?
               ValueType(alpha /(2*math::Pow3(beta))) : 0;
    }

    template<typename StencilT>
    static typename StencilT::ValueType normGrad(const TranslationMap&, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType alpha, beta;
        return ISMeanCurvature<DiffScheme2, DiffScheme1>::result(stencil, alpha, beta) ?
               ValueType(alpha/(2*math::Pow2(beta))) : 0;
    }
};


template<DDScheme DiffScheme2, DScheme DiffScheme1>
struct MeanCurvature<UniformScaleMap, DiffScheme2, DiffScheme1>
{
    // random access version
    template<typename Accessor>
    static typename Accessor::ValueType result(const UniformScaleMap& map,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(grid, ijk, alpha, beta)) {
            ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
            return ValueType(alpha*inv2dx/math::Pow3(beta));
        }
        return 0;
    }

    template<typename Accessor>
    static typename Accessor::ValueType normGrad(const UniformScaleMap& map,
        const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(grid, ijk, alpha, beta)) {
            ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
            return ValueType(alpha*invdxdx/(2*math::Pow2(beta)));
        }
        return 0;
    }

    // stencil access version
    template<typename StencilT>
    static typename StencilT::ValueType result(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(stencil, alpha, beta)) {
            ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
            return ValueType(alpha*inv2dx/math::Pow3(beta));
        }
        return 0;
    }

    template<typename StencilT>
    static typename StencilT::ValueType normGrad(const UniformScaleMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(stencil, alpha, beta)) {
            ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
            return ValueType(alpha*invdxdx/(2*math::Pow2(beta)));
        }
        return 0;
    }
};


template<DDScheme DiffScheme2, DScheme DiffScheme1>
struct MeanCurvature<UniformScaleTranslateMap, DiffScheme2, DiffScheme1>
{
    // random access version
    template<typename Accessor> static typename Accessor::ValueType
    result(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(grid, ijk, alpha, beta)) {
            ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
            return ValueType(alpha*inv2dx/math::Pow3(beta));
        }
        return 0;
    }

    template<typename Accessor> static typename Accessor::ValueType
    normGrad(const UniformScaleTranslateMap& map, const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(grid, ijk, alpha, beta)) {
            ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
            return ValueType(alpha*invdxdx/(2*math::Pow2(beta)));
        }
        return 0;
    }

    // stencil access version
    template<typename StencilT> static typename StencilT::ValueType
    result(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(stencil, alpha, beta)) {
            ValueType inv2dx = ValueType(map.getInvTwiceScale()[0]);
            return ValueType(alpha*inv2dx/math::Pow3(beta));
        }
        return 0;
    }

    template<typename StencilT> static typename StencilT::ValueType
    normGrad(const UniformScaleTranslateMap& map, const StencilT& stencil)
    {
        using ValueType = typename StencilT::ValueType;

        ValueType alpha, beta;
        if (ISMeanCurvature<DiffScheme2, DiffScheme1>::result(stencil, alpha, beta)) {
            ValueType invdxdx = ValueType(map.getInvScaleSqr()[0]);
            return ValueType(alpha*invdxdx/(2*math::Pow2(beta)));
        }
        return 0;
    }
};


/// @brief A wrapper that holds a MapBase::ConstPtr and exposes a reduced set
/// of functionality needed by the mathematical operators
/// @details This may be used in some <tt>Map</tt>-templated code, when the overhead of
/// actually resolving the @c Map type is large compared to the map work to be done.
class GenericMap
{
public:
    template<typename GridType>
    GenericMap(const GridType& g): mMap(g.transform().baseMap()) {}

    GenericMap(const Transform& t): mMap(t.baseMap()) {}
    GenericMap(MapBase::Ptr map): mMap(ConstPtrCast<const MapBase>(map)) {}
    GenericMap(MapBase::ConstPtr map): mMap(map) {}
    ~GenericMap() {}

    Vec3d applyMap(const Vec3d& in) const { return mMap->applyMap(in); }
    Vec3d applyInverseMap(const Vec3d& in) const { return mMap->applyInverseMap(in); }

    Vec3d applyIJT(const Vec3d& in) const { return mMap->applyIJT(in); }
    Vec3d applyIJT(const Vec3d& in, const Vec3d& pos) const { return mMap->applyIJT(in, pos); }
    Mat3d applyIJC(const Mat3d& m) const { return mMap->applyIJC(m); }
    Mat3d applyIJC(const Mat3d& m, const Vec3d& v, const Vec3d& pos) const
        { return mMap->applyIJC(m,v,pos); }

    double determinant() const { return mMap->determinant(); }
    double determinant(const Vec3d& in) const { return mMap->determinant(in); }

    Vec3d voxelSize() const { return mMap->voxelSize(); }
    Vec3d voxelSize(const Vec3d&v) const { return mMap->voxelSize(v); }

private:
    MapBase::ConstPtr mMap;
};

} // end math namespace
} // namespace OPENVDB_VERSION_NAME
} // end openvdb namespace

#endif // OPENVDB_MATH_OPERATORS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
