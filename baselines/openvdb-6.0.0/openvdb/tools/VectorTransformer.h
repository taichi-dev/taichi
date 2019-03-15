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
//
/// @file VectorTransformer.h

#ifndef OPENVDB_TOOLS_VECTORTRANSFORMER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VECTORTRANSFORMER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/math/Mat4.h>
#include <openvdb/math/Vec3.h>
#include "ValueTransformer.h" // for tools::foreach()
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Apply an affine transform to the voxel values of a vector-valued grid
/// in accordance with the grid's vector type (covariant, contravariant, etc.).
/// @throw TypeError if the grid is not vector-valued
template<typename GridType>
inline void
transformVectors(GridType&, const Mat4d&);


////////////////////////////////////////


// Functors for use with tools::foreach() to transform vector voxel values

struct HomogeneousMatMul
{
    const Mat4d mat;
    HomogeneousMatMul(const Mat4d& _mat): mat(_mat) {}
    template<typename TreeIterT> void operator()(const TreeIterT& it) const
    {
        Vec3d v(*it);
        it.setValue(mat.transformH(v));
    }
};

struct MatMul
{
    const Mat4d mat;
    MatMul(const Mat4d& _mat): mat(_mat) {}
    template<typename TreeIterT>
    void operator()(const TreeIterT& it) const
    {
        Vec3d v(*it);
        it.setValue(mat.transform3x3(v));
    }
};

struct MatMulNormalize
{
    const Mat4d mat;
    MatMulNormalize(const Mat4d& _mat): mat(_mat) {}
    template<typename TreeIterT>
    void operator()(const TreeIterT& it) const
    {
        Vec3d v(*it);
        v = mat.transform3x3(v);
        v.normalize();
        it.setValue(v);
    }
};


//{
/// @cond OPENVDB_VECTOR_TRANSFORMER_INTERNAL

/// @internal This overload is enabled only for scalar-valued grids.
template<typename GridType> inline
typename std::enable_if<!VecTraits<typename GridType::ValueType>::IsVec, void>::type
doTransformVectors(GridType&, const Mat4d&)
{
    OPENVDB_THROW(TypeError, "tools::transformVectors() requires a vector-valued grid");
}

/// @internal This overload is enabled only for vector-valued grids.
template<typename GridType> inline
typename std::enable_if<VecTraits<typename GridType::ValueType>::IsVec, void>::type
doTransformVectors(GridType& grid, const Mat4d& mat)
{
    if (!grid.isInWorldSpace()) return;

    const VecType vecType = grid.getVectorType();
    switch (vecType) {
        case VEC_COVARIANT:
        case VEC_COVARIANT_NORMALIZE:
        {
            Mat4d invmat = mat.inverse();
            invmat = invmat.transpose();

            if (vecType == VEC_COVARIANT_NORMALIZE) {
                foreach(grid.beginValueAll(), MatMulNormalize(invmat));
            } else {
                foreach(grid.beginValueAll(), MatMul(invmat));
            }
            break;
        }

        case VEC_CONTRAVARIANT_RELATIVE:
            foreach(grid.beginValueAll(), MatMul(mat));
            break;

        case VEC_CONTRAVARIANT_ABSOLUTE:
            foreach(grid.beginValueAll(), HomogeneousMatMul(mat));
            break;

        case VEC_INVARIANT:
            break;
    }
}

/// @endcond
//}


template<typename GridType>
inline void
transformVectors(GridType& grid, const Mat4d& mat)
{
    doTransformVectors<GridType>(grid, mat);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VECTORTRANSFORMER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
