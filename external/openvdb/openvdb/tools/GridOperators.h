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

/// @file tools/GridOperators.h
///
/// @brief Apply an operator to an input grid to produce an output grid
/// with the same active voxel topology but a potentially different value type.

#ifndef OPENVDB_TOOLS_GRID_OPERATORS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_GRID_OPERATORS_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/Operators.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/ValueAccessor.h>
#include "ValueTransformer.h" // for tools::foreach()
#include <tbb/parallel_for.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief VectorToScalarConverter<VectorGridType>::Type is the type of a grid
/// having the same tree configuration as VectorGridType but a scalar value type, T,
/// where T is the type of the original vector components.
/// @details For example, VectorToScalarConverter<Vec3DGrid>::Type is equivalent to DoubleGrid.
template<typename VectorGridType> struct VectorToScalarConverter {
    typedef typename VectorGridType::ValueType::value_type VecComponentValueT;
    typedef typename VectorGridType::template ValueConverter<VecComponentValueT>::Type Type;
};

/// @brief ScalarToVectorConverter<ScalarGridType>::Type is the type of a grid
/// having the same tree configuration as ScalarGridType but value type Vec3<T>
/// where T is ScalarGridType::ValueType.
/// @details For example, ScalarToVectorConverter<DoubleGrid>::Type is equivalent to Vec3DGrid.
template<typename ScalarGridType> struct ScalarToVectorConverter {
    typedef math::Vec3<typename ScalarGridType::ValueType> VectorValueT;
    typedef typename ScalarGridType::template ValueConverter<VectorValueT>::Type Type;
};


/// @brief Compute the Closest-Point Transform (CPT) from a distance field.
/// @return a new vector-valued grid with the same numerical precision as the input grid
///     (for example, if the input grid is a DoubleGrid, the output grid will be a Vec3DGrid)
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
cpt(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
cpt(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
cpt(const GridType& grid, bool threaded = true)
{
    return cpt<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
cpt(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return cpt<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Compute the curl of the given vector-valued grid.
/// @return a new vector-valued grid
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
curl(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
curl(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename GridType::Ptr
curl(const GridType& grid, bool threaded = true)
{
    return curl<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename GridType::Ptr
curl(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return curl<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Compute the divergence of the given vector-valued grid.
/// @return a new scalar-valued grid with the same numerical precision as the input grid
///     (for example, if the input grid is a Vec3DGrid, the output grid will be a DoubleGrid)
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
divergence(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
divergence(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
divergence(const GridType& grid, bool threaded = true)
{
    return divergence<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
divergence(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return divergence<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Compute the gradient of the given scalar grid.
/// @return a new vector-valued grid with the same numerical precision as the input grid
///     (for example, if the input grid is a DoubleGrid, the output grid will be a Vec3DGrid)
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
gradient(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
gradient(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
gradient(const GridType& grid, bool threaded = true)
{
    return gradient<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
gradient(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return gradient<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Compute the Laplacian of the given scalar grid.
/// @return a new scalar grid
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
laplacian(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
laplacian(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename GridType::Ptr
laplacian(const GridType& grid, bool threaded = true)
{
    return laplacian<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename GridType::Ptr
laplacian(const GridType& grid, const MaskT mask, bool threaded = true)
{
    return laplacian<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Compute the mean curvature of the given grid.
/// @return a new grid
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
meanCurvature(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
meanCurvature(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename GridType::Ptr
meanCurvature(const GridType& grid, bool threaded = true)
{
    return meanCurvature<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename GridType::Ptr
meanCurvature(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return meanCurvature<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Compute the magnitudes of the vectors of the given vector-valued grid.
/// @return a new scalar-valued grid with the same numerical precision as the input grid
///     (for example, if the input grid is a Vec3DGrid, the output grid will be a DoubleGrid)
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
magnitude(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
magnitude(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
magnitude(const GridType& grid, bool threaded = true)
{
    return magnitude<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
magnitude(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return magnitude<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


/// @brief Normalize the vectors of the given vector-valued grid.
/// @return a new vector-valued grid
/// @details When a mask grid is specified, the solution is calculated only in
/// the intersection of the mask active topology and the input active topology
/// independent of the transforms associated with either grid.
template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
normalize(const GridType& grid, bool threaded, InterruptT* interrupt);

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
normalize(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt);

template<typename GridType> inline
typename GridType::Ptr
normalize(const GridType& grid, bool threaded = true)
{
    return normalize<GridType, util::NullInterrupter>(grid, threaded, nullptr);
}

template<typename GridType, typename MaskT> inline
typename GridType::Ptr
normalize(const GridType& grid, const MaskT& mask, bool threaded = true)
{
    return normalize<GridType, MaskT, util::NullInterrupter>(grid, mask, threaded, nullptr);
}


////////////////////////////////////////


namespace gridop {

/// @brief ToMaskGrid<T>::Type is the type of a grid having the same
/// tree hierarchy as grid type T but a value equal to its active state.
/// @details For example, ToMaskGrid<FloatGrid>::Type is equivalent to MaskGrid.
template<typename GridType>
struct ToMaskGrid {
    typedef Grid<typename GridType::TreeType::template ValueConverter<ValueMask>::Type> Type;
};


/// @brief Apply an operator to an input grid to produce an output grid
/// with the same active voxel topology but a potentially different value type.
/// @details To facilitate inlining, this class is also templated on a Map type.
///
/// @note This is a helper class and should never be used directly.
template<
    typename InGridT,
    typename MaskGridType,
    typename OutGridT,
    typename MapT,
    typename OperatorT,
    typename InterruptT = util::NullInterrupter>
class GridOperator
{
public:
    typedef typename OutGridT::TreeType           OutTreeT;
    typedef typename OutTreeT::LeafNodeType       OutLeafT;
    typedef typename tree::LeafManager<OutTreeT>  LeafManagerT;

    GridOperator(const InGridT& grid, const MaskGridType* mask, const MapT& map,
        InterruptT* interrupt = nullptr, bool densify = true)
        : mAcc(grid.getConstAccessor())
        , mMap(map)
        , mInterrupt(interrupt)
        , mMask(mask)
        , mDensify(densify) ///< @todo consider adding a "NeedsDensification" operator trait
    {
    }
    GridOperator(const GridOperator&) = default;
    GridOperator& operator=(const GridOperator&) = default;
    virtual ~GridOperator() = default;

    typename OutGridT::Ptr process(bool threaded = true)
    {
        if (mInterrupt) mInterrupt->start("Processing grid");

        // Derive background value of the output grid
        typename InGridT::TreeType tmp(mAcc.tree().background());
        typename OutGridT::ValueType backg = OperatorT::result(mMap, tmp, math::Coord(0));

        // The output tree is topology copy, optionally densified, of the input tree.
        // (Densification is necessary for some operators because applying the operator to
        // a constant tile produces distinct output values, particularly along tile borders.)
        /// @todo Can tiles be handled correctly without densification, or by densifying
        /// only to the width of the operator stencil?
        typename OutTreeT::Ptr tree(new OutTreeT(mAcc.tree(), backg, TopologyCopy()));
        if (mDensify) tree->voxelizeActiveTiles();

        // create grid with output tree and unit transform
        typename OutGridT::Ptr result(new OutGridT(tree));

        // Modify the solution area if a mask was supplied.
        if (mMask) {
            result->topologyIntersection(*mMask);
        }

        // transform of output grid = transform of input grid
        result->setTransform(math::Transform::Ptr(new math::Transform( mMap.copy() )));

        LeafManagerT leafManager(*tree);

        if (threaded) {
            tbb::parallel_for(leafManager.leafRange(), *this);
        } else {
            (*this)(leafManager.leafRange());
        }

        // If the tree wasn't densified, it might have active tiles that need to be processed.
        if (!mDensify) {
            using TileIter = typename OutTreeT::ValueOnIter;

            TileIter tileIter = tree->beginValueOn();
            tileIter.setMaxDepth(tileIter.getLeafDepth() - 1); // skip leaf values (i.e., voxels)

            AccessorT inAcc = mAcc; // each thread needs its own accessor, captured by value
            auto tileOp = [this, inAcc](const TileIter& it) {
                // Apply the operator to the input grid's tile value at the iterator's
                // current coordinates, and set the output tile's value to the result.
                it.setValue(OperatorT::result(this->mMap, inAcc, it.getCoord()));
            };

            // Apply the operator to tile values, optionally in parallel.
            // (But don't share the functor; each thread needs its own accessor.)
            tools::foreach(tileIter, tileOp, threaded, /*shareFunctor=*/false);
        }

        if (mDensify) tree->prune();

        if (mInterrupt) mInterrupt->end();
        return result;
    }

    /// @brief Iterate sequentially over LeafNodes and voxels in the output
    /// grid and apply the operator using a value accessor for the input grid.
    ///
    /// @note Never call this public method directly - it is called by
    /// TBB threads only!
    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        if (util::wasInterrupted(mInterrupt)) tbb::task::self().cancel_group_execution();

        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {
            for (typename OutLeafT::ValueOnIter value=leaf->beginValueOn(); value; ++value) {
                value.setValue(OperatorT::result(mMap, mAcc, value.getCoord()));
            }
        }
    }

protected:
    typedef typename InGridT::ConstAccessor  AccessorT;
    mutable AccessorT   mAcc;
    const MapT&         mMap;
    InterruptT*         mInterrupt;
    const MaskGridType* mMask;
    const bool          mDensify;
}; // end of GridOperator class

} // namespace gridop


////////////////////////////////////////


/// @brief Compute the closest-point transform of a scalar grid.
template<
    typename InGridT,
    typename MaskGridType = typename gridop::ToMaskGrid<InGridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Cpt
{
public:
    typedef InGridT                                         InGridType;
    typedef typename ScalarToVectorConverter<InGridT>::Type OutGridType;

    Cpt(const InGridType& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Cpt(const InGridType& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename OutGridType::Ptr process(bool threaded = true, bool useWorldTransform = true)
    {
        Functor functor(mInputGrid, mMask, threaded, useWorldTransform, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        if (functor.mOutputGrid) functor.mOutputGrid->setVectorType(VEC_CONTRAVARIANT_ABSOLUTE);
        return functor.mOutputGrid;
    }

private:
    struct IsOpT
    {
        template<typename MapT, typename AccT>
        static typename OutGridType::ValueType
        result(const MapT& map, const AccT& acc, const Coord& xyz)
        {
            return math::CPT<MapT, math::CD_2ND>::result(map, acc, xyz);
        }
    };
    struct WsOpT
    {
        template<typename MapT, typename AccT>
        static typename OutGridType::ValueType
        result(const MapT& map, const AccT& acc, const Coord& xyz)
        {
            return math::CPT_RANGE<MapT, math::CD_2ND>::result(map, acc, xyz);
        }
    };
    struct Functor
    {
        Functor(const InGridType& grid, const MaskGridType* mask,
            bool threaded, bool worldspace, InterruptT* interrupt)
        : mThreaded(threaded)
        , mWorldSpace(worldspace)
        , mInputGrid(grid)
        , mInterrupt(interrupt)
        , mMask(mask)
        {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            if (mWorldSpace) {
                gridop::GridOperator<InGridType, MaskGridType, OutGridType, MapT, WsOpT, InterruptT>
                    op(mInputGrid, mMask, map, mInterrupt, /*densify=*/false);
                mOutputGrid = op.process(mThreaded); // cache the result
            } else {
                gridop::GridOperator<InGridType, MaskGridType, OutGridType, MapT, IsOpT, InterruptT>
                    op(mInputGrid, mMask, map, mInterrupt, /*densify=*/false);
                mOutputGrid = op.process(mThreaded); // cache the result
            }
        }
        const bool                mThreaded;
        const bool                mWorldSpace;
        const InGridType&         mInputGrid;
        typename OutGridType::Ptr mOutputGrid;
        InterruptT*               mInterrupt;
        const MaskGridType*       mMask;
    };
    const InGridType&   mInputGrid;
    InterruptT*         mInterrupt;
    const MaskGridType* mMask;
}; // end of Cpt class


////////////////////////////////////////


/// @brief Compute the curl of a vector grid.
template<
    typename GridT,
    typename MaskGridType = typename gridop::ToMaskGrid<GridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Curl
{
public:
    typedef GridT  InGridType;
    typedef GridT  OutGridType;

    Curl(const GridT& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Curl(const GridT& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename GridT::Ptr process(bool threaded = true)
    {
        Functor functor(mInputGrid, mMask, threaded, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        if (functor.mOutputGrid) functor.mOutputGrid->setVectorType(VEC_COVARIANT);
        return functor.mOutputGrid;
    }

private:
    struct Functor
    {
        Functor(const GridT& grid, const MaskGridType* mask,
                bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            typedef math::Curl<MapT, math::CD_2ND> OpT;
            gridop::GridOperator<GridT, MaskGridType, GridT, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool           mThreaded;
        const GridT&         mInputGrid;
        typename GridT::Ptr  mOutputGrid;
        InterruptT*          mInterrupt;
        const MaskGridType*  mMask;
    }; // Private Functor

    const GridT&         mInputGrid;
    InterruptT*          mInterrupt;
    const MaskGridType*  mMask;
}; // end of Curl class


////////////////////////////////////////


/// @brief Compute the divergence of a vector grid.
template<
    typename InGridT,
    typename MaskGridType = typename gridop::ToMaskGrid<InGridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Divergence
{
public:
    typedef InGridT                                         InGridType;
    typedef typename VectorToScalarConverter<InGridT>::Type OutGridType;

    Divergence(const InGridT& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Divergence(const InGridT& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename OutGridType::Ptr process(bool threaded = true)
    {
        if (mInputGrid.getGridClass() == GRID_STAGGERED) {
            Functor<math::FD_1ST> functor(mInputGrid, mMask, threaded, mInterrupt);
            processTypedMap(mInputGrid.transform(), functor);
            return functor.mOutputGrid;
        } else {
            Functor<math::CD_2ND> functor(mInputGrid, mMask, threaded, mInterrupt);
            processTypedMap(mInputGrid.transform(), functor);
            return functor.mOutputGrid;
        }
    }

protected:
    template<math::DScheme DiffScheme>
    struct Functor
    {
        Functor(const InGridT& grid, const MaskGridType* mask,
            bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            typedef math::Divergence<MapT, DiffScheme> OpT;
            gridop::GridOperator<InGridType, MaskGridType, OutGridType, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool                 mThreaded;
        const InGridType&          mInputGrid;
        typename OutGridType::Ptr  mOutputGrid;
        InterruptT*                mInterrupt;
        const MaskGridType*        mMask;
    }; // Private Functor

    const InGridType&    mInputGrid;
    InterruptT*          mInterrupt;
    const MaskGridType*  mMask;
}; // end of Divergence class


////////////////////////////////////////


/// @brief Compute the gradient of a scalar grid.
template<
    typename InGridT,
    typename MaskGridType = typename gridop::ToMaskGrid<InGridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Gradient
{
public:
    typedef InGridT                                         InGridType;
    typedef typename ScalarToVectorConverter<InGridT>::Type OutGridType;

    Gradient(const InGridT& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Gradient(const InGridT& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename OutGridType::Ptr process(bool threaded = true)
    {
        Functor functor(mInputGrid, mMask, threaded, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        if (functor.mOutputGrid) functor.mOutputGrid->setVectorType(VEC_COVARIANT);
        return functor.mOutputGrid;
    }

protected:
    struct Functor
    {
        Functor(const InGridT& grid, const MaskGridType* mask,
            bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            typedef math::Gradient<MapT, math::CD_2ND> OpT;
            gridop::GridOperator<InGridType, MaskGridType, OutGridType, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool                 mThreaded;
        const InGridT&             mInputGrid;
        typename OutGridType::Ptr  mOutputGrid;
        InterruptT*                mInterrupt;
        const MaskGridType*        mMask;
    }; // Private Functor

    const InGridT&       mInputGrid;
    InterruptT*          mInterrupt;
    const MaskGridType*  mMask;
}; // end of Gradient class


////////////////////////////////////////


template<
    typename GridT,
    typename MaskGridType = typename gridop::ToMaskGrid<GridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Laplacian
{
public:
    typedef GridT  InGridType;
    typedef GridT  OutGridType;

    Laplacian(const GridT& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Laplacian(const GridT& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename GridT::Ptr process(bool threaded = true)
    {
        Functor functor(mInputGrid, mMask, threaded, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        if (functor.mOutputGrid) functor.mOutputGrid->setVectorType(VEC_COVARIANT);
        return functor.mOutputGrid;
    }

protected:
    struct Functor
    {
        Functor(const GridT& grid, const MaskGridType* mask, bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            typedef math::Laplacian<MapT, math::CD_SECOND> OpT;
            gridop::GridOperator<GridT, MaskGridType, GridT, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool           mThreaded;
        const GridT&         mInputGrid;
        typename GridT::Ptr  mOutputGrid;
        InterruptT*          mInterrupt;
        const MaskGridType*  mMask;
    }; // Private Functor

    const GridT&        mInputGrid;
    InterruptT*         mInterrupt;
    const MaskGridType* mMask;
}; // end of Laplacian class


////////////////////////////////////////


template<
    typename GridT,
    typename MaskGridType = typename gridop::ToMaskGrid<GridT>::Type,
    typename InterruptT = util::NullInterrupter>
class MeanCurvature
{
public:
    typedef GridT  InGridType;
    typedef GridT  OutGridType;

    MeanCurvature(const GridT& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    MeanCurvature(const GridT& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename GridT::Ptr process(bool threaded = true)
    {
        Functor functor(mInputGrid, mMask, threaded, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        if (functor.mOutputGrid) functor.mOutputGrid->setVectorType(VEC_COVARIANT);
        return functor.mOutputGrid;
    }

protected:
    struct Functor
    {
        Functor(const GridT& grid, const MaskGridType* mask, bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            typedef math::MeanCurvature<MapT, math::CD_SECOND, math::CD_2ND> OpT;
            gridop::GridOperator<GridT, MaskGridType, GridT, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool           mThreaded;
        const GridT&         mInputGrid;
        typename GridT::Ptr  mOutputGrid;
        InterruptT*          mInterrupt;
        const MaskGridType*  mMask;
    }; // Private Functor

    const GridT&        mInputGrid;
    InterruptT*         mInterrupt;
    const MaskGridType* mMask;
}; // end of MeanCurvature class


////////////////////////////////////////


template<
    typename InGridT,
    typename MaskGridType = typename gridop::ToMaskGrid<InGridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Magnitude
{
public:
    typedef InGridT                                         InGridType;
    typedef typename VectorToScalarConverter<InGridT>::Type OutGridType;

    Magnitude(const InGridType& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Magnitude(const InGridType& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename OutGridType::Ptr process(bool threaded = true)
    {
        Functor functor(mInputGrid, mMask, threaded, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        return functor.mOutputGrid;
    }

protected:
    struct OpT
    {
        template<typename MapT, typename AccT>
        static typename OutGridType::ValueType
        result(const MapT&, const AccT& acc, const Coord& xyz) { return acc.getValue(xyz).length();}
    };
    struct Functor
    {
        Functor(const InGridT& grid, const MaskGridType* mask,
            bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            gridop::GridOperator<InGridType, MaskGridType, OutGridType, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt, /*densify=*/false);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool                 mThreaded;
        const InGridType&          mInputGrid;
        typename OutGridType::Ptr  mOutputGrid;
        InterruptT*                mInterrupt;
        const MaskGridType*        mMask;
    }; // Private Functor

    const InGridType&    mInputGrid;
    InterruptT*          mInterrupt;
    const MaskGridType*  mMask;
}; // end of Magnitude class


////////////////////////////////////////


template<
    typename GridT,
    typename MaskGridType = typename gridop::ToMaskGrid<GridT>::Type,
    typename InterruptT = util::NullInterrupter>
class Normalize
{
public:
    typedef GridT  InGridType;
    typedef GridT  OutGridType;

    Normalize(const GridT& grid, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(nullptr)
    {
    }

    Normalize(const GridT& grid, const MaskGridType& mask, InterruptT* interrupt = nullptr):
        mInputGrid(grid), mInterrupt(interrupt), mMask(&mask)
    {
    }

    typename GridT::Ptr process(bool threaded = true)
    {
        Functor functor(mInputGrid, mMask, threaded, mInterrupt);
        processTypedMap(mInputGrid.transform(), functor);
        if (typename GridT::Ptr outGrid = functor.mOutputGrid) {
            const VecType vecType = mInputGrid.getVectorType();
            if (vecType == VEC_COVARIANT) {
                outGrid->setVectorType(VEC_COVARIANT_NORMALIZE);
            } else {
                outGrid->setVectorType(vecType);
            }
        }
        return functor.mOutputGrid;
    }

protected:
    struct OpT
    {
        template<typename MapT, typename AccT>
        static typename OutGridType::ValueType
        result(const MapT&, const AccT& acc, const Coord& xyz)
        {
            typename OutGridType::ValueType vec = acc.getValue(xyz);
            if ( !vec.normalize() ) vec.setZero();
            return vec;
        }
    };
    struct Functor
    {
        Functor(const GridT& grid, const MaskGridType* mask, bool threaded, InterruptT* interrupt):
            mThreaded(threaded), mInputGrid(grid), mInterrupt(interrupt), mMask(mask) {}

        template<typename MapT>
        void operator()(const MapT& map)
        {
            gridop::GridOperator<GridT, MaskGridType, GridT, MapT, OpT, InterruptT>
                op(mInputGrid, mMask, map, mInterrupt, /*densify=*/false);
            mOutputGrid = op.process(mThreaded); // cache the result
        }

        const bool           mThreaded;
        const GridT&         mInputGrid;
        typename GridT::Ptr  mOutputGrid;
        InterruptT*          mInterrupt;
        const MaskGridType*  mMask;
    }; // Private Functor

    const GridT&        mInputGrid;
    InterruptT*         mInterrupt;
    const MaskGridType* mMask;
}; // end of Normalize class


////////////////////////////////////////


template<typename GridType, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
cpt(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Cpt<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT> op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
cpt(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Cpt<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
curl(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Curl<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT> op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
curl(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Curl<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
divergence(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Divergence<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT>
        op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
divergence(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Divergence<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
gradient(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Gradient<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT>
        op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename ScalarToVectorConverter<GridType>::Type::Ptr
gradient(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Gradient<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
laplacian(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Laplacian<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT>
        op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
laplacian(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Laplacian<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
meanCurvature(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    MeanCurvature<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT>
        op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
meanCurvature(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    MeanCurvature<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
magnitude(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Magnitude<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT>
        op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename VectorToScalarConverter<GridType>::Type::Ptr
magnitude(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Magnitude<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename InterruptT> inline
typename GridType::Ptr
normalize(const GridType& grid, bool threaded, InterruptT* interrupt)
{
    Normalize<GridType, typename gridop::ToMaskGrid<GridType>::Type, InterruptT>
        op(grid, interrupt);
    return op.process(threaded);
}

template<typename GridType, typename MaskT, typename InterruptT> inline
typename GridType::Ptr
normalize(const GridType& grid, const MaskT& mask, bool threaded, InterruptT* interrupt)
{
    Normalize<GridType, MaskT, InterruptT> op(grid, mask, interrupt);
    return op.process(threaded);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_GRID_OPERATORS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
