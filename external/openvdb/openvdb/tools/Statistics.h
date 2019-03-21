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
/// @file Statistics.h
///
/// @brief Functions to efficiently compute histograms, extremas
/// (min/max) and statistics (mean, variance, etc.) of grid values

#ifndef OPENVDB_TOOLS_STATISTICS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_STATISTICS_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>
#include <openvdb/math/Stats.h>
#include "ValueTransformer.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Iterate over a scalar grid and compute a histogram of the values
/// of the voxels that are visited, or iterate over a vector-valued grid
/// and compute a histogram of the magnitudes of the vectors.
/// @param iter      an iterator over the values of a grid or its tree
///                  (@c Grid::ValueOnCIter, @c Tree::ValueOffIter, etc.)
/// @param minVal    the smallest value that can be added to the histogram
/// @param maxVal    the largest value that can be added to the histogram
/// @param numBins   the number of histogram bins
/// @param threaded  if true, iterate over the grid in parallel
template<typename IterT>
inline math::Histogram
histogram(const IterT& iter, double minVal, double maxVal,
          size_t numBins = 10, bool threaded = true);

/// @brief Iterate over a scalar grid and compute extrema (min/max) of the
/// values of the voxels that are visited, or iterate over a vector-valued grid
/// and compute extrema of the magnitudes of the vectors.
/// @param iter      an iterator over the values of a grid or its tree
///                  (@c Grid::ValueOnCIter, @c Tree::ValueOffIter, etc.)
/// @param threaded  if true, iterate over the grid in parallel
template<typename IterT>
inline math::Extrema
extrema(const IterT& iter, bool threaded = true);

/// @brief Iterate over a scalar grid and compute statistics (mean, variance, etc.)
/// of the values of the voxels that are visited, or iterate over a vector-valued grid
/// and compute statistics of the magnitudes of the vectors.
/// @param iter      an iterator over the values of a grid or its tree
///                  (@c Grid::ValueOnCIter, @c Tree::ValueOffIter, etc.)
/// @param threaded  if true, iterate over the grid in parallel
template<typename IterT>
inline math::Stats
statistics(const IterT& iter, bool threaded = true);

/// @brief Iterate over a grid and compute extrema (min/max) of
/// the values produced by applying the given functor at each voxel that is visited.
/// @param iter      an iterator over the values of a grid or its tree
///                  (@c Grid::ValueOnCIter, @c Tree::ValueOffIter, etc.)
/// @param op        a functor of the form <tt>void op(const IterT&, math::Stats&)</tt>,
///                  where @c IterT is the type of @a iter, that inserts zero or more
///                  floating-point values into the provided @c math::Stats object
/// @param threaded  if true, iterate over the grid in parallel
/// @note When @a threaded is true, each thread gets its own copy of the functor.
///
/// @par Example:
/// Compute statistics of just the active and positive-valued voxels of a scalar,
/// floating-point grid.
/// @code
/// struct Local {
///     static inline
///     void addIfPositive(const FloatGrid::ValueOnCIter& iter, math::Extrema& ex)
///     {
///         const float f = *iter;
///         if (f > 0.0) {
///             if (iter.isVoxelValue()) ex.add(f);
///             else ex.add(f, iter.getVoxelCount());
///         }
///     }
/// };
/// FloatGrid grid = ...;
/// math::Extrema stats =
///     tools::extrema(grid.cbeginValueOn(), Local::addIfPositive, /*threaded=*/true);
/// @endcode
template<typename IterT, typename ValueOp>
inline math::Extrema
extrema(const IterT& iter, const ValueOp& op, bool threaded);

/// @brief Iterate over a grid and compute statistics (mean, variance, etc.) of
/// the values produced by applying the given functor at each voxel that is visited.
/// @param iter      an iterator over the values of a grid or its tree
///                  (@c Grid::ValueOnCIter, @c Tree::ValueOffIter, etc.)
/// @param op        a functor of the form <tt>void op(const IterT&, math::Stats&)</tt>,
///                  where @c IterT is the type of @a iter, that inserts zero or more
///                  floating-point values into the provided @c math::Stats object
/// @param threaded  if true, iterate over the grid in parallel
/// @note When @a threaded is true, each thread gets its own copy of the functor.
///
/// @par Example:
/// Compute statistics of just the active and positive-valued voxels of a scalar,
/// floating-point grid.
/// @code
/// struct Local {
///     static inline
///     void addIfPositive(const FloatGrid::ValueOnCIter& iter, math::Stats& stats)
///     {
///         const float f = *iter;
///         if (f > 0.0) {
///             if (iter.isVoxelValue()) stats.add(f);
///             else stats.add(f, iter.getVoxelCount());
///         }
///     }
/// };
/// FloatGrid grid = ...;
/// math::Stats stats =
///     tools::statistics(grid.cbeginValueOn(), Local::addIfPositive, /*threaded=*/true);
/// @endcode
template<typename IterT, typename ValueOp>
inline math::Stats
statistics(const IterT& iter, const ValueOp& op, bool threaded);


/// @brief Iterate over a grid and compute statistics (mean, variance, etc.)
/// of the values produced by applying a given operator (see math/Operators.h)
/// at each voxel that is visited.
/// @param iter      an iterator over the values of a grid or its tree
///                  (@c Grid::ValueOnCIter, @c Tree::ValueOffIter, etc.)
/// @param op        an operator object with a method of the form
///                  <tt>double result(Accessor&, const Coord&)</tt>
/// @param threaded  if true, iterate over the grid in parallel
/// @note World-space operators, whose @c result() methods are of the form
/// <tt>double result(const Map&, Accessor&, const Coord&)</tt>, must be wrapped
/// in a math::MapAdapter.
/// @note Vector-valued operators like math::Gradient must be wrapped in an adapter
/// such as math::OpMagnitude.
///
/// @par Example:
/// Compute statistics of the magnitude of the gradient at the active voxels of
/// a scalar, floating-point grid.  (Note the use of the math::MapAdapter and
/// math::OpMagnitude adapters.)
/// @code
/// FloatGrid grid = ...;
///
/// // Assume that we know that the grid has a uniform scale map.
/// typedef math::UniformScaleMap MapType;
/// // Specify a world-space gradient operator that uses first-order differencing.
/// typedef math::Gradient<MapType, math::FD_1ST> GradientOp;
/// // Wrap the operator with an adapter that computes the magnitude of the gradient.
/// typedef math::OpMagnitude<GradientOp, MapType> MagnitudeOp;
/// // Wrap the operator with an adapter that associates a map with it.
/// typedef math::MapAdapter<MapType, GradientOp, double> CompoundOp;
///
/// if (MapType::Ptr map = grid.constTransform().constMap<MapType>()) {
///     math::Stats stats = tools::opStatistics(grid.cbeginValueOn(), CompoundOp(*map));
/// }
/// @endcode
///
/// @par Example:
/// Compute statistics of the divergence at the active voxels of a vector-valued grid.
/// @code
/// Vec3SGrid grid = ...;
///
/// // Assume that we know that the grid has a uniform scale map.
/// typedef math::UniformScaleMap MapType;
/// // Specify a world-space divergence operator that uses first-order differencing.
/// typedef math::Divergence<MapType, math::FD_1ST> DivergenceOp;
/// // Wrap the operator with an adapter that associates a map with it.
/// typedef math::MapAdapter<MapType, DivergenceOp, double> CompoundOp;
///
/// if (MapType::Ptr map = grid.constTransform().constMap<MapType>()) {
///     math::Stats stats = tools::opStatistics(grid.cbeginValueOn(), CompoundOp(*map));
/// }
/// @endcode
///
/// @par Example:
/// As above, but computing the divergence in index space.
/// @code
/// Vec3SGrid grid = ...;
///
/// // Specify an index-space divergence operator that uses first-order differencing.
/// typedef math::ISDivergence<math::FD_1ST> DivergenceOp;
///
/// math::Stats stats = tools::opStatistics(grid.cbeginValueOn(), DivergenceOp());
/// @endcode
template<typename OperatorT, typename IterT>
inline math::Stats
opStatistics(const IterT& iter, const OperatorT& op = OperatorT(), bool threaded = true);

/// @brief Same as opStatistics except it returns a math::Extrema vs a math::Stats
template<typename OperatorT, typename IterT>
inline math::Extrema
opExtrema(const IterT& iter, const OperatorT& op = OperatorT(), bool threaded = true);

////////////////////////////////////////


namespace stats_internal {

/// @todo This traits class is needed because tree::TreeValueIteratorBase uses
/// the name ValueT for the type of the value to which the iterator points,
/// whereas node-level iterators use the name ValueType.
template<typename IterT, typename AuxT = void>
struct IterTraits {
    typedef typename IterT::ValueType ValueType;
};

template<typename TreeT, typename ValueIterT>
struct IterTraits<tree::TreeValueIteratorBase<TreeT, ValueIterT> > {
    typedef typename tree::TreeValueIteratorBase<TreeT, ValueIterT>::ValueT ValueType;
};


// Helper class to compute a scalar value from either a scalar or a vector value
// (the latter by computing the vector's magnitude)
template<typename T, bool IsVector> struct GetValImpl;

template<typename T>
struct GetValImpl<T, /*IsVector=*/false> {
    static inline double get(const T& val) { return double(val); }
};

template<typename T>
struct GetValImpl<T, /*IsVector=*/true> {
    static inline double get(const T& val) { return val.length(); }
};


// Helper class to compute a scalar value from a tree or node iterator
// that points to a value in either a scalar or a vector grid, and to
// add that value to a math::Stats object.
template<typename IterT, typename StatsT>
struct GetVal
{
    typedef typename IterTraits<IterT>::ValueType ValueT;
    typedef GetValImpl<ValueT, VecTraits<ValueT>::IsVec> ImplT;

    inline void operator()(const IterT& iter, StatsT& stats) const {
        if (iter.isVoxelValue()) stats.add(ImplT::get(*iter));
        else stats.add(ImplT::get(*iter), iter.getVoxelCount());
    }
};

// Helper class to accumulate scalar voxel values or vector voxel magnitudes
// into a math::Stats object
template<typename IterT, typename ValueOp, typename StatsT>
struct StatsOp
{
    StatsOp(const ValueOp& op): getValue(op) {}

    // Accumulate voxel and tile values into this functor's Stats object.
    inline void operator()(const IterT& iter) { getValue(iter, stats); }

    // Accumulate another functor's Stats object into this functor's.
    inline void join(StatsOp& other) { stats.add(other.stats); }

    StatsT stats;
    ValueOp getValue;
};


// Helper class to accumulate scalar voxel values or vector voxel magnitudes
// into a math::Histogram object
template<typename IterT, typename ValueOp>
struct HistOp
{
    HistOp(const ValueOp& op, double vmin, double vmax, size_t bins):
        hist(vmin, vmax, bins), getValue(op)
    {}

    // Accumulate voxel and tile values into this functor's Histogram object.
    inline void operator()(const IterT& iter) { getValue(iter, hist); }

    // Accumulate another functor's Histogram object into this functor's.
    inline void join(HistOp& other) { hist.add(other.hist); }

    math::Histogram hist;
    ValueOp getValue;
};


// Helper class to apply an operator such as math::Gradient or math::Laplacian
// to voxels and accumulate the scalar results or the magnitudes of vector results
// into a math::Stats object
template<typename IterT, typename OpT, typename StatsT>
struct MathOp
{
    typedef typename IterT::TreeT                     TreeT;
    typedef typename TreeT::ValueType                 ValueT;
    typedef typename tree::ValueAccessor<const TreeT> ConstAccessor;

    // Each thread gets its own accessor and its own copy of the operator.
    ConstAccessor mAcc;
    OpT mOp;
    StatsT mStats;

    template<typename TreeT>
    static inline TreeT* THROW_IF_NULL(TreeT* ptr) {
        if (ptr == NULL) OPENVDB_THROW(ValueError, "iterator references a null tree");
        return ptr;
    }

    MathOp(const IterT& iter, const OpT& op):
        mAcc(*THROW_IF_NULL(iter.getTree())), mOp(op)
    {}

    // Accumulate voxel and tile values into this functor's Stats object.
    void operator()(const IterT& it)
    {
        if (it.isVoxelValue()) {
            // Add the magnitude of the gradient at a single voxel.
            mStats.add(mOp.result(mAcc, it.getCoord()));
        } else {
            // Iterate over the voxels enclosed by a tile and add the results
            // of applying the operator at each voxel.
            /// @todo This could be specialized to be done more efficiently for some operators.
            /// For example, all voxels in the interior of a tile (i.e., not on the borders)
            /// have gradient zero, so there's no need to apply the operator to every voxel.
            CoordBBox bbox = it.getBoundingBox();
            Coord xyz;
            int &x = xyz.x(), &y = xyz.y(), &z = xyz.z();
            for (x = bbox.min().x(); x <= bbox.max().x(); ++x) {
                for (y = bbox.min().y(); y <= bbox.max().y(); ++y) {
                    for (z = bbox.min().z(); z <= bbox.max().z(); ++z) {
                        mStats.add(mOp.result(mAcc, it.getCoord()));
                    }
                }
            }
        }
    }

    // Accumulate another functor's Stats object into this functor's.
    inline void join(MathOp& other) { mStats.add(other.mStats); }
}; // struct MathOp

} // namespace stats_internal


template<typename IterT>
inline math::Histogram
histogram(const IterT& iter, double vmin, double vmax, size_t numBins, bool threaded)
{
    typedef stats_internal::GetVal<IterT, math::Histogram> ValueOp;
    ValueOp valOp;
    stats_internal::HistOp<IterT, ValueOp> op(valOp, vmin, vmax, numBins);
    tools::accumulate(iter, op, threaded);
    return op.hist;
}

template<typename IterT>
inline math::Extrema
extrema(const IterT& iter, bool threaded)
{
    stats_internal::GetVal<IterT, math::Extrema> valOp;
    return extrema(iter, valOp, threaded);
}

template<typename IterT>
inline math::Stats
statistics(const IterT& iter, bool threaded)
{
    stats_internal::GetVal<IterT, math::Stats> valOp;
    return statistics(iter, valOp, threaded);
}

template<typename IterT, typename ValueOp>
inline math::Extrema
extrema(const IterT& iter, const ValueOp& valOp, bool threaded)
{
    stats_internal::StatsOp<IterT, const ValueOp, math::Extrema> op(valOp);
    tools::accumulate(iter, op, threaded);
    return op.stats;
}

template<typename IterT, typename ValueOp>
inline math::Stats
statistics(const IterT& iter, const ValueOp& valOp, bool threaded)
{
    stats_internal::StatsOp<IterT, const ValueOp, math::Stats> op(valOp);
    tools::accumulate(iter, op, threaded);
    return op.stats;
}


template<typename OperatorT, typename IterT>
inline math::Extrema
opExtrema(const IterT& iter, const OperatorT& op, bool threaded)
{
    stats_internal::MathOp<IterT, OperatorT, math::Extrema> func(iter, op);
    tools::accumulate(iter, func, threaded);
    return func.mStats;
}

template<typename OperatorT, typename IterT>
inline math::Stats
opStatistics(const IterT& iter, const OperatorT& op, bool threaded)
{
    stats_internal::MathOp<IterT, OperatorT, math::Stats> func(iter, op);
    tools::accumulate(iter, func, threaded);
    return func.mStats;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_STATISTICS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
