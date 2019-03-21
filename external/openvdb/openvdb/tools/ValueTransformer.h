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
/// @file ValueTransformer.h
///
/// @author Peter Cucka
///
/// tools::foreach() and tools::transformValues() transform the values in a grid
/// by iterating over the grid with a user-supplied iterator and applying a
/// user-supplied functor at each step of the iteration.  With tools::foreach(),
/// the transformation is done in-place on the input grid, whereas with
/// tools::transformValues(), transformed values are written to an output grid
/// (which can, for example, have a different value type than the input grid).
/// Both functions can optionally transform multiple values of the grid in parallel.
///
/// tools::accumulate() can be used to accumulate the results of applying a functor
/// at each step of a grid iteration.  (The functor is responsible for storing and
/// updating intermediate results.)  When the iteration is done serially the behavior is
/// the same as with tools::foreach(), but when multiple values are processed in parallel,
/// an additional step is performed: when any two threads finish processing,
/// @c op.join(otherOp) is called on one thread's functor to allow it to coalesce
/// its intermediate result with the other thread's.
///
/// Finally, tools::setValueOnMin(), tools::setValueOnMax(), tools::setValueOnSum()
/// and tools::setValueOnMult() are wrappers around Tree::modifyValue() (or
/// ValueAccessor::modifyValue()) for some commmon in-place operations.
/// These are typically significantly faster than calling getValue() followed by setValue().

#ifndef OPENVDB_TOOLS_VALUETRANSFORMER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VALUETRANSFORMER_HAS_BEEN_INCLUDED

#include <algorithm> // for std::min(), std::max()
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Iterate over a grid and at each step call @c op(iter).
/// @param iter      an iterator over a grid or its tree (@c Grid::ValueOnCIter,
///                  @c Tree::NodeIter, etc.)
/// @param op        a functor of the form <tt>void op(const IterT&)</tt>, where @c IterT is
///                  the type of @a iter
/// @param threaded  if true, transform multiple values of the grid in parallel
/// @param shareOp   if true and @a threaded is true, all threads use the same functor;
///                  otherwise, each thread gets its own copy of the @e original functor
///
/// @par Example:
/// Multiply all values (both set and unset) of a scalar, floating-point grid by two.
/// @code
/// struct Local {
///     static inline void op(const FloatGrid::ValueAllIter& iter) {
///         iter.setValue(*iter * 2);
///     }
/// };
/// FloatGrid grid = ...;
/// tools::foreach(grid.beginValueAll(), Local::op);
/// @endcode
///
/// @par Example:
/// Rotate all active vectors of a vector grid by 45 degrees about the y axis.
/// @code
/// namespace {
///     struct MatMul {
///         math::Mat3s M;
///         MatMul(const math::Mat3s& mat): M(mat) {}
///         inline void operator()(const VectorGrid::ValueOnIter& iter) const {
///             iter.setValue(M.transform(*iter));
///         }
///     };
/// }
/// {
///     VectorGrid grid = ...;
///     tools::foreach(grid.beginValueOn(),
///         MatMul(math::rotation<math::Mat3s>(math::Y, M_PI_4)));
/// }
/// @endcode
///
/// @note For more complex operations that require finer control over threading,
/// consider using @c tbb::parallel_for() or @c tbb::parallel_reduce() in conjunction
/// with a tree::IteratorRange that wraps a grid or tree iterator.
template<typename IterT, typename XformOp>
inline void foreach(const IterT& iter, XformOp& op,
    bool threaded = true, bool shareOp = true);

template<typename IterT, typename XformOp>
inline void foreach(const IterT& iter, const XformOp& op,
    bool threaded = true, bool shareOp = true);


/// Iterate over a grid and at each step call <tt>op(iter, accessor)</tt> to
/// populate (via the accessor) the given output grid, whose @c ValueType
/// need not be the same as the input grid's.
/// @param inIter    a non-<tt>const</tt> or (preferably) @c const iterator over an
///                  input grid or its tree (@c Grid::ValueOnCIter, @c Tree::NodeIter, etc.)
/// @param outGrid   an empty grid to be populated
/// @param op        a functor of the form
///                  <tt>void op(const InIterT&, OutGridT::ValueAccessor&)</tt>,
///                  where @c InIterT is the type of @a inIter
/// @param threaded  if true, transform multiple values of the input grid in parallel
/// @param shareOp   if true and @a threaded is true, all threads use the same functor;
///                  otherwise, each thread gets its own copy of the @e original functor
/// @param merge     how to merge intermediate results from multiple threads (see Types.h)
///
/// @par Example:
/// Populate a scalar floating-point grid with the lengths of the vectors from all
/// active voxels of a vector-valued input grid.
/// @code
/// struct Local {
///     static void op(
///         const Vec3fGrid::ValueOnCIter& iter,
///         FloatGrid::ValueAccessor& accessor)
///     {
///         if (iter.isVoxelValue()) { // set a single voxel
///             accessor.setValue(iter.getCoord(), iter->length());
///         } else { // fill an entire tile
///             CoordBBox bbox;
///             iter.getBoundingBox(bbox);
///             accessor.getTree()->fill(bbox, iter->length());
///         }
///     }
/// };
/// Vec3fGrid inGrid = ...;
/// FloatGrid outGrid;
/// tools::transformValues(inGrid.cbeginValueOn(), outGrid, Local::op);
/// @endcode
///
/// @note For more complex operations that require finer control over threading,
/// consider using @c tbb::parallel_for() or @c tbb::parallel_reduce() in conjunction
/// with a tree::IteratorRange that wraps a grid or tree iterator.
template<typename InIterT, typename OutGridT, typename XformOp>
inline void transformValues(const InIterT& inIter, OutGridT& outGrid,
    XformOp& op, bool threaded = true, bool shareOp = true,
    MergePolicy merge = MERGE_ACTIVE_STATES);

#ifndef _MSC_VER
template<typename InIterT, typename OutGridT, typename XformOp>
inline void transformValues(const InIterT& inIter, OutGridT& outGrid,
    const XformOp& op, bool threaded = true, bool shareOp = true,
    MergePolicy merge = MERGE_ACTIVE_STATES);
#endif


/// Iterate over a grid and at each step call @c op(iter).  If threading is enabled,
/// call @c op.join(otherOp) to accumulate intermediate results from pairs of threads.
/// @param iter      an iterator over a grid or its tree (@c Grid::ValueOnCIter,
///                  @c Tree::NodeIter, etc.)
/// @param op        a functor with a join method of the form <tt>void join(XformOp&)</tt>
///                  and a call method of the form <tt>void op(const IterT&)</tt>,
///                  where @c IterT is the type of @a iter
/// @param threaded  if true, transform multiple values of the grid in parallel
/// @note If @a threaded is true, each thread gets its own copy of the @e original functor.
/// The order in which threads are joined is unspecified.
/// @note If @a threaded is false, the join method is never called.
///
/// @par Example:
/// Compute the average of the active values of a scalar, floating-point grid
/// using the math::Stats class.
/// @code
/// namespace {
///     struct Average {
///         math::Stats stats;
///
///         // Accumulate voxel and tile values into this functor's Stats object.
///         inline void operator()(const FloatGrid::ValueOnCIter& iter) {
///             if (iter.isVoxelValue()) stats.add(*iter);
///             else stats.add(*iter, iter.getVoxelCount());
///         }
///
///         // Accumulate another functor's Stats object into this functor's.
///         inline void join(Average& other) { stats.add(other.stats); }
///
///         // Return the cumulative result.
///         inline double average() const { return stats.mean(); }
///     };
/// }
/// {
///     FloatGrid grid = ...;
///     Average op;
///     tools::accumulate(grid.cbeginValueOn(), op);
///     double average = op.average();
/// }
/// @endcode
///
/// @note For more complex operations that require finer control over threading,
/// consider using @c tbb::parallel_for() or @c tbb::parallel_reduce() in conjunction
/// with a tree::IteratorRange that wraps a grid or tree iterator.
template<typename IterT, typename XformOp>
inline void accumulate(const IterT& iter, XformOp& op, bool threaded = true);


/// @brief Set the value of the voxel at the given coordinates in @a tree to
/// the minimum of its current value and @a value, and mark the voxel as active.
/// @details This is typically significantly faster than calling getValue()
/// followed by setValueOn().
/// @note @a TreeT can be either a Tree or a ValueAccessor.
template<typename TreeT>
inline void setValueOnMin(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value);

/// @brief Set the value of the voxel at the given coordinates in @a tree to
/// the maximum of its current value and @a value, and mark the voxel as active.
/// @details This is typically significantly faster than calling getValue()
/// followed by setValueOn().
/// @note @a TreeT can be either a Tree or a ValueAccessor.
template<typename TreeT>
inline void setValueOnMax(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value);

/// @brief Set the value of the voxel at the given coordinates in @a tree to
/// the sum of its current value and @a value, and mark the voxel as active.
/// @details This is typically significantly faster than calling getValue()
/// followed by setValueOn().
/// @note @a TreeT can be either a Tree or a ValueAccessor.
template<typename TreeT>
inline void setValueOnSum(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value);

/// @brief Set the value of the voxel at the given coordinates in @a tree to
/// the product of its current value and @a value, and mark the voxel as active.
/// @details This is typically significantly faster than calling getValue()
/// followed by setValueOn().
/// @note @a TreeT can be either a Tree or a ValueAccessor.
template<typename TreeT>
inline void setValueOnMult(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value);


////////////////////////////////////////


namespace valxform {

template<typename ValueType>
struct MinOp {
    const ValueType val;
    MinOp(const ValueType& v): val(v) {}
    inline void operator()(ValueType& v) const { v = std::min<ValueType>(v, val); }
};

template<typename ValueType>
struct MaxOp {
    const ValueType val;
    MaxOp(const ValueType& v): val(v) {}
    inline void operator()(ValueType& v) const { v = std::max<ValueType>(v, val); }
};

template<typename ValueType>
struct SumOp {
    const ValueType val;
    SumOp(const ValueType& v): val(v) {}
    inline void operator()(ValueType& v) const { v += val; }
};

template<typename ValueType>
struct MultOp {
    const ValueType val;
    MultOp(const ValueType& v): val(v) {}
    inline void operator()(ValueType& v) const { v *= val; }
};

}


template<typename TreeT>
inline void
setValueOnMin(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value)
{
    tree.modifyValue(xyz, valxform::MinOp<typename TreeT::ValueType>(value));
}


template<typename TreeT>
inline void
setValueOnMax(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value)
{
    tree.modifyValue(xyz, valxform::MaxOp<typename TreeT::ValueType>(value));
}


template<typename TreeT>
inline void
setValueOnSum(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value)
{
    tree.modifyValue(xyz, valxform::SumOp<typename TreeT::ValueType>(value));
}


template<typename TreeT>
inline void
setValueOnMult(TreeT& tree, const Coord& xyz, const typename TreeT::ValueType& value)
{
    tree.modifyValue(xyz, valxform::MultOp<typename TreeT::ValueType>(value));
}


////////////////////////////////////////


namespace valxform {

template<typename IterT, typename OpT>
class SharedOpApplier
{
public:
    typedef typename tree::IteratorRange<IterT> IterRange;

    SharedOpApplier(const IterT& iter, OpT& op): mIter(iter), mOp(op) {}

    void process(bool threaded = true)
    {
        IterRange range(mIter);
        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }

    void operator()(IterRange& r) const { for ( ; r; ++r) mOp(r.iterator()); }

private:
    IterT mIter;
    OpT& mOp;
};


template<typename IterT, typename OpT>
class CopyableOpApplier
{
public:
    typedef typename tree::IteratorRange<IterT> IterRange;

    CopyableOpApplier(const IterT& iter, const OpT& op): mIter(iter), mOp(op), mOrigOp(&op) {}

    // When splitting this task, give the subtask a copy of the original functor,
    // not of this task's functor, which might have been modified arbitrarily.
    CopyableOpApplier(const CopyableOpApplier& other):
        mIter(other.mIter), mOp(*other.mOrigOp), mOrigOp(other.mOrigOp) {}

    void process(bool threaded = true)
    {
        IterRange range(mIter);
        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }

    void operator()(IterRange& r) const { for ( ; r; ++r) mOp(r.iterator()); }

private:
    IterT mIter;
    OpT mOp; // copy of original functor
    OpT const * const mOrigOp; // pointer to original functor
};

} // namespace valxform


template<typename IterT, typename XformOp>
inline void
foreach(const IterT& iter, XformOp& op, bool threaded, bool shared)
{
    if (shared) {
        typename valxform::SharedOpApplier<IterT, XformOp> proc(iter, op);
        proc.process(threaded);
    } else {
        typedef typename valxform::CopyableOpApplier<IterT, XformOp> Processor;
        Processor proc(iter, op);
        proc.process(threaded);
    }
}

template<typename IterT, typename XformOp>
inline void
foreach(const IterT& iter, const XformOp& op, bool threaded, bool /*shared*/)
{
    // Const ops are shared across threads, not copied.
    typename valxform::SharedOpApplier<IterT, const XformOp> proc(iter, op);
    proc.process(threaded);
}


////////////////////////////////////////


namespace valxform {

template<typename InIterT, typename OutTreeT, typename OpT>
class SharedOpTransformer
{
public:
    typedef typename InIterT::TreeT InTreeT;
    typedef typename tree::IteratorRange<InIterT> IterRange;
    typedef typename OutTreeT::ValueType OutValueT;

    SharedOpTransformer(const InIterT& inIter, OutTreeT& outTree, OpT& op, MergePolicy merge):
        mIsRoot(true),
        mInputIter(inIter),
        mInputTree(inIter.getTree()),
        mOutputTree(&outTree),
        mOp(op),
        mMergePolicy(merge)
    {
        if (static_cast<const void*>(mInputTree) == static_cast<void*>(mOutputTree)) {
            OPENVDB_LOG_INFO("use tools::foreach(), not transformValues(),"
                " to transform a grid in place");
        }
    }

    /// Splitting constructor
    SharedOpTransformer(SharedOpTransformer& other, tbb::split):
        mIsRoot(false),
        mInputIter(other.mInputIter),
        mInputTree(other.mInputTree),
        mOutputTree(new OutTreeT(zeroVal<OutValueT>())),
        mOp(other.mOp),
        mMergePolicy(other.mMergePolicy)
        {}

    ~SharedOpTransformer()
    {
        // Delete the output tree only if it was allocated locally
        // (the top-level output tree was supplied by the caller).
        if (!mIsRoot) {
            delete mOutputTree;
            mOutputTree = NULL;
        }
    }

    void process(bool threaded = true)
    {
        if (!mInputTree || !mOutputTree) return;

        IterRange range(mInputIter);

        // Independently transform elements in the iterator range,
        // either in parallel or serially.
        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }
    }

    /// Transform each element in the given range.
    void operator()(IterRange& range) const
    {
        if (!mOutputTree) return;
        typename tree::ValueAccessor<OutTreeT> outAccessor(*mOutputTree);
        for ( ; range; ++range) {
            mOp(range.iterator(), outAccessor);
        }
    }

    void join(const SharedOpTransformer& other)
    {
        if (mOutputTree && other.mOutputTree) {
            mOutputTree->merge(*other.mOutputTree, mMergePolicy);
        }
    }

private:
    bool mIsRoot;
    InIterT mInputIter;
    const InTreeT* mInputTree;
    OutTreeT* mOutputTree;
    OpT& mOp;
    MergePolicy mMergePolicy;
}; // class SharedOpTransformer


template<typename InIterT, typename OutTreeT, typename OpT>
class CopyableOpTransformer
{
public:
    typedef typename InIterT::TreeT InTreeT;
    typedef typename tree::IteratorRange<InIterT> IterRange;
    typedef typename OutTreeT::ValueType OutValueT;

    CopyableOpTransformer(const InIterT& inIter, OutTreeT& outTree,
        const OpT& op, MergePolicy merge):
        mIsRoot(true),
        mInputIter(inIter),
        mInputTree(inIter.getTree()),
        mOutputTree(&outTree),
        mOp(op),
        mOrigOp(&op),
        mMergePolicy(merge)
    {
        if (static_cast<const void*>(mInputTree) == static_cast<void*>(mOutputTree)) {
            OPENVDB_LOG_INFO("use tools::foreach(), not transformValues(),"
                " to transform a grid in place");
        }
    }

    // When splitting this task, give the subtask a copy of the original functor,
    // not of this task's functor, which might have been modified arbitrarily.
    CopyableOpTransformer(CopyableOpTransformer& other, tbb::split):
        mIsRoot(false),
        mInputIter(other.mInputIter),
        mInputTree(other.mInputTree),
        mOutputTree(new OutTreeT(zeroVal<OutValueT>())),
        mOp(*other.mOrigOp),
        mOrigOp(other.mOrigOp),
        mMergePolicy(other.mMergePolicy)
        {}

    ~CopyableOpTransformer()
    {
        // Delete the output tree only if it was allocated locally
        // (the top-level output tree was supplied by the caller).
        if (!mIsRoot) {
            delete mOutputTree;
            mOutputTree = NULL;
        }
    }

    void process(bool threaded = true)
    {
        if (!mInputTree || !mOutputTree) return;

        IterRange range(mInputIter);

        // Independently transform elements in the iterator range,
        // either in parallel or serially.
        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }
    }

    /// Transform each element in the given range.
    void operator()(IterRange& range)
    {
        if (!mOutputTree) return;
        typename tree::ValueAccessor<OutTreeT> outAccessor(*mOutputTree);
        for ( ; range; ++range) {
            mOp(range.iterator(), outAccessor);
        }
    }

    void join(const CopyableOpTransformer& other)
    {
        if (mOutputTree && other.mOutputTree) {
            mOutputTree->merge(*other.mOutputTree, mMergePolicy);
        }
    }

private:
    bool mIsRoot;
    InIterT mInputIter;
    const InTreeT* mInputTree;
    OutTreeT* mOutputTree;
    OpT mOp; // copy of original functor
    OpT const * const mOrigOp; // pointer to original functor
    MergePolicy mMergePolicy;
}; // class CopyableOpTransformer

} // namespace valxform


////////////////////////////////////////


template<typename InIterT, typename OutGridT, typename XformOp>
inline void
transformValues(const InIterT& inIter, OutGridT& outGrid, XformOp& op,
    bool threaded, bool shared, MergePolicy merge)
{
    typedef TreeAdapter<OutGridT> Adapter;
    typedef typename Adapter::TreeType OutTreeT;
    if (shared) {
        typedef typename valxform::SharedOpTransformer<InIterT, OutTreeT, XformOp> Processor;
        Processor proc(inIter, Adapter::tree(outGrid), op, merge);
        proc.process(threaded);
    } else {
        typedef typename valxform::CopyableOpTransformer<InIterT, OutTreeT, XformOp> Processor;
        Processor proc(inIter, Adapter::tree(outGrid), op, merge);
        proc.process(threaded);
    }
}

#ifndef _MSC_VER
template<typename InIterT, typename OutGridT, typename XformOp>
inline void
transformValues(const InIterT& inIter, OutGridT& outGrid, const XformOp& op,
    bool threaded, bool /*share*/, MergePolicy merge)
{
    typedef TreeAdapter<OutGridT> Adapter;
    typedef typename Adapter::TreeType OutTreeT;
    // Const ops are shared across threads, not copied.
    typedef typename valxform::SharedOpTransformer<InIterT, OutTreeT, const XformOp> Processor;
    Processor proc(inIter, Adapter::tree(outGrid), op, merge);
    proc.process(threaded);
}
#endif


////////////////////////////////////////


namespace valxform {

template<typename IterT, typename OpT>
class OpAccumulator
{
public:
    typedef typename tree::IteratorRange<IterT> IterRange;

    // The root task makes a const copy of the original functor (mOrigOp)
    // and keeps a pointer to the original functor (mOp), which it then modifies.
    // Each subtask keeps a const pointer to the root task's mOrigOp
    // and makes and then modifies a non-const copy (mOp) of it.
    OpAccumulator(const IterT& iter, OpT& op):
        mIsRoot(true),
        mIter(iter),
        mOp(&op),
        mOrigOp(new OpT(op))
    {}

    // When splitting this task, give the subtask a copy of the original functor,
    // not of this task's functor, which might have been modified arbitrarily.
    OpAccumulator(OpAccumulator& other, tbb::split):
        mIsRoot(false),
        mIter(other.mIter),
        mOp(new OpT(*other.mOrigOp)),
        mOrigOp(other.mOrigOp)
    {}

    ~OpAccumulator() { if (mIsRoot) delete mOrigOp; else delete mOp; }

    void process(bool threaded = true)
    {
        IterRange range(mIter);
        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }
    }

    void operator()(IterRange& r) { for ( ; r; ++r) (*mOp)(r.iterator()); }

    void join(OpAccumulator& other) { mOp->join(*other.mOp); }

private:
    const bool mIsRoot;
    const IterT mIter;
    OpT* mOp; // pointer to original functor, which might get modified
    OpT const * const mOrigOp; // const copy of original functor
}; // class OpAccumulator

} // namespace valxform


////////////////////////////////////////


template<typename IterT, typename XformOp>
inline void
accumulate(const IterT& iter, XformOp& op, bool threaded)
{
    typename valxform::OpAccumulator<IterT, XformOp> proc(iter, op);
    proc.process(threaded);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VALUETRANSFORMER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
