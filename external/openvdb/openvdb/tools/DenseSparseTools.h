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

#ifndef OPENVDB_TOOLS_DENSESPARSETOOLS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_DENSESPARSETOOLS_HAS_BEEN_INCLUDED

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range3d.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafManager.h>
#include "Dense.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Selectively extract and transform data from a dense grid, producing a
/// sparse tree with leaf nodes only (e.g. create a tree from the square
/// of values greater than a cutoff.)
/// @param dense       A dense grid that acts as a data source
/// @param functor     A functor that selects and transforms data for output
/// @param background  The background value of the resulting sparse grid
/// @param threaded    Option to use threaded or serial code path
/// @return @c Ptr to tree with the valuetype and configuration defined
/// by typedefs in the @c functor.
/// @note To achieve optimal sparsity  consider calling the prune()
/// method on the result.
/// @note To simply copy the all the data from a Dense grid to a
/// OpenVDB Grid, use tools::copyFromDense() for better performance.
///
/// The type of the sparse tree is determined by the specified OtpType
/// functor by means of the typedef OptType::ResultTreeType
///
/// The OptType function is responsible for the the transformation of
/// dense grid data to sparse grid data on a per-voxel basis.
///
/// Only leaf nodes with active values will be added to the sparse grid.
///
/// The OpType must struct that defines a the minimal form
/// @code
/// struct ExampleOp
/// {
///     typedef DesiredTreeType   ResultTreeType;
///
///     template<typename IndexOrCoord>
///      void OpType::operator() (const DenseValueType a, const IndexOrCoord& ijk,
///                    ResultTreeType::LeafNodeType* leaf);
/// };
/// @endcode
///
/// For example, to generate a <ValueType, 5, 4, 3> tree with valuesOn
/// at locations greater than a given maskvalue
/// @code
/// template <typename ValueType>
/// class Rule
/// {
/// public:
///     // Standard tree type (e.g. MaskTree or FloatTree in openvdb.h)
///     typedef typename openvdb::tree::Tree4<ValueType, 5, 4, 3>::Type  ResultTreeType;
///
///     typedef typename ResultTreeType::LeafNodeType  ResultLeafNodeType;
///     typedef typename ResultTreeType::ValueType     ResultValueType;
///
///     typedef float                         DenseValueType;
///
///     typedef vdbmath::Coord::ValueType     Index;
///
///     Rule(const DenseValueType& value): mMaskValue(value){};
///
///     template <typename IndexOrCoord>
///     void operator()(const DenseValueType& a, const IndexOrCoord& offset,
///                 ResultLeafNodeType* leaf) const
///     {
///             if (a > mMaskValue) {
///                 leaf->setValueOn(offset, a);
///             }
///     }
///
/// private:
///     const DenseValueType mMaskValue;
/// };
/// @endcode
template<typename OpType, typename DenseType>
typename OpType::ResultTreeType::Ptr
extractSparseTree(const DenseType& dense, const OpType& functor,
                  const typename OpType::ResultValueType& background,
                  bool threaded = true);

/// This struct that aids template resolution of a new tree type
/// has the same configuration at TreeType, but the ValueType from
/// DenseType.
template <typename DenseType, typename TreeType> struct DSConverter {
    typedef typename DenseType::ValueType  ValueType;

    typedef typename TreeType::template ValueConverter<ValueType>::Type Type;
};


/// @brief Copy data from the intersection of a sparse tree and a dense input grid.
/// The resulting tree has the same configuration as the sparse tree, but holds
/// the data type specified by the dense input.
/// @param dense       A dense grid that acts as a data source
/// @param mask        The active voxels and tiles intersected with dense define iteration mask
/// @param background  The background value of the resulting sparse grid
/// @param threaded    Option to use threaded or serial code path
/// @return @c Ptr to tree with the same configuration as @c mask but of value type
/// defined by @c dense.
template<typename DenseType, typename MaskTreeType>
typename DSConverter<DenseType, MaskTreeType>::Type::Ptr
extractSparseTreeWithMask(const DenseType& dense,
                          const MaskTreeType& mask,
                          const typename DenseType::ValueType& background,
                          bool threaded = true);


/// Apply a point-wise functor to the intersection of a dense grid and a given bounding box
/// @param dense A dense grid to be transformed
/// @param bbox  Index space bounding box, define region where the transformation is applied
/// @param op    A functor that acts on the dense grid value type
/// @param parallel Used to select multithreaded or single threaded
/// Minimally, the @c op class has to support a @c operator() method,
/// @code
/// // Square values in a grid
/// struct Op
/// {
///     ValueT operator()(const ValueT& in) const
///     {
///       // do work
///       ValueT result = in * in;
///
///       return result;
///     }
/// };
/// @endcode
/// NB: only Dense grids with memory layout zxy are supported
template<typename ValueT, typename OpType>
void transformDense(Dense<ValueT, openvdb::tools::LayoutZYX>& dense,
                    const openvdb::CoordBBox& bbox, const OpType& op, bool parallel=true);

/// We currrently support the following operations when compositing sparse
/// data into a dense grid.
enum DSCompositeOp {
    DS_OVER, DS_ADD, DS_SUB, DS_MIN, DS_MAX, DS_MULT, DS_SET
};

/// @brief Composite data from a sparse tree into a dense array of the same value type.
/// @param dense    Dense grid to be altered by the operation
/// @param source   Sparse data to composite into @c dense
/// @param alpha    Sparse Alpha mask used in compositing operations.
/// @param beta     Constant multiplier on src
/// @param strength Constant multiplier on alpha
/// @param threaded Enable threading for this operation.
template<DSCompositeOp, typename TreeT>
void compositeToDense(Dense<typename TreeT::ValueType, LayoutZYX>& dense,
                      const TreeT& source,
                      const TreeT& alpha,
                      const typename TreeT::ValueType beta,
                      const typename TreeT::ValueType strength,
                      bool threaded = true);


/// @brief Functor-based class used to extract data that satisfies some
/// criteria defined by the embedded @c OpType functor. The @c extractSparseTree
/// function wraps this class.
template<typename OpType, typename DenseType>
class SparseExtractor
{

public:

    typedef openvdb::math::Coord::ValueType              Index;

    typedef typename DenseType::ValueType                 DenseValueType;
    typedef typename OpType::ResultTreeType               ResultTreeType;
    typedef typename ResultTreeType::ValueType            ResultValueType;
    typedef typename ResultTreeType::LeafNodeType         ResultLeafNodeType;
    typedef typename ResultTreeType::template ValueConverter<ValueMask>::Type MaskTree;

    typedef tbb::blocked_range3d<Index, Index, Index>     Range3d;


private:

    const DenseType&                     mDense;
    const OpType&                        mFunctor;
    const ResultValueType                mBackground;
    const openvdb::math::CoordBBox       mBBox;
    const Index                          mWidth;
    typename ResultTreeType::Ptr         mMask;
    openvdb::math::Coord                 mMin;


public:

    SparseExtractor(const DenseType& dense, const OpType& functor,
                    const ResultValueType background) :
        mDense(dense), mFunctor(functor),
        mBackground(background),
        mBBox(dense.bbox()),
        mWidth(ResultLeafNodeType::DIM),
        mMask( new ResultTreeType(mBackground))
    {}


    SparseExtractor(const DenseType& dense,
                    const openvdb::math::CoordBBox& bbox,
                    const OpType& functor,
                    const ResultValueType background) :
        mDense(dense), mFunctor(functor),
        mBackground(background),
        mBBox(bbox),
        mWidth(ResultLeafNodeType::DIM),
        mMask( new ResultTreeType(mBackground))
    {
        // mBBox must be inside the coordinate rage of the dense grid
        if (!dense.bbox().isInside(mBBox)) {
            OPENVDB_THROW(ValueError, "Data extraction window out of bound");
        }
    }


    SparseExtractor(SparseExtractor& other, tbb::split):
        mDense(other.mDense), mFunctor(other.mFunctor),
        mBackground(other.mBackground), mBBox(other.mBBox),
        mWidth(other.mWidth),
        mMask(new ResultTreeType(mBackground)),
        mMin(other.mMin)
    {}

    typename ResultTreeType::Ptr extract(bool threaded = true) {


        // Construct 3D range of leaf nodes that
        // intersect mBBox.

        // Snap the bbox to nearest leaf nodes min and max

        openvdb::math::Coord padded_min = mBBox.min();
        openvdb::math::Coord padded_max = mBBox.max();


        padded_min &= ~(mWidth - 1);
        padded_max &= ~(mWidth - 1);

        padded_max[0] += mWidth - 1;
        padded_max[1] += mWidth - 1;
        padded_max[2] += mWidth - 1;


        // number of leaf nodes in each direction
        // division by leaf width, e.g. 8 in most cases

        const Index xleafCount = ( padded_max.x() - padded_min.x() + 1 ) / mWidth;
        const Index yleafCount = ( padded_max.y() - padded_min.y() + 1 ) / mWidth;
        const Index zleafCount = ( padded_max.z() - padded_min.z() + 1 ) / mWidth;

        mMin = padded_min;


        Range3d  leafRange(0, xleafCount, 1,
                           0, yleafCount, 1,
                           0, zleafCount, 1);


        // Iterate over the leafnodes applying *this as a functor.
        if (threaded) {
            tbb::parallel_reduce(leafRange, *this);
        } else {
            (*this)(leafRange);
        }

        return mMask;
    }


    void operator()(const Range3d& range) {

        ResultLeafNodeType* leaf = NULL;

        // Unpack the range3d item.
        const Index imin = range.pages().begin();
        const Index imax = range.pages().end();

        const Index jmin = range.rows().begin();
        const Index jmax = range.rows().end();

        const Index kmin = range.cols().begin();
        const Index kmax = range.cols().end();


        // loop over all the candidate leafs. Adding only those with 'true' values
        // to the tree

        for (Index i = imin; i < imax; ++i) {
            for (Index j = jmin; j < jmax; ++j) {
                for (Index k = kmin; k < kmax; ++k) {

                    // Calculate the origin of candidate leaf
                    const openvdb::math::Coord origin =
                        mMin + openvdb::math::Coord(mWidth * i,
                                                    mWidth * j,
                                                    mWidth * k );

                    if (leaf == NULL) {
                        leaf = new ResultLeafNodeType(origin, mBackground);
                    } else {
                        leaf->setOrigin(origin);
                        leaf->fill(mBackground);
                        leaf->setValuesOff();
                    }

                    // The bounding box for this leaf

                    openvdb::math::CoordBBox localBBox = leaf->getNodeBoundingBox();

                    // Shrink to the intersection with mBBox (i.e. the dense
                    // volume)

                    localBBox.intersect(mBBox);

                    // Early out for non-intersecting leafs

                    if (localBBox.empty()) continue;


                    const openvdb::math::Coord start = localBBox.getStart();
                    const openvdb::math::Coord end   = localBBox.getEnd();

                    // Order the looping to respect the memory layout in
                    // the Dense source

                    if (mDense.memoryLayout() == openvdb::tools::LayoutZYX) {

                        openvdb::math::Coord ijk;
                        Index offset;
                        const DenseValueType* dp;
                        for (ijk[0] = start.x(); ijk[0] < end.x(); ++ijk[0] ) {
                            for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1] ) {
                                for (ijk[2] = start.z(),
                                         offset = ResultLeafNodeType::coordToOffset(ijk),
                                         dp = &mDense.getValue(ijk);
                                     ijk[2] < end.z(); ++ijk[2], ++offset, ++dp) {

                                    mFunctor(*dp, offset, leaf);
                                }
                            }
                        }

                    } else {

                        openvdb::math::Coord ijk;
                        const DenseValueType* dp;
                        for (ijk[2] = start.z(); ijk[2] < end.z(); ++ijk[2]) {
                            for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1]) {
                                for (ijk[0] = start.x(),
                                         dp = &mDense.getValue(ijk);
                                     ijk[0] < end.x(); ++ijk[0], ++dp) {

                                    mFunctor(*dp, ijk, leaf);

                                }
                            }
                        }
                    }

                    // Only add non-empty leafs (empty is defined as all inactive)

                    if (!leaf->isEmpty()) {
                        mMask->addLeaf(leaf);
                        leaf = NULL;
                    }

                }
            }
        }

        // Clean up an unused leaf.

        if (leaf != NULL) delete leaf;
    }

    void join(SparseExtractor& rhs) {
        mMask->merge(*rhs.mMask);
    }
}; // class SparseExtractor


template<typename OpType, typename DenseType>
typename OpType::ResultTreeType::Ptr
extractSparseTree(const DenseType& dense, const OpType& functor,
                  const typename OpType::ResultValueType& background,
                  bool threaded)
{

    // Construct the mask using a parallel reduce pattern.
    // Each thread computes disjoint mask-trees.  The join merges
    // into a single tree.

    SparseExtractor<OpType, DenseType> extractor(dense, functor, background);

    return extractor.extract(threaded);
}


/// @brief Functor-based class used to extract data from a dense grid, at
/// the index-space intersection with a supplied mask in the form of a sparse tree.
/// The @c extractSparseTreeWithMask function wraps this class.
template <typename DenseType, typename MaskTreeType>
class SparseMaskedExtractor
{
public:

    typedef typename DSConverter<DenseType, MaskTreeType>::Type  _ResultTreeType;
    typedef _ResultTreeType                                      ResultTreeType;
    typedef typename ResultTreeType::LeafNodeType                ResultLeafNodeType;
    typedef typename ResultTreeType::ValueType                   ResultValueType;
    typedef ResultValueType                                      DenseValueType;

    typedef typename ResultTreeType::template ValueConverter<ValueMask>::Type  MaskTree;
    typedef typename MaskTree::LeafCIter                         MaskLeafCIter;
    typedef std::vector<const typename MaskTree::LeafNodeType*>  MaskLeafVec;


    SparseMaskedExtractor(const DenseType& dense,
                  const ResultValueType& background,
                  const MaskLeafVec& leafVec
                  ):
        mDense(dense), mBackground(background), mBBox(dense.bbox()),
        mLeafVec(leafVec),
        mResult(new ResultTreeType(mBackground))
    {}



    SparseMaskedExtractor(const SparseMaskedExtractor& other, tbb::split):
        mDense(other.mDense), mBackground(other.mBackground), mBBox(other.mBBox),
        mLeafVec(other.mLeafVec), mResult( new ResultTreeType(mBackground))
    {}

    typename ResultTreeType::Ptr extract(bool threaded = true) {

        tbb::blocked_range<size_t> range(0, mLeafVec.size());

        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }

        return mResult;
    }


    // Used in looping over leaf nodes in the masked grid
    // and using the active mask to select data to
    void operator()(const tbb::blocked_range<size_t>& range) {

        ResultLeafNodeType* leaf = NULL;


        // loop over all the candidate leafs. Adding only those with 'true' values
        // to the tree

        for (size_t idx = range.begin(); idx < range.end(); ++ idx) {

            const typename MaskTree::LeafNodeType* maskLeaf = mLeafVec[idx];

            // The bounding box for this leaf

            openvdb::math::CoordBBox localBBox = maskLeaf->getNodeBoundingBox();

            // Shrink to the intersection with the dense volume

            localBBox.intersect(mBBox);

            // Early out if there was no intersection

            if (localBBox.empty()) continue;

            // Reset or allocate the target leaf

            if (leaf == NULL) {
                leaf = new ResultLeafNodeType(maskLeaf->origin(), mBackground);
            } else {
                leaf->setOrigin(maskLeaf->origin());
                leaf->fill(mBackground);
                leaf->setValuesOff();
            }


            // Iterate over the intersecting bounding box
            // copying active values to the result tree

            const openvdb::math::Coord start = localBBox.getStart();
            const openvdb::math::Coord end   = localBBox.getEnd();


            openvdb::math::Coord ijk;

            if (mDense.memoryLayout() == openvdb::tools::LayoutZYX
                  && maskLeaf->isDense()) {

                Index offset;
                const DenseValueType* src;
                for (ijk[0] = start.x(); ijk[0] < end.x(); ++ijk[0] ) {
                    for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1] ) {
                        for (ijk[2] = start.z(),
                                 offset = ResultLeafNodeType::coordToOffset(ijk),
                                 src  = &mDense.getValue(ijk);
                             ijk[2] < end.z(); ++ijk[2], ++offset, ++src) {

                            // copy into leaf
                            leaf->setValueOn(offset, *src);
                        }

                    }
                }

            } else {

                Index offset;
                for (ijk[0] = start.x(); ijk[0] < end.x(); ++ijk[0] ) {
                    for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1] ) {
                        for (ijk[2] = start.z(),
                                 offset = ResultLeafNodeType::coordToOffset(ijk);
                             ijk[2] < end.z(); ++ijk[2], ++offset) {

                            if (maskLeaf->isValueOn(offset)) {
                                const ResultValueType denseValue =  mDense.getValue(ijk);
                                leaf->setValueOn(offset, denseValue);
                            }
                        }
                    }
                }
            }
            // Only add non-empty leafs (empty is defined as all inactive)

            if (!leaf->isEmpty()) {
                mResult->addLeaf(leaf);
                leaf = NULL;
            }
        }

        // Clean up an unused leaf.

        if (leaf != NULL) delete leaf;
    }

    void join(SparseMaskedExtractor& rhs) {
        mResult->merge(*rhs.mResult);
    }


private:
    const DenseType&                   mDense;
    const ResultValueType              mBackground;
    const openvdb::math::CoordBBox&    mBBox;
    const MaskLeafVec&                 mLeafVec;

    typename ResultTreeType::Ptr       mResult;

}; // class SparseMaskedExtractor


/// @brief a simple utility class used by @c extractSparseTreeWithMask
template<typename _ResultTreeType, typename DenseValueType>
struct ExtractAll
{
    typedef  _ResultTreeType                       ResultTreeType;
    typedef typename ResultTreeType::LeafNodeType  ResultLeafNodeType;

    template<typename CoordOrIndex> inline void
    operator()(const DenseValueType& a, const CoordOrIndex& offset, ResultLeafNodeType* leaf) const
    {
        leaf->setValueOn(offset, a);
    }
};


template <typename DenseType, typename MaskTreeType>
typename DSConverter<DenseType, MaskTreeType>::Type::Ptr
extractSparseTreeWithMask(const DenseType& dense,
                          const MaskTreeType& maskProxy,
                          const typename DenseType::ValueType& background,
                          bool threaded)
{
    typedef SparseMaskedExtractor<DenseType, MaskTreeType>       LeafExtractor;
    typedef typename LeafExtractor::DenseValueType               DenseValueType;
    typedef typename LeafExtractor::ResultTreeType               ResultTreeType;
    typedef typename LeafExtractor::MaskLeafVec                  MaskLeafVec;
    typedef typename LeafExtractor::MaskTree                     MaskTree;
    typedef typename LeafExtractor::MaskLeafCIter                MaskLeafCIter;
    typedef ExtractAll<ResultTreeType, DenseValueType>           ExtractionRule;

    // Use Mask tree to hold the topology

    MaskTree maskTree(maskProxy, false, TopologyCopy());

    // Construct an array of pointers to the mask leafs.

    const size_t leafCount = maskTree.leafCount();
    MaskLeafVec leafarray(leafCount);
    MaskLeafCIter leafiter = maskTree.cbeginLeaf();
    for (size_t n = 0; n != leafCount; ++n, ++leafiter) {
        leafarray[n] = leafiter.getLeaf();
    }


    // Extract the data that is masked leaf nodes in the mask.

    LeafExtractor leafextractor(dense, background, leafarray);
    typename ResultTreeType::Ptr resultTree = leafextractor.extract(threaded);


    // Extract data that is masked by tiles in the mask.


    // Loop over the mask tiles, extracting the data into new trees.
    // These trees will be leaf-orthogonal to the leafTree (i.e. no leaf
    // nodes will overlap).  Merge these trees into the result.

    typename MaskTreeType::ValueOnCIter tileIter(maskProxy);
    tileIter.setMaxDepth(MaskTreeType::ValueOnCIter::LEAF_DEPTH - 1);

    // Return the leaf tree if the mask had no tiles

    if (!tileIter) return resultTree;

    ExtractionRule allrule;

    // Loop over the tiles in series, but the actual data extraction
    // is in parallel.

    CoordBBox bbox;
    for ( ; tileIter; ++tileIter) {

        // Find the intersection of the tile with the dense grid.

        tileIter.getBoundingBox(bbox);
        bbox.intersect(dense.bbox());

        if (bbox.empty()) continue;

        SparseExtractor<ExtractionRule, DenseType> copyData(dense, bbox, allrule, background);
        typename ResultTreeType::Ptr fromTileTree = copyData.extract(threaded);
        resultTree->merge(*fromTileTree);
    }

    return resultTree;
}


/// @brief Class that applies a functor to the index space intersection
/// of a prescribed bounding box and the dense grid.
/// NB: This class only supports DenseGrids with ZYX memory layout.
template <typename _ValueT, typename OpType>
class DenseTransformer
{
public:

    typedef _ValueT                                 ValueT;
    typedef Dense<ValueT, openvdb::tools::LayoutZYX>       DenseT;
    typedef openvdb::math::Coord::ValueType         IntType;
    typedef tbb::blocked_range2d<IntType, IntType>  RangeType;


private:

    DenseT&                  mDense;
    const OpType&            mOp;
    openvdb::math::CoordBBox mBBox;

public:
    DenseTransformer(DenseT& dense,
                     const openvdb::math::CoordBBox& bbox,
                     const OpType& functor):
        mDense(dense), mOp(functor), mBBox(dense.bbox())
    {
        // The iteration space is the intersection of the
        // input bbox and the index-space covered by the dense grid
        mBBox.intersect(bbox);
    }

    DenseTransformer(const DenseTransformer& other) :
        mDense(other.mDense), mOp(other.mOp), mBBox(other.mBBox) {}

    void apply(bool threaded = true) {

        // Early out if the iteration space is empty

        if (mBBox.empty()) return;


        const openvdb::math::Coord start = mBBox.getStart();
        const openvdb::math::Coord end   = mBBox.getEnd();

        // The iteration range only the slower two directions.
        const RangeType range(start.x(), end.x(), 1,
                              start.y(), end.y(), 1);

        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }

    void operator()(const RangeType& range) const {

        // The stride in the z-direction.
        // Note: the bbox is [inclusive, inclusive]

        const size_t zlength = size_t(mBBox.max().z() - mBBox.min().z() + 1);

        const IntType imin = range.rows().begin();
        const IntType imax = range.rows().end();
        const IntType jmin = range.cols().begin();
        const IntType jmax = range.cols().end();


        openvdb::math::Coord xyz(imin, jmin, mBBox.min().z());
        for (xyz[0] = imin; xyz[0] != imax; ++xyz[0]) {
            for (xyz[1] = jmin; xyz[1] != jmax; ++xyz[1]) {

                mOp.transform(mDense, xyz, zlength);
            }
        }
    }
}; // class DenseTransformer


/// @brief a wrapper struct used to avoid unnecessary computation of
/// memory access from @c Coord when all offsets are guaranteed to be
/// within the dense grid.
template <typename ValueT, typename PointWiseOp>
struct ContiguousOp
{
    ContiguousOp(const PointWiseOp& op) : mOp(op){}

    typedef Dense<ValueT, openvdb::tools::LayoutZYX>  DenseT;
    inline void transform(DenseT& dense, openvdb::math::Coord& ijk, size_t size) const
    {
        ValueT* dp = const_cast<ValueT*>(&dense.getValue(ijk));

        for (size_t offset = 0; offset < size; ++offset) {
            dp[offset] = mOp(dp[offset]);
        }
    }

    const PointWiseOp mOp;
};


/// Apply a point-wise functor to the intersection of a dense grid and a given bounding box
template <typename ValueT, typename PointwiseOpT>
void
transformDense(Dense<ValueT, openvdb::tools::LayoutZYX>& dense,
               const openvdb::CoordBBox& bbox,
               const PointwiseOpT& functor, bool parallel)
{
    typedef ContiguousOp<ValueT, PointwiseOpT>  OpT;

    // Convert the Op so it operates on a contiguous line in memory

    OpT op(functor);

    // Apply to the index space intersection in the dense grid
    DenseTransformer<ValueT, OpT> transformer(dense, bbox, op);
    transformer.apply(parallel);
}


template <typename CompositeMethod, typename _TreeT>
class SparseToDenseCompositor
{

public:
    typedef _TreeT                                               TreeT;
    typedef typename TreeT::ValueType                            ValueT;
    typedef typename TreeT::LeafNodeType                         LeafT;
    typedef typename TreeT::template ValueConverter<ValueMask>::Type  MaskTreeT;
    typedef typename MaskTreeT::LeafNodeType                     MaskLeafT;
    typedef Dense<ValueT, openvdb::tools::LayoutZYX>             DenseT;
    typedef openvdb::math::Coord::ValueType                      Index;
    typedef tbb::blocked_range3d<Index, Index, Index>            Range3d;

    SparseToDenseCompositor(DenseT& dense, const TreeT& source, const TreeT& alpha,
                            const ValueT beta, const ValueT strength) :
        mDense(dense), mSource(source), mAlpha(alpha), mBeta(beta), mStrength(strength)
    {}

    SparseToDenseCompositor(const SparseToDenseCompositor& other):
        mDense(other.mDense), mSource(other.mSource), mAlpha(other.mAlpha),
        mBeta(other.mBeta), mStrength(other.mStrength) {}



    void sparseComposite(bool threaded) {

        const ValueT beta = mBeta;
        const ValueT strenght = mStrength;

        // construct a tree that defines the iteration space

        MaskTreeT maskTree(mSource, false /*background*/, openvdb::TopologyCopy());
        maskTree.topologyUnion(mAlpha);

        // Composite regions that are represented by leafnodes in either mAlpha or mSource
        // Parallelize over bool-leafs

        openvdb::tree::LeafManager<const MaskTreeT> maskLeafs(maskTree);
        maskLeafs.foreach(*this, threaded);

        // Composite regions that are represented by tiles
        // Parallelize within each tile.

        typename MaskTreeT::ValueOnCIter citer = maskTree.cbeginValueOn();
        citer.setMaxDepth(MaskTreeT::ValueOnCIter::LEAF_DEPTH - 1);

        if (!citer) return;

        typename tree::ValueAccessor<const TreeT>   alphaAccessor(mAlpha);
        typename tree::ValueAccessor<const TreeT>   sourceAccessor(mSource);

        for (; citer; ++citer) {

            const openvdb::math::Coord org = citer.getCoord();

            // Early out if both alpha and source are zero in this tile.

            const ValueT alphaValue = alphaAccessor.getValue(org);
            const ValueT sourceValue = sourceAccessor.getValue(org);

            if (openvdb::math::isZero(alphaValue) &&
                openvdb::math::isZero(sourceValue) ) continue;

            // Compute overlap of tile with the dense grid

            openvdb::math::CoordBBox localBBox = citer.getBoundingBox();
            localBBox.intersect(mDense.bbox());

            // Early out if there is no intersection

            if (localBBox.empty()) continue;

            // Composite the tile-uniform values into the dense grid.
            compositeFromTile(mDense, localBBox, sourceValue,
                              alphaValue, beta, strenght, threaded);
        }
    }

    // Composites leaf values where the alpha values are active.
    // Used in sparseComposite
    void inline operator()(const MaskLeafT& maskLeaf, size_t /*i*/) const
    {

        typedef UniformLeaf   ULeaf;
        openvdb::math::CoordBBox localBBox = maskLeaf.getNodeBoundingBox();
        localBBox.intersect(mDense.bbox());

        // Early out for non-overlapping leafs

        if (localBBox.empty()) return;

        const openvdb::math::Coord org = maskLeaf.origin();
        const LeafT* alphaLeaf = mAlpha.probeLeaf(org);
        const LeafT* sourceLeaf   = mSource.probeLeaf(org);

        if (!sourceLeaf) {

            // Create a source leaf proxy with the correct value
            ULeaf uniformSource(mSource.getValue(org));

            if (!alphaLeaf) {

                // Create an alpha leaf proxy with the correct value
                ULeaf uniformAlpha(mAlpha.getValue(org));

                compositeFromLeaf(mDense, localBBox, uniformSource, uniformAlpha,
                                  mBeta, mStrength);
            } else {

                compositeFromLeaf(mDense, localBBox, uniformSource, *alphaLeaf,
                                  mBeta, mStrength);
            }
        } else {
            if (!alphaLeaf) {

                // Create an alpha leaf proxy with the correct value
                ULeaf uniformAlpha(mAlpha.getValue(org));

                compositeFromLeaf(mDense, localBBox, *sourceLeaf, uniformAlpha,
                                  mBeta, mStrength);
            } else {

                compositeFromLeaf(mDense, localBBox, *sourceLeaf, *alphaLeaf,
                                  mBeta, mStrength);
            }
        }
    }
    // i.e.  it assumes that all valueOff Alpha voxels have value 0.

    template <typename LeafT1, typename LeafT2>
    inline static void compositeFromLeaf(DenseT& dense, const openvdb::math::CoordBBox& bbox,
                                         const LeafT1& source, const LeafT2& alpha,
                                         const ValueT beta, const ValueT strength)
    {
        typedef openvdb::math::Coord::ValueType  IntType;

        const ValueT sbeta = strength * beta;
        openvdb::math::Coord ijk = bbox.min();


        if (alpha.isDense() /*all active values*/) {

            // Optimal path for dense alphaLeaf
            const IntType size = bbox.max().z() + 1 - bbox.min().z();

            for (ijk[0] = bbox.min().x(); ijk[0] < bbox.max().x() + 1; ++ijk[0]) {
                for (ijk[1] = bbox.min().y(); ijk[1] < bbox.max().y() + 1; ++ijk[1]) {

                    ValueT* d = const_cast<ValueT*>(&dense.getValue(ijk));
                    const ValueT* a = &alpha.getValue(ijk);
                    const ValueT* s = &source.getValue(ijk);

                    for (IntType idx = 0; idx < size; ++idx) {
                        d[idx] = CompositeMethod::apply(d[idx], a[idx], s[idx],
                                                        strength, beta, sbeta);
                    }
                }
            }
        }  else {

            // AlphaLeaf has non-active cells.

            for (ijk[0] = bbox.min().x(); ijk[0] < bbox.max().x() + 1; ++ijk[0]) {
                for (ijk[1] = bbox.min().y(); ijk[1] < bbox.max().y() + 1; ++ijk[1]) {
                    for (ijk[2] = bbox.min().z(); ijk[2] < bbox.max().z() + 1; ++ijk[2]) {

                        if (alpha.isValueOn(ijk)) {

                            dense.setValue(ijk,
                             CompositeMethod::apply(dense.getValue(ijk),
                                                    alpha.getValue(ijk), source.getValue(ijk),
                                                    strength, beta, sbeta)
                                           );
                        }
                    }
                }
            }
        }
    }

    inline static void compositeFromTile(DenseT& dense, openvdb::math::CoordBBox& bbox,
                                         const ValueT& sourceValue, const ValueT& alphaValue,
                                         const ValueT& beta, const ValueT& strength,
                                         bool threaded)
    {

        typedef UniformTransformer TileTransformer;
        TileTransformer functor(sourceValue, alphaValue, beta, strength);

        // Transform the data inside the bbox according to the TileTranformer.

        transformDense(dense, bbox, functor, threaded);

    }


    void denseComposite(bool threaded)
    {
        /// Construct a range that corresponds to the
        /// bounding box of the dense volume
        const openvdb::math::CoordBBox& bbox = mDense.bbox();

        Range3d  range(bbox.min().x(), bbox.max().x(), LeafT::DIM,
                       bbox.min().y(), bbox.max().y(), LeafT::DIM,
                       bbox.min().z(), bbox.max().z(), LeafT::DIM);

        // Iterate over the range, compositing into
        // the dense grid using value accessors for
        // sparse the grids.
        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }

    }

    // Composites a dense region using value accessors
    // into a dense grid
    void inline operator()(const Range3d& range) const
    {
        // Use value accessors to alpha and source

        typename tree::ValueAccessor<const TreeT>   alphaAccessor(mAlpha);
        typename tree::ValueAccessor<const TreeT>   sourceAccessor(mSource);

        const ValueT strength = mStrength;
        const ValueT beta     = mBeta;
        const ValueT sbeta    = strength * beta;

        // Unpack the range3d item.
        const Index imin = range.pages().begin();
        const Index imax = range.pages().end();

        const Index jmin = range.rows().begin();
        const Index jmax = range.rows().end();

        const Index kmin = range.cols().begin();
        const Index kmax = range.cols().end();

        openvdb::Coord ijk;
        for (ijk[0] = imin; ijk[0] < imax; ++ijk[0]) {
            for (ijk[1] = jmin; ijk[1] < jmax; ++ijk[1]) {
                for (ijk[2] = kmin; ijk[2] < kmax; ++ijk[2]) {
                    const ValueT d_old = mDense.getValue(ijk);
                    const ValueT& alpha = alphaAccessor.getValue(ijk);
                    const ValueT& src   = sourceAccessor.getValue(ijk);

                    mDense.setValue(ijk, CompositeMethod::apply(d_old, alpha, src,
                                                                strength, beta, sbeta));
                }
            }
        }

    }


private:

    // Internal class that wraps the templated composite method
    // for use when both alpha and source are uniform over
    // a prescribed bbox (e.g. a tile).
    class UniformTransformer
    {
    public:
        UniformTransformer(const ValueT& source, const ValueT& alpha, const ValueT& _beta,
                           const ValueT& _strength) :
            mSource(source), mAlpha(alpha), mBeta(_beta),
            mStrength(_strength), mSBeta(_strength * _beta)
        {}

        ValueT operator()(const ValueT& input) const
        {
            return CompositeMethod::apply(input, mAlpha, mSource,
                                          mStrength, mBeta, mSBeta);
        }

    private:
        const ValueT mSource;   const ValueT mAlpha; const ValueT mBeta;
        const ValueT mStrength; const ValueT mSBeta;
    };


    // Simple Class structure that mimics a leaf
    // with uniform values. Holds LeafT::DIM copies
    // of a value in an array.
    struct Line {  ValueT mValues[LeafT::DIM]; };
    class UniformLeaf : private Line
    {
    public:
        typedef typename LeafT::ValueType ValueT;

        typedef Line   BaseT;
        UniformLeaf(const ValueT& value) : BaseT(init(value)) {}

        static const BaseT init(const ValueT& value) {
            BaseT tmp;
            for (openvdb::Index i = 0; i < LeafT::DIM; ++i) {
                tmp.mValues[i] = value;
            }
            return tmp;
        }

        bool isDense() const { return true; }
        bool isValueOn(openvdb::math::Coord&) const { return true; }

        inline const ValueT& getValue(const openvdb::math::Coord& ) const
        {return  BaseT::mValues[0];}
    };

private:
    DenseT&       mDense;
    const TreeT&  mSource;
    const TreeT&  mAlpha;
    ValueT        mBeta;
    ValueT        mStrength;
}; // class SparseToDenseCompositor


namespace ds
{
    //@{
    /// @brief Point wise methods used to apply various compositing operations.
    template <typename ValueT>
    struct OpOver
    {
        static inline ValueT apply(const ValueT u, const ValueT alpha,
                                   const ValueT v,
                                   const ValueT strength,
                                   const ValueT beta,
                                   const ValueT /*sbeta*/)
        { return (u + strength * alpha * (beta * v - u)); }
    };


    template <typename ValueT>
    struct OpAdd
    {
        static inline ValueT apply(const ValueT u, const ValueT alpha,
                                   const ValueT v,
                                   const ValueT /*strength*/,
                                   const ValueT /*beta*/,
                                   const ValueT sbeta)
        { return (u + sbeta * alpha * v); }
    };

    template <typename ValueT>
    struct OpSub
    {
        static inline ValueT apply(const ValueT u, const ValueT alpha,
                                   const ValueT v,
                                   const ValueT /*strength*/,
                                   const ValueT /*beta*/,
                                   const ValueT sbeta)
        { return (u - sbeta * alpha * v); }
    };

    template <typename ValueT>
    struct OpMin
    {
        static inline ValueT apply(const ValueT u, const ValueT alpha,
                                   const ValueT v,
                                   const ValueT s /*trength*/,
                                   const ValueT beta,
                                   const ValueT /*sbeta*/)
        { return ( ( 1 - s * alpha) * u + s * alpha * std::min(u, beta * v) ); }
    };


    template <typename ValueT>
    struct OpMax
    {
        static inline ValueT apply(const ValueT u, const ValueT alpha,
                                   const ValueT v,
                                   const ValueT s/*trength*/,
                                   const ValueT beta,
                                   const ValueT /*sbeta*/)
        { return ( ( 1 - s * alpha ) * u + s * alpha * std::min(u, beta * v) ); }
    };

    template <typename ValueT>
    struct OpMult
    {
        static inline ValueT apply(const ValueT u, const ValueT alpha,
                                   const ValueT v,
                                   const ValueT s/*trength*/,
                                   const ValueT /*beta*/,
                                   const ValueT sbeta)
        { return ( ( 1 + alpha * (sbeta * v - s)) * u ); }
    };
    //@}

    //@{
    /// Translator that converts an enum to compositing functor types
    template <DSCompositeOp OP, typename ValueT>
    struct CompositeFunctorTranslator{};

    template <typename ValueT>
    struct CompositeFunctorTranslator<DS_OVER, ValueT>{ typedef OpOver<ValueT>   OpT; };

    template <typename ValueT>
    struct CompositeFunctorTranslator<DS_ADD, ValueT>{ typedef OpAdd<ValueT>   OpT; };

    template <typename ValueT>
    struct CompositeFunctorTranslator<DS_SUB, ValueT>{ typedef OpSub<ValueT>   OpT; };

    template <typename ValueT>
    struct CompositeFunctorTranslator<DS_MIN, ValueT>{ typedef OpMin<ValueT>   OpT; };

    template <typename ValueT>
    struct CompositeFunctorTranslator<DS_MAX, ValueT>{ typedef OpMax<ValueT>   OpT; };

    template <typename ValueT>
    struct CompositeFunctorTranslator<DS_MULT, ValueT>{ typedef OpMult<ValueT>   OpT; };
    //@}

} // namespace ds


template <DSCompositeOp OpT, typename TreeT>
void compositeToDense(
    Dense<typename TreeT::ValueType, LayoutZYX>& dense,
    const TreeT& source, const TreeT& alpha,
    const typename TreeT::ValueType beta,
    const typename TreeT::ValueType strength,
    bool threaded)
{
    typedef typename TreeT::ValueType  ValueT;
    typedef ds::CompositeFunctorTranslator<OpT, ValueT> Translator;
    typedef typename Translator::OpT  Method;

    if (openvdb::math::isZero(strength)) return;

    SparseToDenseCompositor<Method, TreeT> tool(dense, source, alpha, beta, strength);

    if (openvdb::math::isZero(alpha.background()) &&
        openvdb::math::isZero(source.background()))
    {
        // Use the sparsity of (alpha U source) as the iteration space.
        tool.sparseComposite(threaded);
    } else {
        // Use the bounding box of dense as the iteration space.
        tool.denseComposite(threaded);
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif //OPENVDB_TOOLS_DENSESPARSETOOLS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
