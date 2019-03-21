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
/// @file Composite.h
///
/// @brief Functions to efficiently perform various compositing operations on grids
///
/// @authors Peter Cucka, Mihai Alden, Ken Museth

#ifndef OPENVDB_TOOLS_COMPOSITE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_COMPOSITE_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h> // for isExactlyEqual()
#include "ValueTransformer.h" // for transformValues()
#include "Prune.h"// for prune
#include "SignedFloodFill.h" // for signedFloodFill()

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

#include <type_traits>
#include <functional>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Given two level set grids, replace the A grid with the union of A and B.
/// @throw ValueError if the background value of either grid is not greater than zero.
/// @note This operation always leaves the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void csgUnion(GridOrTreeT& a, GridOrTreeT& b, bool prune = true);
/// @brief Given two level set grids, replace the A grid with the intersection of A and B.
/// @throw ValueError if the background value of either grid is not greater than zero.
/// @note This operation always leaves the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void csgIntersection(GridOrTreeT& a, GridOrTreeT& b, bool prune = true);
/// @brief Given two level set grids, replace the A grid with the difference A / B.
/// @throw ValueError if the background value of either grid is not greater than zero.
/// @note This operation always leaves the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void csgDifference(GridOrTreeT& a, GridOrTreeT& b, bool prune = true);

/// @brief  Threaded CSG union operation that produces a new grid or tree from
///         immutable inputs.
/// @return The CSG union of the @a and @b level set inputs.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline typename GridOrTreeT::Ptr csgUnionCopy(const GridOrTreeT& a, const GridOrTreeT& b);
/// @brief  Threaded CSG intersection operation that produces a new grid or tree from
///         immutable inputs.
/// @return The CSG intersection of the @a and @b level set inputs.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline typename GridOrTreeT::Ptr csgIntersectionCopy(const GridOrTreeT& a, const GridOrTreeT& b);
/// @brief  Threaded CSG difference operation that produces a new grid or tree from
///         immutable inputs.
/// @return The CSG difference of the @a and @b level set inputs.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline typename GridOrTreeT::Ptr csgDifferenceCopy(const GridOrTreeT& a, const GridOrTreeT& b);

/// @brief Given grids A and B, compute max(a, b) per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compMax(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute min(a, b) per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compMin(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute a + b per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compSum(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute a * b per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compMul(GridOrTreeT& a, GridOrTreeT& b);
/// @brief Given grids A and B, compute a / b per voxel (using sparse traversal).
/// Store the result in the A grid and leave the B grid empty.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compDiv(GridOrTreeT& a, GridOrTreeT& b);

/// Copy the active voxels of B into A.
template<typename GridOrTreeT> OPENVDB_STATIC_SPECIALIZATION
inline void compReplace(GridOrTreeT& a, const GridOrTreeT& b);


////////////////////////////////////////


namespace composite {

// composite::min() and composite::max() for non-vector types compare with operator<().
template<typename T> inline
const typename std::enable_if<!VecTraits<T>::IsVec, T>::type& // = T if T is not a vector type
min(const T& a, const T& b) { return std::min(a, b); }

template<typename T> inline
const typename std::enable_if<!VecTraits<T>::IsVec, T>::type&
max(const T& a, const T& b) { return std::max(a, b); }


// composite::min() and composite::max() for OpenVDB vector types compare by magnitude.
template<typename T> inline
const typename std::enable_if<VecTraits<T>::IsVec, T>::type& // = T if T is a vector type
min(const T& a, const T& b)
{
    const typename T::ValueType aMag = a.lengthSqr(), bMag = b.lengthSqr();
    return (aMag < bMag ? a : (bMag < aMag ? b : std::min(a, b)));
}

template<typename T> inline
const typename std::enable_if<VecTraits<T>::IsVec, T>::type&
max(const T& a, const T& b)
{
    const typename T::ValueType aMag = a.lengthSqr(), bMag = b.lengthSqr();
    return (aMag < bMag ? b : (bMag < aMag ? a : std::max(a, b)));
}


template<typename T> inline
typename std::enable_if<!std::is_integral<T>::value, T>::type // = T if T is not an integer type
divide(const T& a, const T& b) { return a / b; }

template<typename T> inline
typename std::enable_if<std::is_integral<T>::value, T>::type // = T if T is an integer type
divide(const T& a, const T& b)
{
    const T zero(0);
    if (b != zero) return a / b;
    if (a == zero) return 0;
    return (a > 0 ? std::numeric_limits<T>::max() : -std::numeric_limits<T>::max());
}

// If b is true, return a / 1 = a.
// If b is false and a is true, return 1 / 0 = inf = MAX_BOOL = 1 = a.
// If b is false and a is false, return 0 / 0 = NaN = 0 = a.
inline bool divide(bool a, bool /*b*/) { return a; }


enum CSGOperation { CSG_UNION, CSG_INTERSECTION, CSG_DIFFERENCE };

template<typename TreeType, CSGOperation Operation>
struct BuildPrimarySegment
{
    typedef typename TreeType::ValueType                                            ValueType;
    typedef typename TreeType::Ptr                                                  TreePtrType;
    typedef typename TreeType::LeafNodeType                                         LeafNodeType;
    typedef typename LeafNodeType::NodeMaskType                                     NodeMaskType;
    typedef typename TreeType::RootNodeType                                         RootNodeType;
    typedef typename RootNodeType::NodeChainType                                    NodeChainType;
    typedef typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type      InternalNodeType;

    BuildPrimarySegment(const TreeType& lhs, const TreeType& rhs)
        : mSegment(new TreeType(lhs.background()))
        , mLhsTree(&lhs)
        , mRhsTree(&rhs)
    {
    }

    void operator()() const
    {
        std::vector<const LeafNodeType*> leafNodes;

        {
            std::vector<const InternalNodeType*> internalNodes;
            mLhsTree->getNodes(internalNodes);

            ProcessInternalNodes op(internalNodes, *mRhsTree, *mSegment, leafNodes);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), op);
        }

        ProcessLeafNodes op(leafNodes, *mRhsTree, *mSegment);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leafNodes.size()), op);
    }

    TreePtrType& segment() { return mSegment; }

private:

    struct ProcessInternalNodes {

        ProcessInternalNodes(std::vector<const InternalNodeType*>& lhsNodes, const TreeType& rhsTree,
            TreeType& outputTree, std::vector<const LeafNodeType*>& outputLeafNodes)
            : mLhsNodes(lhsNodes.empty() ? NULL : &lhsNodes.front())
            , mRhsTree(&rhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&outputTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&outputLeafNodes)
        {
        }

        ProcessInternalNodes(ProcessInternalNodes& other, tbb::split)
            : mLhsNodes(other.mLhsNodes)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&mLocalTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&mLocalLeafNodes)
        {
        }

        void join(ProcessInternalNodes& other)
        {
            mOutputTree->merge(*other.mOutputTree);
            mOutputLeafNodes->insert(mOutputLeafNodes->end(),
                other.mOutputLeafNodes->begin(), other.mOutputLeafNodes->end());
        }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            tree::ValueAccessor<const TreeType> rhsAcc(*mRhsTree);
            tree::ValueAccessor<TreeType>       outputAcc(*mOutputTree);

            std::vector<const LeafNodeType*> tmpLeafNodes;

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const InternalNodeType& lhsNode = *mLhsNodes[n];
                const Coord& ijk = lhsNode.origin();
                const InternalNodeType * rhsNode = rhsAcc.template probeConstNode<InternalNodeType>(ijk);

                if (rhsNode) {
                    lhsNode.getNodes(*mOutputLeafNodes);
                } else {
                    if (Operation == CSG_INTERSECTION) {
                        if (rhsAcc.getValue(ijk) < ValueType(0.0)) {
                            tmpLeafNodes.clear();
                            lhsNode.getNodes(tmpLeafNodes);
                            for (size_t i = 0, I = tmpLeafNodes.size(); i < I; ++i) {
                                outputAcc.addLeaf(new LeafNodeType(*tmpLeafNodes[i]));
                            }
                        }
                    } else { // Union & Difference
                        if (!(rhsAcc.getValue(ijk) < ValueType(0.0))) {
                            tmpLeafNodes.clear();
                            lhsNode.getNodes(tmpLeafNodes);
                            for (size_t i = 0, I = tmpLeafNodes.size(); i < I; ++i) {
                                outputAcc.addLeaf(new LeafNodeType(*tmpLeafNodes[i]));
                            }
                        }
                    }
                }
            } //  end range loop
        }

        InternalNodeType const * const * const mLhsNodes;
        TreeType                 const * const mRhsTree;
        TreeType                               mLocalTree;
        TreeType                       * const mOutputTree;

        std::vector<const LeafNodeType*>         mLocalLeafNodes;
        std::vector<const LeafNodeType*> * const mOutputLeafNodes;
    }; // struct ProcessInternalNodes

    struct ProcessLeafNodes {

        ProcessLeafNodes(std::vector<const LeafNodeType*>& lhsNodes, const TreeType& rhsTree, TreeType& output)
            : mLhsNodes(lhsNodes.empty() ? NULL : &lhsNodes.front())
            , mRhsTree(&rhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&output)
        {
        }

        ProcessLeafNodes(ProcessLeafNodes& other, tbb::split)
            : mLhsNodes(other.mLhsNodes)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&mLocalTree)
        {
        }

        void join(ProcessLeafNodes& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            tree::ValueAccessor<const TreeType> rhsAcc(*mRhsTree);
            tree::ValueAccessor<TreeType>       outputAcc(*mOutputTree);

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const LeafNodeType& lhsNode = *mLhsNodes[n];
                const Coord& ijk = lhsNode.origin();

                const LeafNodeType* rhsNodePt = rhsAcc.probeConstLeaf(ijk);

                if (rhsNodePt) { // combine overlapping nodes

                    LeafNodeType* outputNode = outputAcc.touchLeaf(ijk);
                    ValueType * outputData = outputNode->buffer().data();
                    NodeMaskType& outputMask = outputNode->getValueMask();

                    const ValueType * lhsData = lhsNode.buffer().data();
                    const NodeMaskType& lhsMask = lhsNode.getValueMask();

                    const ValueType * rhsData = rhsNodePt->buffer().data();
                    const NodeMaskType& rhsMask = rhsNodePt->getValueMask();

                    if (Operation == CSG_INTERSECTION) {
                        for (Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                            const bool fromRhs = lhsData[pos] < rhsData[pos];
                            outputData[pos] = fromRhs ? rhsData[pos] : lhsData[pos];
                            outputMask.set(pos, fromRhs ? rhsMask.isOn(pos) : lhsMask.isOn(pos));
                        }
                    } else if (Operation == CSG_DIFFERENCE){
                        for (Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                            const ValueType rhsVal = math::negative(rhsData[pos]);
                            const bool fromRhs = lhsData[pos] < rhsVal;
                            outputData[pos] = fromRhs ? rhsVal : lhsData[pos];
                            outputMask.set(pos, fromRhs ? rhsMask.isOn(pos) : lhsMask.isOn(pos));
                        }
                    } else { // Union
                        for (Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                            const bool fromRhs = lhsData[pos] > rhsData[pos];
                            outputData[pos] = fromRhs ? rhsData[pos] : lhsData[pos];
                            outputMask.set(pos, fromRhs ? rhsMask.isOn(pos) : lhsMask.isOn(pos));
                        }
                    }

                } else {
                    if (Operation == CSG_INTERSECTION) {
                        if (rhsAcc.getValue(ijk) < ValueType(0.0)) {
                            outputAcc.addLeaf(new LeafNodeType(lhsNode));
                        }
                    } else { // Union & Difference
                        if (!(rhsAcc.getValue(ijk) < ValueType(0.0))) {
                            outputAcc.addLeaf(new LeafNodeType(lhsNode));
                        }
                    }
                }
            } //  end range loop
        }

        LeafNodeType const * const * const mLhsNodes;
        TreeType             const * const mRhsTree;
        TreeType                           mLocalTree;
        TreeType                   * const mOutputTree;
    }; // struct ProcessLeafNodes

    TreePtrType               mSegment;
    TreeType    const * const mLhsTree;
    TreeType    const * const mRhsTree;
}; // struct BuildPrimarySegment


template<typename TreeType, CSGOperation Operation>
struct BuildSecondarySegment
{
    typedef typename TreeType::ValueType                                            ValueType;
    typedef typename TreeType::Ptr                                                  TreePtrType;
    typedef typename TreeType::LeafNodeType                                         LeafNodeType;
    typedef typename LeafNodeType::NodeMaskType                                     NodeMaskType;
    typedef typename TreeType::RootNodeType                                         RootNodeType;
    typedef typename RootNodeType::NodeChainType                                    NodeChainType;
    typedef typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type      InternalNodeType;

    BuildSecondarySegment(const TreeType& lhs, const TreeType& rhs)
        : mSegment(new TreeType(lhs.background()))
        , mLhsTree(&lhs)
        , mRhsTree(&rhs)
    {
    }

    void operator()() const
    {
        std::vector<const LeafNodeType*> leafNodes;

        {
            std::vector<const InternalNodeType*> internalNodes;
            mRhsTree->getNodes(internalNodes);

            ProcessInternalNodes op(internalNodes, *mLhsTree, *mSegment, leafNodes);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), op);
        }

        ProcessLeafNodes op(leafNodes, *mLhsTree, *mSegment);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leafNodes.size()), op);
    }

    TreePtrType& segment() { return mSegment; }

private:

    struct ProcessInternalNodes {

        ProcessInternalNodes(std::vector<const InternalNodeType*>& rhsNodes, const TreeType& lhsTree,
            TreeType& outputTree, std::vector<const LeafNodeType*>& outputLeafNodes)
            : mRhsNodes(rhsNodes.empty() ? NULL : &rhsNodes.front())
            , mLhsTree(&lhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&outputTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&outputLeafNodes)
        {
        }

        ProcessInternalNodes(ProcessInternalNodes& other, tbb::split)
            : mRhsNodes(other.mRhsNodes)
            , mLhsTree(other.mLhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&mLocalLeafNodes)
        {
        }

        void join(ProcessInternalNodes& other)
        {
            mOutputTree->merge(*other.mOutputTree);
            mOutputLeafNodes->insert(mOutputLeafNodes->end(),
                other.mOutputLeafNodes->begin(), other.mOutputLeafNodes->end());
        }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            tree::ValueAccessor<const TreeType> lhsAcc(*mLhsTree);
            tree::ValueAccessor<TreeType>       outputAcc(*mOutputTree);

            std::vector<const LeafNodeType*> tmpLeafNodes;

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const InternalNodeType& rhsNode = *mRhsNodes[n];
                const Coord& ijk = rhsNode.origin();
                const InternalNodeType * lhsNode = lhsAcc.template probeConstNode<InternalNodeType>(ijk);

                if (lhsNode) {
                   rhsNode.getNodes(*mOutputLeafNodes);
                } else {
                    if (Operation == CSG_INTERSECTION) {
                        if (lhsAcc.getValue(ijk) < ValueType(0.0)) {
                            tmpLeafNodes.clear();
                            rhsNode.getNodes(tmpLeafNodes);
                            for (size_t i = 0, I = tmpLeafNodes.size(); i < I; ++i) {
                                outputAcc.addLeaf(new LeafNodeType(*tmpLeafNodes[i]));
                            }
                        }
                    } else if (Operation == CSG_DIFFERENCE) {
                        if (lhsAcc.getValue(ijk) < ValueType(0.0)) {
                            tmpLeafNodes.clear();
                            rhsNode.getNodes(tmpLeafNodes);
                            for (size_t i = 0, I = tmpLeafNodes.size(); i < I; ++i) {
                                LeafNodeType* outputNode = new LeafNodeType(*tmpLeafNodes[i]);
                                outputNode->negate();
                                outputAcc.addLeaf(outputNode);
                            }
                        }
                    } else { // Union
                        if (!(lhsAcc.getValue(ijk) < ValueType(0.0))) {
                            tmpLeafNodes.clear();
                            rhsNode.getNodes(tmpLeafNodes);
                            for (size_t i = 0, I = tmpLeafNodes.size(); i < I; ++i) {
                                outputAcc.addLeaf(new LeafNodeType(*tmpLeafNodes[i]));
                            }
                        }
                    }
                }
            } //  end range loop
        }

        InternalNodeType const * const * const mRhsNodes;
        TreeType                 const * const mLhsTree;
        TreeType                               mLocalTree;
        TreeType                       * const mOutputTree;

        std::vector<const LeafNodeType*>         mLocalLeafNodes;
        std::vector<const LeafNodeType*> * const mOutputLeafNodes;
    }; // struct ProcessInternalNodes

    struct ProcessLeafNodes {

        ProcessLeafNodes(std::vector<const LeafNodeType*>& rhsNodes, const TreeType& lhsTree, TreeType& output)
            : mRhsNodes(rhsNodes.empty() ? NULL : &rhsNodes.front())
            , mLhsTree(&lhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&output)
        {
        }

        ProcessLeafNodes(ProcessLeafNodes& rhs, tbb::split)
            : mRhsNodes(rhs.mRhsNodes)
            , mLhsTree(rhs.mLhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
        {
        }

        void join(ProcessLeafNodes& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            tree::ValueAccessor<const TreeType> lhsAcc(*mLhsTree);
            tree::ValueAccessor<TreeType>       outputAcc(*mOutputTree);

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const LeafNodeType& rhsNode = *mRhsNodes[n];
                const Coord& ijk = rhsNode.origin();

                const LeafNodeType* lhsNode = lhsAcc.probeConstLeaf(ijk);

                if (!lhsNode) {
                    if (Operation == CSG_INTERSECTION) {
                        if (lhsAcc.getValue(ijk) < ValueType(0.0)) {
                            outputAcc.addLeaf(new LeafNodeType(rhsNode));
                        }
                    } else if (Operation == CSG_DIFFERENCE) {
                        if (lhsAcc.getValue(ijk) < ValueType(0.0)) {
                            LeafNodeType* outputNode = new LeafNodeType(rhsNode);
                            outputNode->negate();
                            outputAcc.addLeaf(outputNode);
                        }
                    } else { // Union
                        if (!(lhsAcc.getValue(ijk) < ValueType(0.0))) {
                            outputAcc.addLeaf(new LeafNodeType(rhsNode));
                        }
                    }
                }
            } //  end range loop
        }

        LeafNodeType const * const * const mRhsNodes;
        TreeType             const * const mLhsTree;
        TreeType                           mLocalTree;
        TreeType                   * const mOutputTree;
    }; // struct ProcessLeafNodes

    TreePtrType               mSegment;
    TreeType    const * const mLhsTree;
    TreeType    const * const mRhsTree;
}; // struct BuildSecondarySegment


template<CSGOperation Operation, typename TreeType>
inline typename TreeType::Ptr
doCSGCopy(const TreeType& lhs, const TreeType& rhs)
{
    BuildPrimarySegment<TreeType, Operation> primary(lhs, rhs);
    BuildSecondarySegment<TreeType, Operation> secondary(lhs, rhs);

    // Exploiting nested parallelism
    tbb::task_group tasks;
    tasks.run(primary);
    tasks.run(secondary);
    tasks.wait();

    primary.segment()->merge(*secondary.segment());

    // The leafnode (level = 0) sign is set in the segment construction.
    tools::signedFloodFill(*primary.segment(), /*threaded=*/true, /*grainSize=*/1, /*minLevel=*/1);

    return primary.segment();
}


////////////////////////////////////////


template<typename TreeType>
struct GridOrTreeConstructor
{
    typedef typename TreeType::Ptr TreeTypePtr;
    static TreeTypePtr construct(const TreeType&, TreeTypePtr& tree) { return tree; }
};


template<typename TreeType>
struct GridOrTreeConstructor<Grid<TreeType> >
{
    typedef Grid<TreeType>                  GridType;
    typedef typename Grid<TreeType>::Ptr    GridTypePtr;
    typedef typename TreeType::Ptr          TreeTypePtr;

    static GridTypePtr construct(const GridType& grid, TreeTypePtr& tree) {
        GridTypePtr maskGrid(GridType::create(tree));
        maskGrid->setTransform(grid.transform().copy());
        maskGrid->insertMeta(grid);
        return maskGrid;
    }
};


////////////////////////////////////////

/// @cond COMPOSITE_INTERNAL
/// List of pairs of leaf node pointers
template <typename LeafT>
using LeafPairList = std::vector<std::pair<LeafT*, LeafT*>>;
/// @endcond

/// @cond COMPOSITE_INTERNAL
/// Transfers leaf nodes from a source tree into a
/// desitnation tree, unless it already exists in the destination tree
/// in which case pointers to both leaf nodes are added to a list for
/// subsequent compositing operations.
template <typename TreeT>
inline void transferLeafNodes(TreeT &srcTree, TreeT &dstTree,
                              LeafPairList<typename TreeT::LeafNodeType> &overlapping)
{
    using LeafT = typename TreeT::LeafNodeType;
    tree::ValueAccessor<TreeT> acc(dstTree);//destination
    std::vector<LeafT*> srcLeafNodes;
    srcLeafNodes.reserve(srcTree.leafCount());
    srcTree.stealNodes(srcLeafNodes);
    srcTree.clear();
    for (LeafT *srcLeaf : srcLeafNodes) {
        LeafT *dstLeaf = acc.probeLeaf(srcLeaf->origin());
        if (dstLeaf) {
            overlapping.emplace_back(dstLeaf, srcLeaf);//dst, src
        } else {
            acc.addLeaf(srcLeaf);
        }
    }
}
/// @endcond

/// @cond COMPOSITE_INTERNAL
/// Template specailization of compActiveLeafVoxels
template <typename TreeT, typename OpT>
inline
typename std::enable_if<!std::is_same<typename TreeT::ValueType, bool>::value &&
                        !std::is_same<typename TreeT::BuildType, ValueMask>::value &&
                         std::is_same<typename TreeT::LeafNodeType::Buffer::ValueType,
                                      typename TreeT::LeafNodeType::Buffer::StorageType>::value>::type
doCompActiveLeafVoxels(TreeT &srcTree, TreeT &dstTree, OpT op)
{
    using LeafT  = typename TreeT::LeafNodeType;
    LeafPairList<LeafT> overlapping;//dst, src
    transferLeafNodes(srcTree, dstTree, overlapping);

    using RangeT = tbb::blocked_range<size_t>;
    tbb::parallel_for(RangeT(0, overlapping.size()), [op, &overlapping](const RangeT& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *dstLeaf = overlapping[i].first, *srcLeaf = overlapping[i].second;
            dstLeaf->getValueMask() |= srcLeaf->getValueMask();
            auto *ptr = dstLeaf->buffer().data();
            for (auto v = srcLeaf->cbeginValueOn(); v; ++v) op(ptr[v.pos()], *v);
            delete srcLeaf;
        }
   });
}
/// @endcond

/// @cond COMPOSITE_INTERNAL
/// Template specailization of compActiveLeafVoxels
template <typename TreeT, typename OpT>
inline
typename std::enable_if<std::is_same<typename TreeT::BuildType, ValueMask>::value &&
                        std::is_same<typename TreeT::ValueType, bool>::value>::type
doCompActiveLeafVoxels(TreeT &srcTree, TreeT &dstTree, OpT)
{
    using LeafT  = typename TreeT::LeafNodeType;
    LeafPairList<LeafT> overlapping;//dst, src
    transferLeafNodes(srcTree, dstTree, overlapping);

    using RangeT = tbb::blocked_range<size_t>;
    tbb::parallel_for(RangeT(0, overlapping.size()), [&overlapping](const RangeT& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            overlapping[i].first->getValueMask() |= overlapping[i].second->getValueMask();
            delete overlapping[i].second;
        }
    });
}

/// @cond COMPOSITE_INTERNAL
/// Template specailization of compActiveLeafVoxels
template <typename TreeT, typename OpT>
inline
typename std::enable_if<std::is_same<typename TreeT::ValueType, bool>::value &&
                        !std::is_same<typename TreeT::BuildType, ValueMask>::value>::type
doCompActiveLeafVoxels(TreeT &srcTree, TreeT &dstTree, OpT op)
{
    using LeafT = typename TreeT::LeafNodeType;
    LeafPairList<LeafT> overlapping;//dst, src
    transferLeafNodes(srcTree, dstTree, overlapping);

    using RangeT = tbb::blocked_range<size_t>;
    using WordT = typename LeafT::Buffer::WordType;
    tbb::parallel_for(RangeT(0, overlapping.size()), [op, &overlapping](const RangeT& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            LeafT *dstLeaf = overlapping[i].first, *srcLeaf = overlapping[i].second;
            WordT *w1 = dstLeaf->buffer().data();
            const WordT *w2 = srcLeaf->buffer().data();
            const WordT *w3 = &(srcLeaf->getValueMask().template getWord<WordT>(0));
            for (Index32 n = LeafT::Buffer::WORD_COUNT; n--; ++w1) {
                WordT tmp = *w1, state = *w3++;
                op (tmp, *w2++);
                *w1 = (state & tmp) | (~state & *w1);//inactive values are unchanged
            }
            dstLeaf->getValueMask() |= srcLeaf->getValueMask();
            delete srcLeaf;
        }
    });
}
/// @endcond

/// @cond COMPOSITE_INTERNAL
/// Default functor for compActiveLeafVoxels
template <typename TreeT>
struct CopyOp
{
    using ValueT = typename TreeT::ValueType;
    CopyOp() = default;
    void operator()(ValueT& dst, const ValueT& src) const { dst = src; }
};
/// @endcond

} // namespace composite


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compMax(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT>    Adapter;
    typedef typename Adapter::TreeType  TreeT;
    typedef typename TreeT::ValueType   ValueT;
    struct Local {
        static inline void op(CombineArgs<ValueT>& args) {
            args.setResult(composite::max(args.a(), args.b()));
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compMin(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT>    Adapter;
    typedef typename Adapter::TreeType  TreeT;
    typedef typename TreeT::ValueType   ValueT;
    struct Local {
        static inline void op(CombineArgs<ValueT>& args) {
            args.setResult(composite::min(args.a(), args.b()));
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compSum(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    struct Local {
        static inline void op(CombineArgs<typename TreeT::ValueType>& args) {
            args.setResult(args.a() + args.b());
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compMul(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    struct Local {
        static inline void op(CombineArgs<typename TreeT::ValueType>& args) {
            args.setResult(args.a() * args.b());
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compDiv(GridOrTreeT& aTree, GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    struct Local {
        static inline void op(CombineArgs<typename TreeT::ValueType>& args) {
            args.setResult(composite::divide(args.a(), args.b()));
        }
    };
    Adapter::tree(aTree).combineExtended(Adapter::tree(bTree), Local::op, /*prune=*/false);
}


////////////////////////////////////////


template<typename TreeT>
struct CompReplaceOp
{
    TreeT* const aTree;

    CompReplaceOp(TreeT& _aTree): aTree(&_aTree) {}

    /// @note fill operation is not thread safe
    void operator()(const typename TreeT::ValueOnCIter& iter) const
    {
        CoordBBox bbox;
        iter.getBoundingBox(bbox);
        aTree->fill(bbox, *iter);
    }

    void operator()(const typename TreeT::LeafCIter& leafIter) const
    {
        tree::ValueAccessor<TreeT> acc(*aTree);
        for (typename TreeT::LeafCIter::LeafNodeT::ValueOnCIter iter =
            leafIter->cbeginValueOn(); iter; ++iter)
        {
            acc.setValue(iter.getCoord(), *iter);
        }
    }
};


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
compReplace(GridOrTreeT& aTree, const GridOrTreeT& bTree)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    typedef typename TreeT::ValueOnCIter ValueOnCIterT;

    // Copy active states (but not values) from B to A.
    Adapter::tree(aTree).topologyUnion(Adapter::tree(bTree));

    CompReplaceOp<TreeT> op(Adapter::tree(aTree));

    // Copy all active tile values from B to A.
    ValueOnCIterT iter = bTree.cbeginValueOn();
    iter.setMaxDepth(iter.getLeafDepth() - 1); // don't descend into leaf nodes
    foreach(iter, op, /*threaded=*/false);

    // Copy all active voxel values from B to A.
    foreach(Adapter::tree(bTree).cbeginLeaf(), op);
}


////////////////////////////////////////


/// Base visitor class for CSG operations
/// (not intended to be used polymorphically, so no virtual functions)
template<typename TreeType>
class CsgVisitorBase
{
public:
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = 3 };

    CsgVisitorBase(const TreeT& aTree, const TreeT& bTree):
        mAOutside(aTree.background()),
        mAInside(math::negative(mAOutside)),
        mBOutside(bTree.background()),
        mBInside(math::negative(mBOutside))
    {
        const ValueT zero = zeroVal<ValueT>();
        if (!(mAOutside > zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid A outside value > 0, got " << mAOutside);
        }
        if (!(mAInside < zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid A inside value < 0, got " << mAInside);
        }
        if (!(mBOutside > zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid B outside value > 0, got " << mBOutside);
        }
        if (!(mBInside < zero)) {
            OPENVDB_THROW(ValueError,
                "expected grid B outside value < 0, got " << mBOutside);
        }
    }

protected:
    ValueT mAOutside, mAInside, mBOutside, mBInside;
};


////////////////////////////////////////


template<typename TreeType>
struct CsgUnionVisitor: public CsgVisitorBase<TreeType>
{
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = CsgVisitorBase<TreeT>::STOP };

    CsgUnionVisitor(const TreeT& a, const TreeT& b): CsgVisitorBase<TreeT>(a, b) {}

    /// Don't process nodes that are at different tree levels.
    template<typename AIterT, typename BIterT>
    inline int operator()(AIterT&, BIterT&) { return 0; }

    /// Process root and internal nodes.
    template<typename IterT>
    inline int operator()(IterT& aIter, IterT& bIter)
    {
        ValueT aValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* aChild = aIter.probeChild(aValue);
        if (!aChild && aValue < zeroVal<ValueT>()) {
            // A is an inside tile.  Leave it alone and stop traversing this branch.
            return STOP;
        }

        ValueT bValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* bChild = bIter.probeChild(bValue);
        if (!bChild && bValue < zeroVal<ValueT>()) {
            // B is an inside tile.  Make A an inside tile and stop traversing this branch.
            aIter.setValue(this->mAInside);
            aIter.setValueOn(bIter.isValueOn());
            delete aChild;
            return STOP;
        }

        if (!aChild && aValue > zeroVal<ValueT>()) {
            // A is an outside tile.  If B has a child, transfer it to A,
            // otherwise leave A alone.
            if (bChild) {
                bIter.setValue(this->mBOutside);
                bIter.setValueOff();
                bChild->resetBackground(this->mBOutside, this->mAOutside);
                aIter.setChild(bChild); // transfer child
                delete aChild;
            }
            return STOP;
        }

        // If A has a child and B is an outside tile, stop traversing this branch.
        // Continue traversal only if A and B both have children.
        return (aChild && bChild) ? 0 : STOP;
    }

    /// Process leaf node values.
    inline int operator()(ChildIterT& aIter, ChildIterT& bIter)
    {
        ValueT aValue, bValue;
        aIter.probeValue(aValue);
        bIter.probeValue(bValue);
        if (aValue > bValue) { // a = min(a, b)
            aIter.setValue(bValue);
            aIter.setValueOn(bIter.isValueOn());
        }
        return 0;
    }
};



////////////////////////////////////////


template<typename TreeType>
struct CsgIntersectVisitor: public CsgVisitorBase<TreeType>
{
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = CsgVisitorBase<TreeT>::STOP };

    CsgIntersectVisitor(const TreeT& a, const TreeT& b): CsgVisitorBase<TreeT>(a, b) {}

    /// Don't process nodes that are at different tree levels.
    template<typename AIterT, typename BIterT>
    inline int operator()(AIterT&, BIterT&) { return 0; }

    /// Process root and internal nodes.
    template<typename IterT>
    inline int operator()(IterT& aIter, IterT& bIter)
    {
        ValueT aValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* aChild = aIter.probeChild(aValue);
        if (!aChild && !(aValue < zeroVal<ValueT>())) {
            // A is an outside tile.  Leave it alone and stop traversing this branch.
            return STOP;
        }

        ValueT bValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* bChild = bIter.probeChild(bValue);
        if (!bChild && !(bValue < zeroVal<ValueT>())) {
            // B is an outside tile.  Make A an outside tile and stop traversing this branch.
            aIter.setValue(this->mAOutside);
            aIter.setValueOn(bIter.isValueOn());
            delete aChild;
            return STOP;
        }

        if (!aChild && aValue < zeroVal<ValueT>()) {
            // A is an inside tile.  If B has a child, transfer it to A,
            // otherwise leave A alone.
            if (bChild) {
                bIter.setValue(this->mBOutside);
                bIter.setValueOff();
                bChild->resetBackground(this->mBOutside, this->mAOutside);
                aIter.setChild(bChild); // transfer child
                delete aChild;
            }
            return STOP;
        }

        // If A has a child and B is an outside tile, stop traversing this branch.
        // Continue traversal only if A and B both have children.
        return (aChild && bChild) ? 0 : STOP;
    }

    /// Process leaf node values.
    inline int operator()(ChildIterT& aIter, ChildIterT& bIter)
    {
        ValueT aValue, bValue;
        aIter.probeValue(aValue);
        bIter.probeValue(bValue);
        if (aValue < bValue) { // a = max(a, b)
            aIter.setValue(bValue);
            aIter.setValueOn(bIter.isValueOn());
        }
        return 0;
    }
};


////////////////////////////////////////


template<typename TreeType>
struct CsgDiffVisitor: public CsgVisitorBase<TreeType>
{
    typedef TreeType TreeT;
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType::ChildAllIter ChildIterT;

    enum { STOP = CsgVisitorBase<TreeT>::STOP };

    CsgDiffVisitor(const TreeT& a, const TreeT& b): CsgVisitorBase<TreeT>(a, b) {}

    /// Don't process nodes that are at different tree levels.
    template<typename AIterT, typename BIterT>
    inline int operator()(AIterT&, BIterT&) { return 0; }

    /// Process root and internal nodes.
    template<typename IterT>
    inline int operator()(IterT& aIter, IterT& bIter)
    {
        ValueT aValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* aChild = aIter.probeChild(aValue);
        if (!aChild && !(aValue < zeroVal<ValueT>())) {
            // A is an outside tile.  Leave it alone and stop traversing this branch.
            return STOP;
        }

        ValueT bValue = zeroVal<ValueT>();
        typename IterT::ChildNodeType* bChild = bIter.probeChild(bValue);
        if (!bChild && bValue < zeroVal<ValueT>()) {
            // B is an inside tile.  Make A an inside tile and stop traversing this branch.
            aIter.setValue(this->mAOutside);
            aIter.setValueOn(bIter.isValueOn());
            delete aChild;
            return STOP;
        }

        if (!aChild && aValue < zeroVal<ValueT>()) {
            // A is an inside tile.  If B has a child, transfer it to A,
            // otherwise leave A alone.
            if (bChild) {
                bIter.setValue(this->mBOutside);
                bIter.setValueOff();
                bChild->resetBackground(this->mBOutside, this->mAOutside);
                aIter.setChild(bChild); // transfer child
                bChild->negate();
                delete aChild;
            }
            return STOP;
        }

        // If A has a child and B is an outside tile, stop traversing this branch.
        // Continue traversal only if A and B both have children.
        return (aChild && bChild) ? 0 : STOP;
    }

    /// Process leaf node values.
    inline int operator()(ChildIterT& aIter, ChildIterT& bIter)
    {
        ValueT aValue, bValue;
        aIter.probeValue(aValue);
        bIter.probeValue(bValue);
        bValue = math::negative(bValue);
        if (aValue < bValue) { // a = max(a, -b)
            aIter.setValue(bValue);
            aIter.setValueOn(bIter.isValueOn());
        }
        return 0;
    }
};


////////////////////////////////////////


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
csgUnion(GridOrTreeT& a, GridOrTreeT& b, bool prune)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    TreeT &aTree = Adapter::tree(a), &bTree = Adapter::tree(b);
    CsgUnionVisitor<TreeT> visitor(aTree, bTree);
    aTree.visit2(bTree, visitor);
    if (prune) tools::pruneLevelSet(aTree);
}

template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
csgIntersection(GridOrTreeT& a, GridOrTreeT& b, bool prune)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    TreeT &aTree = Adapter::tree(a), &bTree = Adapter::tree(b);
    CsgIntersectVisitor<TreeT> visitor(aTree, bTree);
    aTree.visit2(bTree, visitor);
    if (prune) tools::pruneLevelSet(aTree);
}

template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline void
csgDifference(GridOrTreeT& a, GridOrTreeT& b, bool prune)
{
    typedef TreeAdapter<GridOrTreeT> Adapter;
    typedef typename Adapter::TreeType TreeT;
    TreeT &aTree = Adapter::tree(a), &bTree = Adapter::tree(b);
    CsgDiffVisitor<TreeT> visitor(aTree, bTree);
    aTree.visit2(bTree, visitor);
    if (prune) tools::pruneLevelSet(aTree);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline typename GridOrTreeT::Ptr
csgUnionCopy(const GridOrTreeT& a, const GridOrTreeT& b)
{
    typedef TreeAdapter<GridOrTreeT>            Adapter;
    typedef typename Adapter::TreeType::Ptr     TreePtrT;

    TreePtrT output = composite::doCSGCopy<composite::CSG_UNION>(
                        Adapter::tree(a), Adapter::tree(b));

    return composite::GridOrTreeConstructor<GridOrTreeT>::construct(a, output);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline typename GridOrTreeT::Ptr
csgIntersectionCopy(const GridOrTreeT& a, const GridOrTreeT& b)
{
    typedef TreeAdapter<GridOrTreeT>            Adapter;
    typedef typename Adapter::TreeType::Ptr     TreePtrT;

    TreePtrT output = composite::doCSGCopy<composite::CSG_INTERSECTION>(
                        Adapter::tree(a), Adapter::tree(b));

    return composite::GridOrTreeConstructor<GridOrTreeT>::construct(a, output);
}


template<typename GridOrTreeT>
OPENVDB_STATIC_SPECIALIZATION inline typename GridOrTreeT::Ptr
csgDifferenceCopy(const GridOrTreeT& a, const GridOrTreeT& b)
{
    typedef TreeAdapter<GridOrTreeT>            Adapter;
    typedef typename Adapter::TreeType::Ptr     TreePtrT;

    TreePtrT output = composite::doCSGCopy<composite::CSG_DIFFERENCE>(
                        Adapter::tree(a), Adapter::tree(b));

    return composite::GridOrTreeConstructor<GridOrTreeT>::construct(a, output);
}

////////////////////////////////////////////////////////

/// @brief Composite the active values in leaf nodes, i.e. active
///        voxels, of a source tree into a destination tree.
///
/// @param srcTree source tree from which active voxels are composited.
///
/// @param dstTree destination tree into which active voxels are composited.
///
/// @param op      a functor of the form <tt>void op(T& dst, const T& src)</tt>,
///                where @c T is the @c ValueType of the tree, that composites
///                a source value into a destination value. By default
///                it copies the value from src to dst.
///
/// @details All active voxels in the source tree will
///          be active in the destination tree, and their value is
///          determined by a use-defined functor (OpT op) that operates on the
///          source and destination values. The only exception is when
///          the tree type is MaskTree, in which case no functor is
///          needed since by defintion a MaskTree has no values (only topology).
///
/// @warning This function only operated on leaf node values,
///          i.e. tile values are ignored.
template <typename TreeT, typename OpT = composite::CopyOp<TreeT> >
inline void compActiveLeafVoxels(TreeT &srcTree, TreeT &dstTree, OpT op = composite::CopyOp<TreeT>())
{
    composite::doCompActiveLeafVoxels<TreeT, OpT>(srcTree, dstTree, op);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_COMPOSITE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
