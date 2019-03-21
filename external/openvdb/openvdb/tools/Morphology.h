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
/// @file   Morphology.h
///
/// @brief  Implementation of morphological dilation and erosion.
///
/// @note   By design the morphological operations only change the
///         state of voxels, not their values. If one desires to
///         change the values of voxels that change state an efficient
///         technique is to construct a boolean mask by performing a
///         topology difference between the original and final grids.
///
/// @todo   Extend erosion with 18 and 26 neighbors (coming soon!)
///
/// @author Ken Museth
///

#ifndef OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h> // for isApproxEqual()
#include <openvdb/tree/TreeIterator.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include "Prune.h"// for pruneLevelSet
#include "ValueTransformer.h" // for foreach()
#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Voxel topology of nearest neighbors
/// @details
/// <dl>
/// <dt><b>NN_FACE</b>
/// <dd>face adjacency (6 nearest neighbors, defined as all neighbor
/// voxels connected along one of the primary axes)
///
/// <dt><b>NN_FACE_EDGE</b>
/// <dd>face and edge adjacency (18 nearest neighbors, defined as all
/// neighbor voxels connected along either one or two of the primary axes)
///
/// <dt><b>NN_FACE_EDGE_VERTEX</b>
/// <dd>face, edge and vertex adjacency (26 nearest neighbors, defined
/// as all neighbor voxels connected along either one, two or all
/// three of the primary axes)
/// </dl>
enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };

/// @brief Different policies when dilating trees with active tiles
/// @details
/// <dl>
/// <dt><b>IGNORE_TILES</b>
/// <dd>Active tiles are ignores, i.e. only active voxels are dilates.
///
/// <dt><b>EXPAND_TILES</b>
/// <dd>Active tiles are expanded into active voxels and then dilated.
///
/// <dt><b>PRESERVE_TILES</b>
/// <dd>Active tiles remain unchanged but they still contribute to the
/// dilation as if they were active voxels.
/// </dl>
enum TilePolicy { IGNORE_TILES, EXPAND_TILES, PRESERVE_TILES };

/// @brief Topologically dilate all active values (i.e. both voxels
/// and tiles) in a tree using one of three nearest neighbor
/// connectivity patterns.
/// @note This method is fully multi-threaded and support active tiles!
///
/// @param tree          tree to be dilated
/// @param iterations    number of iterations to apply the dilation
/// @param nn            connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
/// @param mode          Defined the policy for handling active tiles
///                      (see above for details)
///
/// @note The values of any voxels are unchanged.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateActiveValues(TreeType& tree,
                               int iterations = 1,
                               NearestNeighbors nn = NN_FACE,
                               TilePolicy mode = PRESERVE_TILES);

/// @brief Topologically dilate all active values (i.e. both voxels
/// and tiles) in a tree using one of three nearest neighbor
/// connectivity patterns.
///
/// @warning Unlike the method above this one takes a LeafManger,
/// however (unlike dilateVoxels method below) it offers no performance
/// advantage over the one that takes a tree. Its merely included for
/// API compatability. The leaf nodes in the manger are updated
/// after the dilation, which incurres a (very small) overhead.
///
/// @note This method is fully multi-threaded and support active tiles!
///
/// @param manager       Leaf node manager for the tree to be dilated.
///                      On exit it is updated to include all the leaf
///                      nodes of the dilated tree.
/// @param iterations    number of iterations to apply the dilation
/// @param nn            connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
/// @param mode          Defined the policy for handling active tiles
///                      (see above for details)
///
/// @note The values of any voxels are unchanged.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateActiveValues(tree::LeafManager<TreeType>& manager,
                               int iterations = 1,
                               NearestNeighbors nn = NN_FACE,
                               TilePolicy mode = PRESERVE_TILES);


/// @brief Topologically dilate all leaf-level active voxels in a tree
/// using one of three nearest neighbor connectivity patterns.
/// @warning This method is NOT multi-threaded and ignores active tiles!
///
/// @param tree          tree to be dilated
/// @param iterations    number of iterations to apply the dilation
/// @param nn            connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
///
/// @note The values of any voxels are unchanged.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateVoxels(TreeType& tree,
                         int iterations = 1,
                         NearestNeighbors nn = NN_FACE);

/// @brief Topologically dilate all leaf-level active voxels in a tree
/// using one of three nearest neighbor connectivity patterns.
/// @warning This method is NOT multi-threaded and ignores active tiles!
///
/// @param manager       LeafManager containing the tree to be dilated.
///                      On exit it is updated to include all the leaf
///                      nodes of the dilated tree.
/// @param iterations    number of iterations to apply the dilation
/// @param nn           connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
///
/// @note The values of any voxels are unchanged.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateVoxels(tree::LeafManager<TreeType>& manager,
                         int iterations = 1,
                         NearestNeighbors nn = NN_FACE);


//@{
/// @brief Topologically erode all leaf-level active voxels in the given tree.
/// @details That is, shrink the set of active voxels by @a iterations voxels
/// in the +x, -x, +y, -y, +z and -z directions, but don't change the values
/// of any voxels, only their active states.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void erodeVoxels(TreeType& tree,
                        int iterations=1,
                        NearestNeighbors nn = NN_FACE);

template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void erodeVoxels(tree::LeafManager<TreeType>& manager,
                        int iterations = 1,
                        NearestNeighbors nn = NN_FACE);
//@}


/// @brief Mark as active any inactive tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
inline void activate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>()
);


/// @brief Mark as inactive any active tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
inline void deactivate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>()
);


////////////////////////////////////////


/// Mapping from a Log2Dim to a data type of size 2^Log2Dim bits
template<Index Log2Dim> struct DimToWord {};
template<> struct DimToWord<3> { typedef uint8_t Type; };
template<> struct DimToWord<4> { typedef uint16_t Type; };
template<> struct DimToWord<5> { typedef uint32_t Type; };
template<> struct DimToWord<6> { typedef uint64_t Type; };


////////////////////////////////////////


template<typename TreeType>
class Morphology
{
public:
    typedef tree::LeafManager<TreeType> ManagerType;

    Morphology(TreeType& tree):
        mOwnsManager(true), mManager(new ManagerType(tree)), mAcc(tree), mSteps(1) {}
    Morphology(ManagerType* mgr):
        mOwnsManager(false), mManager(mgr), mAcc(mgr->tree()), mSteps(1) {}
    virtual ~Morphology() { if (mOwnsManager) delete mManager; }

    /// @brief Face-adjacent dilation pattern
    void dilateVoxels6();
    /// @brief Face- and edge-adjacent dilation pattern.
    void dilateVoxels18();
    /// @brief Face-, edge- and vertex-adjacent dilation pattern.
    void dilateVoxels26();
    void dilateVoxels(int iterations = 1, NearestNeighbors nn = NN_FACE);

    /// @brief Face-adjacent erosion pattern.
    void erodeVoxels6()  { mSteps = 1; this->doErosion(NN_FACE); }
    /// @brief Face- and edge-adjacent erosion pattern.
    void erodeVoxels18() { mSteps = 1; this->doErosion(NN_FACE_EDGE); }
    /// @brief Face-, edge- and vertex-adjacent erosion pattern.
    void erodeVoxels26() { mSteps = 1; this->doErosion(NN_FACE_EDGE_VERTEX); }
    void erodeVoxels(int iterations = 1, NearestNeighbors nn = NN_FACE)
    {
        mSteps = iterations;
        this->doErosion(nn);
    }

protected:

    void doErosion(NearestNeighbors nn);

    typedef typename TreeType::LeafNodeType LeafType;
    typedef typename LeafType::NodeMaskType MaskType;
    typedef tree::ValueAccessor<TreeType>   AccessorType;

    const bool   mOwnsManager;
    ManagerType* mManager;
    AccessorType mAcc;
    int mSteps;

    static const int LEAF_DIM     = LeafType::DIM;
    static const int LEAF_LOG2DIM = LeafType::LOG2DIM;
    typedef typename DimToWord<LEAF_LOG2DIM>::Type Word;

    struct Neighbor {
        LeafType* leaf;//null if a tile
        bool      init;//true if initialization is required
        bool      isOn;//true if an active tile
        Neighbor() : leaf(nullptr), init(true) {}
        inline void clear() { leaf = nullptr; init = true; }
        template<int DX, int DY, int DZ>
        void scatter(AccessorType& acc, const Coord &xyz, int indx, Word mask)
        {
            if (init) {
                init = false;
                Coord orig = xyz.offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);
                leaf = acc.probeLeaf(orig);
                if ((leaf == nullptr) && !acc.isValueOn(orig)) leaf = acc.touchLeaf(orig);
            }
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
            static
#endif
            const int N = (LEAF_DIM - 1)*(DY + DX*LEAF_DIM);
            if (leaf) leaf->getValueMask().template getWord<Word>(indx-N) |= mask;
        }

        template<int DX, int DY, int DZ>
        Word gather(AccessorType& acc, const Coord &xyz, int indx)
        {
            if (init) {
                init = false;
                Coord orig = xyz.offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);
                leaf = acc.probeLeaf(orig);
                isOn = leaf ? false : acc.isValueOn(orig);
            }
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
            static
#endif
            const int N = (LEAF_DIM -1 )*(DY + DX*LEAF_DIM);
            return leaf ? leaf->getValueMask().template getWord<Word>(indx-N)
                : isOn ? ~Word(0) : Word(0);
        }
    };// Neighbor

    struct LeafCache
    {
        LeafCache(size_t n, TreeType& tree) : size(n), leafs(new LeafType*[n]), acc(tree)
        {
            onTile.setValuesOn();
            this->clear();
        }
        ~LeafCache() { delete [] leafs; }
        LeafType*& operator[](int offset) { return leafs[offset]; }
        inline void clear() { for (size_t i = 0; i < size; ++i) leafs[i] = nullptr; }
        inline void setOrigin(const Coord& xyz) { origin = &xyz; }
        inline void scatter(int n, int indx)
        {
            assert(leafs[n]);
            leafs[n]->getValueMask().template getWord<Word>(indx) |= mask;
        }
        template<int DX, int DY, int DZ>
        inline void scatter(int n, int indx)
        {
            if (!leafs[n]) {
                const Coord xyz = origin->offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);
                leafs[n] = acc.probeLeaf(xyz);
                if (!leafs[n]) leafs[n] = acc.isValueOn(xyz) ? &onTile : acc.touchLeaf(xyz);
            }
            this->scatter(n, indx - (LEAF_DIM - 1)*(DY + DX*LEAF_DIM));
        }
        inline Word gather(int n, int indx)
        {
            assert(leafs[n]);
            return leafs[n]->getValueMask().template getWord<Word>(indx);
        }
        template<int DX, int DY, int DZ>
        inline Word gather(int n, int indx)
        {
            if (!leafs[n]) {
                const Coord xyz = origin->offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);
                leafs[n] = acc.probeLeaf(xyz);
                if (!leafs[n]) leafs[n] = acc.isValueOn(xyz) ? &onTile : &offTile;
            }
            return this->gather(n, indx - (LEAF_DIM -1 )*(DY + DX*LEAF_DIM));
        }
        // Scatters in the xy face-directions relative to leaf i1
        void scatterFacesXY(int x, int y, int i1, int n, int i2);

        // Scatters in the xy edge-directions relative to leaf i1
        void scatterEdgesXY(int x, int y, int i1, int n, int i2);

        Word gatherFacesXY(int x, int y, int i1, int n, int i2);

        Word gatherEdgesXY(int x, int y, int i1, int n, int i2);

        const Coord* origin;
        size_t size;
        LeafType** leafs;
        LeafType onTile, offTile;
        AccessorType acc;
        Word mask;
    };// LeafCache

    struct ErodeVoxelsOp {
        typedef tbb::blocked_range<size_t> RangeT;
        ErodeVoxelsOp(std::vector<MaskType>& masks, ManagerType& manager)
            : mTask(0), mSavedMasks(masks) , mManager(manager) {}
        void runParallel(NearestNeighbors nn);
        void operator()(const RangeT& r) const {mTask(const_cast<ErodeVoxelsOp*>(this), r);}
        void erode6( const RangeT&) const;
        void erode18(const RangeT&) const;
        void erode26(const RangeT&) const;
    private:
        using FuncT = typename std::function<void (ErodeVoxelsOp*, const RangeT&)>;
        FuncT                  mTask;
        std::vector<MaskType>& mSavedMasks;
        ManagerType&           mManager;
    };// ErodeVoxelsOp

    struct MaskManager {
        MaskManager(std::vector<MaskType>& masks, ManagerType& manager)
            : mMasks(masks) , mManager(manager), mSaveMasks(true) {}

        void save() { mSaveMasks = true; tbb::parallel_for(mManager.getRange(), *this); }
        void update() { mSaveMasks = false; tbb::parallel_for(mManager.getRange(), *this); }
        void operator()(const tbb::blocked_range<size_t>& range) const
        {
            if (mSaveMasks) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    mMasks[i] = mManager.leaf(i).getValueMask();
                }
            } else {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    mManager.leaf(i).setValueMask(mMasks[i]);
                }
            }
        }
    private:
        std::vector<MaskType>& mMasks;
        ManagerType& mManager;
        bool mSaveMasks;
    };// MaskManager

    struct UpdateMasks {
        UpdateMasks(const std::vector<MaskType>& masks, ManagerType& manager)
            : mMasks(masks), mManager(manager) {}
        void update() { tbb::parallel_for(mManager.getRange(), *this); }
        void operator()(const tbb::blocked_range<size_t>& r) const {
            for (size_t i=r.begin(); i<r.end(); ++i) mManager.leaf(i).setValueMask(mMasks[i]);
        }
        const std::vector<MaskType>& mMasks;
        ManagerType& mManager;
    };
    struct CopyMasks {
        CopyMasks(std::vector<MaskType>& masks, const ManagerType& manager)
            : mMasks(masks), mManager(manager) {}
        void copy() { tbb::parallel_for(mManager.getRange(), *this); }
        void operator()(const tbb::blocked_range<size_t>& r) const {
            for (size_t i=r.begin(); i<r.end(); ++i) mMasks[i]=mManager.leaf(i).getValueMask();
        }
        std::vector<MaskType>& mMasks;
        const ManagerType& mManager;
    };
    void copyMasks(std::vector<MaskType>& a, const ManagerType& b) {CopyMasks c(a, b); c.copy();}
};// Morphology


template<typename TreeType>
inline void
Morphology<TreeType>::dilateVoxels(int iterations, NearestNeighbors nn)
{
    for (int i=0; i<iterations; ++i) {
        switch (nn) {
        case NN_FACE_EDGE:
            this->dilateVoxels18();
            break;
        case NN_FACE_EDGE_VERTEX:
            this->dilateVoxels26();
            break;
        case NN_FACE:
        default:
            this->dilateVoxels6();
        }
    }
}


template<typename TreeType>
inline void
Morphology<TreeType>::dilateVoxels6()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = static_cast<int>(mManager->leafCount());

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    this->copyMasks(savedMasks, *mManager);
    LeafCache cache(7, mManager->tree());
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        const MaskType& oldMask = savedMasks[leafIdx];//original bit-mask of current leaf node
        cache[0] = &mManager->leaf(leafIdx);
        cache.setOrigin(cache[0]->origin());
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that corresponds to a row in z.
                if (const Word w = oldMask.template getWord<Word>(n)) {

                    // Dilate the current leaf in the +z and -z direction
                    cache.mask = Word(w | (w>>1) | (w<<1)); cache.scatter(0, n);

                    // Dilate into neighbor leaf in the -z direction
                    if ( (cache.mask = Word(w<<(LEAF_DIM-1))) ) {
                        cache.template scatter< 0, 0,-1>(1, n);
                    }
                    // Dilate into neighbor leaf in the +z direction
                    if ( (cache.mask = Word(w>>(LEAF_DIM-1))) ) {
                        cache.template scatter< 0, 0, 1>(2, n);
                    }
                    // Dilate in the xy-face directions relative to the center leaf
                    cache.mask = w; cache.scatterFacesXY(x, y, 0, n, 3);
                }
            }// loop over y
        }//loop over x
        cache.clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}//dilateVoxels6


template<typename TreeType>
inline void
Morphology<TreeType>::dilateVoxels18()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = static_cast<int>(mManager->leafCount());

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    this->copyMasks(savedMasks, *mManager);
    LeafCache cache(19, mManager->tree());
    Coord orig_mz, orig_pz;//origins of neighbor leaf nodes in the -z and +z directions
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        const MaskType& oldMask = savedMasks[leafIdx];//original bit-mask of current leaf node
        cache[0] = &mManager->leaf(leafIdx);
        orig_mz = cache[0]->origin().offsetBy(0, 0, -LEAF_DIM);
        orig_pz = cache[0]->origin().offsetBy(0, 0,  LEAF_DIM);
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                if (const Word w = oldMask.template getWord<Word>(n)) {
                    {
                        cache.mask = Word(w | (w>>1) | (w<<1));
                        cache.setOrigin(cache[0]->origin());
                        cache.scatter(0, n);
                        cache.scatterFacesXY(x, y, 0, n, 3);
                        cache.mask = w;
                        cache.scatterEdgesXY(x, y, 0, n, 3);
                    }
                    if ( (cache.mask = Word(w<<(LEAF_DIM-1))) ) {
                        cache.setOrigin(cache[0]->origin());
                        cache.template scatter< 0, 0,-1>(1, n);
                        cache.setOrigin(orig_mz);
                        cache.scatterFacesXY(x, y, 1, n, 11);
                    }
                    if ( (cache.mask = Word(w>>(LEAF_DIM-1))) ) {
                        cache.setOrigin(cache[0]->origin());
                        cache.template scatter< 0, 0, 1>(2, n);
                        cache.setOrigin(orig_pz);
                        cache.scatterFacesXY(x, y, 2, n, 15);
                    }
                }
            }// loop over y
        }//loop over x
        cache.clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}// dilateVoxels18


template<typename TreeType>
inline void
Morphology<TreeType>::dilateVoxels26()
{
    const int leafCount = static_cast<int>(mManager->leafCount());

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    this->copyMasks(savedMasks, *mManager);
    LeafCache cache(27, mManager->tree());
    Coord orig_mz, orig_pz;//origins of neighbor leaf nodes in the -z and +z directions
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        const MaskType& oldMask = savedMasks[leafIdx];//original bit-mask of current leaf node
        cache[0] = &mManager->leaf(leafIdx);
        orig_mz = cache[0]->origin().offsetBy(0, 0, -LEAF_DIM);
        orig_pz = cache[0]->origin().offsetBy(0, 0,  LEAF_DIM);
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                if (const Word w = oldMask.template getWord<Word>(n)) {
                    {
                        cache.mask = Word(w | (w>>1) | (w<<1));
                        cache.setOrigin(cache[0]->origin());
                        cache.scatter(0, n);
                        cache.scatterFacesXY(x, y, 0, n, 3);
                        cache.scatterEdgesXY(x, y, 0, n, 3);
                    }
                    if ( (cache.mask = Word(w<<(LEAF_DIM-1))) ) {
                        cache.setOrigin(cache[0]->origin());
                        cache.template scatter< 0, 0,-1>(1, n);
                        cache.setOrigin(orig_mz);
                        cache.scatterFacesXY(x, y, 1, n, 11);
                        cache.scatterEdgesXY(x, y, 1, n, 11);
                    }
                    if ( (cache.mask = Word(w>>(LEAF_DIM-1))) ) {
                        cache.setOrigin(cache[0]->origin());
                        cache.template scatter< 0, 0, 1>(2, n);
                        cache.setOrigin(orig_pz);
                        cache.scatterFacesXY(x, y, 2, n, 19);
                        cache.scatterEdgesXY(x, y, 2, n, 19);
                    }
                }
            }// loop over y
        }//loop over x
        cache.clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}// dilateVoxels26


template<typename TreeType>
inline void
Morphology<TreeType>::LeafCache::scatterFacesXY(int x, int y, int i1, int n, int i2)
{
    // dilate current leaf or neighbor in the -x direction
    if (x > 0) {
        this->scatter(i1, n-LEAF_DIM);
    } else {
        this->template scatter<-1, 0, 0>(i2, n);
    }
    // dilate current leaf or neighbor in the +x direction
    if (x < LEAF_DIM-1) {
        this->scatter(i1, n+LEAF_DIM);
    } else {
        this->template scatter< 1, 0, 0>(i2+1, n);
    }
    // dilate current leaf or neighbor in the -y direction
    if (y > 0) {
        this->scatter(i1, n-1);
    } else {
        this->template scatter< 0,-1, 0>(i2+2, n);
    }
    // dilate current leaf or neighbor in the +y direction
    if (y < LEAF_DIM-1) {
        this->scatter(i1, n+1);
    } else {
        this->template scatter< 0, 1, 0>(i2+3, n);
    }
}


template<typename TreeType>
inline void
Morphology<TreeType>::LeafCache::scatterEdgesXY(int x, int y, int i1, int n, int i2)
{
    if (x > 0) {
        if (y > 0) {
            this->scatter(i1, n-LEAF_DIM-1);
        } else {
            this->template scatter< 0,-1, 0>(i2+2, n-LEAF_DIM);
        }
        if (y < LEAF_DIM-1) {
            this->scatter(i1, n-LEAF_DIM+1);
        } else {
            this->template scatter< 0, 1, 0>(i2+3, n-LEAF_DIM);
        }
    } else {
        if (y < LEAF_DIM-1) {
            this->template scatter<-1, 0, 0>(i2  , n+1);
        } else {
            this->template scatter<-1, 1, 0>(i2+7, n  );
        }
        if (y > 0) {
            this->template scatter<-1, 0, 0>(i2  , n-1);
        } else {
            this->template scatter<-1,-1, 0>(i2+4, n  );
        }
    }
    if (x < LEAF_DIM-1) {
        if (y > 0) {
            this->scatter(i1, n+LEAF_DIM-1);
        } else {
            this->template scatter< 0,-1, 0>(i2+2, n+LEAF_DIM);
        }
        if (y < LEAF_DIM-1) {
            this->scatter(i1, n+LEAF_DIM+1);
        } else {
            this->template scatter< 0, 1, 0>(i2+3, n+LEAF_DIM);
        }
    } else {
        if (y > 0) {
            this->template scatter< 1, 0, 0>(i2+1, n-1);
        } else {
            this->template scatter< 1,-1, 0>(i2+6, n  );
        }
        if (y < LEAF_DIM-1) {
            this->template scatter< 1, 0, 0>(i2+1, n+1);
        } else {
            this->template scatter< 1, 1, 0>(i2+5, n  );
        }
    }
}


template<typename TreeType>
inline void
Morphology<TreeType>::ErodeVoxelsOp::runParallel(NearestNeighbors nn)
{
    namespace ph = std::placeholders;
    switch (nn) {
    case NN_FACE_EDGE:
        mTask = std::bind(&ErodeVoxelsOp::erode18, ph::_1, ph::_2);
        break;
    case NN_FACE_EDGE_VERTEX:
        mTask = std::bind(&ErodeVoxelsOp::erode26, ph::_1, ph::_2);
        break;
    case NN_FACE:
    default:
        mTask = std::bind(&ErodeVoxelsOp::erode6, ph::_1, ph::_2);
    }
    tbb::parallel_for(mManager.getRange(), *this);
}


template<typename TreeType>
inline typename Morphology<TreeType>::Word
Morphology<TreeType>::LeafCache::gatherFacesXY(int x, int y, int i1, int n, int i2)
{
    // erode current leaf or neighbor in negative x-direction
    Word w = x>0 ? this->gather(i1,n-LEAF_DIM) : this->template gather<-1,0,0>(i2, n);

    // erode current leaf or neighbor in positive x-direction
    w = Word(w & (x<LEAF_DIM-1?this->gather(i1,n+LEAF_DIM):this->template gather<1,0,0>(i2+1,n)));

    // erode current leaf or neighbor in negative y-direction
    w = Word(w & (y>0 ? this->gather(i1, n-1) : this->template gather<0,-1,0>(i2+2, n)));

    // erode current leaf or neighbor in positive y-direction
    w = Word(w & (y<LEAF_DIM-1 ? this->gather(i1, n+1) : this->template gather<0,1,0>(i2+3, n)));

    return w;
}


template<typename TreeType>
inline typename Morphology<TreeType>::Word
Morphology<TreeType>::LeafCache::gatherEdgesXY(int x, int y, int i1, int n, int i2)
{
    Word w = ~Word(0);

    if (x > 0) {
        w &= y > 0 ?          this->gather(i1, n-LEAF_DIM-1) :
                              this->template gather< 0,-1, 0>(i2+2, n-LEAF_DIM);
        w &= y < LEAF_DIM-1 ? this->gather(i1, n-LEAF_DIM+1) :
                              this->template gather< 0, 1, 0>(i2+3, n-LEAF_DIM);
    } else {
        w &= y < LEAF_DIM-1 ? this->template gather<-1, 0, 0>(i2  , n+1):
                              this->template gather<-1, 1, 0>(i2+7, n  );
        w &= y > 0 ?          this->template gather<-1, 0, 0>(i2  , n-1):
                              this->template gather<-1,-1, 0>(i2+4, n  );
    }
    if (x < LEAF_DIM-1) {
        w &= y > 0 ?          this->gather(i1, n+LEAF_DIM-1) :
                              this->template gather< 0,-1, 0>(i2+2, n+LEAF_DIM);
        w &= y < LEAF_DIM-1 ? this->gather(i1, n+LEAF_DIM+1) :
                              this->template gather< 0, 1, 0>(i2+3, n+LEAF_DIM);
    } else {
        w &= y > 0          ? this->template gather< 1, 0, 0>(i2+1, n-1):
                              this->template gather< 1,-1, 0>(i2+6, n  );
        w &= y < LEAF_DIM-1 ? this->template gather< 1, 0, 0>(i2+1, n+1):
                              this->template gather< 1, 1, 0>(i2+5, n  );
    }

    return w;
}


template <typename TreeType>
inline void
Morphology<TreeType>::ErodeVoxelsOp::erode6(const RangeT& range) const
{
    LeafCache cache(7, mManager.tree());
    for (size_t leafIdx = range.begin(); leafIdx < range.end(); ++leafIdx) {
        cache[0] = &mManager.leaf(leafIdx);
        if (cache[0]->isEmpty()) continue;
        cache.setOrigin(cache[0]->origin());
        MaskType& newMask = mSavedMasks[leafIdx];//original bit-mask of current leaf node
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that corresponds to a row in z.
                if (Word& w = newMask.template getWord<Word>(n)) {

                    // erode in two z directions (this is first since it uses the original w)
                    w = Word(w &
                        (Word(w<<1 | (cache.template gather<0,0,-1>(1, n)>>(LEAF_DIM-1))) &
                         Word(w>>1 | (cache.template gather<0,0, 1>(2, n)<<(LEAF_DIM-1)))));

                    w = Word(w & cache.gatherFacesXY(x, y, 0, n, 3));
                }
            }// loop over y
        }//loop over x
        cache.clear();
    }//loop over leafs
}


template <typename TreeType>
inline void
Morphology<TreeType>::ErodeVoxelsOp::erode18(const RangeT&) const
{
    OPENVDB_THROW(NotImplementedError, "tools::erode18 is not implemented yet!");
}


template <typename TreeType>
inline void
Morphology<TreeType>::ErodeVoxelsOp::erode26(const RangeT&) const
{
    OPENVDB_THROW(NotImplementedError, "tools::erode26 is not implemented yet!");
}


template<typename TreeType>
inline void
Morphology<TreeType>::doErosion(NearestNeighbors nn)
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const size_t leafCount = mManager->leafCount();

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    this->copyMasks(savedMasks, *mManager);
    UpdateMasks a(savedMasks, *mManager);
    ErodeVoxelsOp erode(savedMasks, *mManager);

    for (int i = 0; i < mSteps; ++i) {
        erode.runParallel(nn);
        a.update();
    }

    tools::pruneLevelSet(mManager->tree());
}


////////////////////////////////////////


template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateVoxels(tree::LeafManager<TreeType>& manager, int iterations, NearestNeighbors nn)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(&manager);
        m.dilateVoxels(iterations, nn);
    }
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateVoxels(TreeType& tree, int iterations, NearestNeighbors nn)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(tree);
        m.dilateVoxels(iterations, nn);
    }
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
erodeVoxels(tree::LeafManager<TreeType>& manager, int iterations, NearestNeighbors nn)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(&manager);
        m.erodeVoxels(iterations, nn);
    }
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
erodeVoxels(TreeType& tree, int iterations, NearestNeighbors nn)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(tree);
        m.erodeVoxels(iterations, nn);
    }
}


////////////////////////////////////////


namespace activation {

template<typename TreeType>
class ActivationOp
{
public:
    typedef typename TreeType::ValueType ValueT;

    ActivationOp(bool state, const ValueT& val, const ValueT& tol)
        : mActivate(state)
        , mValue(val)
        , mTolerance(tol)
    {}

    void operator()(const typename TreeType::ValueOnIter& it) const
    {
        if (math::isApproxEqual(*it, mValue, mTolerance)) {
            it.setValueOff();
        }
    }

    void operator()(const typename TreeType::ValueOffIter& it) const
    {
        if (math::isApproxEqual(*it, mValue, mTolerance)) {
            it.setActiveState(/*on=*/true);
        }
    }

    void operator()(const typename TreeType::LeafIter& lit) const
    {
        typedef typename TreeType::LeafNodeType LeafT;
        LeafT& leaf = *lit;
        if (mActivate) {
            for (typename LeafT::ValueOffIter it = leaf.beginValueOff(); it; ++it) {
                if (math::isApproxEqual(*it, mValue, mTolerance)) {
                    leaf.setValueOn(it.pos());
                }
            }
        } else {
            for (typename LeafT::ValueOnIter it = leaf.beginValueOn(); it; ++it) {
                if (math::isApproxEqual(*it, mValue, mTolerance)) {
                    leaf.setValueOff(it.pos());
                }
            }
        }
    }

private:
    bool mActivate;
    const ValueT mValue, mTolerance;
}; // class ActivationOp

} // namespace activation


template<typename GridOrTree>
inline void
activate(GridOrTree& gridOrTree, const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance)
{
    typedef TreeAdapter<GridOrTree> Adapter;
    typedef typename Adapter::TreeType TreeType;

    TreeType& tree = Adapter::tree(gridOrTree);

    activation::ActivationOp<TreeType> op(/*activate=*/true, value, tolerance);

    // Process all leaf nodes in parallel.
    foreach(tree.beginLeaf(), op);

    // Process all other inactive values serially (because changing active states
    // is not thread-safe unless no two threads modify the same node).
    typename TreeType::ValueOffIter it = tree.beginValueOff();
    it.setMaxDepth(tree.treeDepth() - 2);
    foreach(it, op, /*threaded=*/false);
}


template<typename GridOrTree>
inline void
deactivate(GridOrTree& gridOrTree, const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance)
{
    typedef TreeAdapter<GridOrTree> Adapter;
    typedef typename Adapter::TreeType TreeType;

    TreeType& tree = Adapter::tree(gridOrTree);

    activation::ActivationOp<TreeType> op(/*activate=*/false, value, tolerance);

    // Process all leaf nodes in parallel.
    foreach(tree.beginLeaf(), op);

    // Process all other active values serially (because changing active states
    // is not thread-safe unless no two threads modify the same node).
    typename TreeType::ValueOnIter it = tree.beginValueOn();
    it.setMaxDepth(tree.treeDepth() - 2);
    foreach(it, op, /*threaded=*/false);
}

/// @brief Class that performs multi-threaded dilation with support for active tiles.
/// @warning Dont use this class directly, instead call the function dilateActiveValues!
template<typename TreeT>
class DilationOp
{
    typedef typename TreeT::template ValueConverter<ValueMask>::Type MaskT;
    typedef tbb::enumerable_thread_specific<MaskT>                   PoolT;
    typedef typename MaskT::LeafNodeType                             LeafT;

    // Very light-weight member data
    const int mIter;// number of iterations
    const tools::NearestNeighbors mNN;//enum to specify the dilation scheme
    PoolT  *mPool;// pointer to the thread-local pool of mask trees
    LeafT **mLeafs;// raw array of pointers to leaf nodes

public:

    DilationOp(TreeT &tree, int iterations, NearestNeighbors nn, TilePolicy mode)
        : mIter(iterations), mNN(nn), mPool(nullptr), mLeafs(nullptr)
    {
        const size_t numLeafs = this->init( tree, mode );
        const size_t numThreads = size_t(tbb::task_scheduler_init::default_num_threads());
        const size_t grainSize = math::Max(size_t(1), numLeafs/(2*numThreads));

        MaskT mask;
        PoolT pool(mask);// Scoped thread-local storage of mask trees
        mPool = &pool;

        tbb::parallel_for(tbb::blocked_range<LeafT**>(mLeafs, mLeafs+numLeafs, grainSize), *this);

        delete [] mLeafs;// no more need for the array of leaf node pointers

        typedef typename PoolT::iterator IterT;
        for (IterT it=pool.begin(); it!=pool.end(); ++it) mask.merge(*it);// fast stealing

        if (mode == PRESERVE_TILES) tools::prune(mask);//multithreaded

        tree.topologyUnion(mask);//multithreaded
    }

    // This is required by tbb and should never be called directly
    void operator()(const tbb::blocked_range<LeafT**> &r) const
    {
        MaskT mask;// thread-local temporary mask tree
        for (LeafT** it=r.begin(); it!=r.end(); ++it) mask.addLeaf( *it );
        tree::LeafManager<MaskT> manager(mask, r.begin(), r.end());
        tools::dilateVoxels(manager, mIter, mNN);// serial dilation of active voxels
        mPool->local().merge(mask, MERGE_ACTIVE_STATES);
    }
private:

    // Simple wrapper of a raw double-pointer to mimic a std container
    struct MyArray {
        typedef LeafT* value_type;//required by Tree::stealNodes
        value_type* ptr;
        MyArray(value_type* array) : ptr(array) {}
        void push_back(value_type leaf) { *ptr++ = leaf; }//required by Tree::stealNodes
    };

    // Convert active tiles to leafs and de-construct the tree into a linear array of leafs.
    size_t linearize(MaskT& mask, TilePolicy mode)
    {
        if (mode != IGNORE_TILES) mask.voxelizeActiveTiles();//lightweight since this is a mask tree
        const size_t numLeafs = mask.leafCount();
        mLeafs = new LeafT*[numLeafs];// fast pre-allocation
        MyArray tmp(mLeafs);
        mask.stealNodes(tmp);// serializes the mask tree and leaves it empty
        return numLeafs;
    }

    template<typename T>
    typename std::enable_if<std::is_same<T, MaskT>::value, size_t>::type
    init(T& tree, TilePolicy mode)
    {
        return this->linearize(tree, mode);
    }

    template<typename T>
    typename std::enable_if<!std::is_same<T, MaskT>::value, size_t>::type
    init(const T& tree, TilePolicy mode)
    {
        MaskT mask(tree, false, true, TopologyCopy());
        return this->linearize(mask, mode);
    }

};// DilationOp

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateActiveValues(TreeType& tree, int iterations, NearestNeighbors nn, TilePolicy mode)
{
    if (iterations > 0 ) DilationOp<TreeType> tmp(tree, iterations, nn, mode);
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateActiveValues(tree::LeafManager<TreeType>& manager,
                   int iterations,
                   NearestNeighbors nn,
                   TilePolicy mode)
{
    if (iterations > 0 ) {
        DilationOp<TreeType> tmp(manager.tree(), iterations, nn, mode);
        manager.rebuildLeafArray();
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
