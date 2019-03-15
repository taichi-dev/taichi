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

#ifndef OPENVDB_UTIL_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_UTIL_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/Prune.h>// for tree::pruneInactive


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

OPENVDB_API extern const Index32 INVALID_IDX;

/// @brief coordinate offset table for neighboring voxels
OPENVDB_API extern const Coord COORD_OFFSETS[26];


////////////////////////////////////////


/// Return @a voxelCoord rounded to the closest integer coordinates.
inline Coord
nearestCoord(const Vec3d& voxelCoord)
{
    Coord ijk;
    ijk[0] = int(std::floor(voxelCoord[0]));
    ijk[1] = int(std::floor(voxelCoord[1]));
    ijk[2] = int(std::floor(voxelCoord[2]));
    return ijk;
}


////////////////////////////////////////


/// @brief Functor for use with tools::foreach() to compute the boolean intersection
/// between the value masks of corresponding leaf nodes in two trees
template<class TreeType1, class TreeType2>
class LeafTopologyIntOp
{
public:
    LeafTopologyIntOp(const TreeType2& tree): mOtherTree(&tree) {}

    inline void operator()(const typename TreeType1::LeafIter& lIter) const
    {
        const Coord xyz = lIter->origin();
        const typename TreeType2::LeafNodeType* leaf = mOtherTree->probeConstLeaf(xyz);
        if (leaf) {//leaf node
            lIter->topologyIntersection(*leaf, zeroVal<typename TreeType1::ValueType>());
        } else if (!mOtherTree->isValueOn(xyz)) {//inactive tile
            lIter->setValuesOff();
        }
    }

private:
    const TreeType2* mOtherTree;
};


/// @brief Functor for use with tools::foreach() to compute the boolean difference
/// between the value masks of corresponding leaf nodes in two trees
template<class TreeType1, class TreeType2>
class LeafTopologyDiffOp
{
public:
    LeafTopologyDiffOp(const TreeType2& tree): mOtherTree(&tree) {}

    inline void operator()(const typename TreeType1::LeafIter& lIter) const
    {
        const Coord xyz = lIter->origin();
        const typename TreeType2::LeafNodeType* leaf = mOtherTree->probeConstLeaf(xyz);
        if (leaf) {//leaf node
            lIter->topologyDifference(*leaf, zeroVal<typename TreeType1::ValueType>());
        } else if (mOtherTree->isValueOn(xyz)) {//active tile
            lIter->setValuesOff();
        }
    }

private:
    const TreeType2* mOtherTree;
};


////////////////////////////////////////


/// @brief Perform a boolean intersection between two leaf nodes' topology masks.
/// @return a pointer to a new, boolean-valued tree containing the overlapping voxels.
template<class TreeType1, class TreeType2>
inline typename TreeType1::template ValueConverter<bool>::Type::Ptr
leafTopologyIntersection(const TreeType1& lhs, const TreeType2& rhs, bool threaded = true)
{
    typedef typename TreeType1::template ValueConverter<bool>::Type BoolTreeType;

    typename BoolTreeType::Ptr topologyTree(new BoolTreeType(
        lhs, /*inactiveValue=*/false, /*activeValue=*/true, TopologyCopy()));

    tools::foreach(topologyTree->beginLeaf(),
        LeafTopologyIntOp<BoolTreeType, TreeType2>(rhs), threaded);

    tools::pruneInactive(*topologyTree, threaded);
    return topologyTree;
}


/// @brief Perform a boolean difference between two leaf nodes' topology masks.
/// @return a pointer to a new, boolean-valued tree containing the non-overlapping
/// voxels from the lhs.
template<class TreeType1, class TreeType2>
inline typename TreeType1::template ValueConverter<bool>::Type::Ptr
leafTopologyDifference(const TreeType1& lhs, const TreeType2& rhs, bool threaded = true)
{
    typedef typename TreeType1::template ValueConverter<bool>::Type BoolTreeType;

    typename BoolTreeType::Ptr topologyTree(new BoolTreeType(
        lhs, /*inactiveValue=*/false, /*activeValue=*/true, TopologyCopy()));

    tools::foreach(topologyTree->beginLeaf(),
        LeafTopologyDiffOp<BoolTreeType, TreeType2>(rhs), threaded);

    tools::pruneInactive(*topologyTree, threaded);
    return topologyTree;
}

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
