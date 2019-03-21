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

/// @file Dense.h
///
/// @brief This file defines a simple dense grid and efficient
/// converters to and from VDB grids.

#ifndef OPENVDB_TOOLS_DENSE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_DENSE_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/Exceptions.h>
#include <openvdb/util/Formats.h>
#include "Prune.h"
#include <tbb/parallel_for.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility> // for std::pair
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Populate a dense grid with the values of voxels from a sparse grid,
/// where the sparse grid intersects the dense grid.
/// @param sparse  an OpenVDB grid or tree from which to copy values
/// @param dense   the dense grid into which to copy values
/// @param serial  if false, process voxels in parallel
template<typename DenseT, typename GridOrTreeT>
void
copyToDense(
    const GridOrTreeT& sparse,
    DenseT& dense,
    bool serial = false);


/// @brief Populate a sparse grid with the values of all of the voxels of a dense grid.
/// @param dense      the dense grid from which to copy values
/// @param sparse     an OpenVDB grid or tree into which to copy values
/// @param tolerance  values in the dense grid that are within this tolerance of the sparse
///     grid's background value become inactive background voxels or tiles in the sparse grid
/// @param serial     if false, process voxels in parallel
template<typename DenseT, typename GridOrTreeT>
void
copyFromDense(
    const DenseT& dense,
    GridOrTreeT& sparse,
    const typename GridOrTreeT::ValueType& tolerance,
    bool serial = false);


////////////////////////////////////////

/// We currently support the following two 3D memory layouts for dense
/// volumes: XYZ, i.e. x is the fastest moving index, and ZYX, i.e. z
/// is the fastest moving index. The ZYX memory layout leads to nested
/// for-loops of the order x, y, z, which we find to be the most
/// intuitive. Hence, ZYX is the layout used throughout VDB. However,
/// other data structures, e.g. Houdini and Maya, employ the XYZ
/// layout. Clearly a dense volume with the ZYX layout converts more
/// efficiently to a VDB, but we support both for convenience.
enum MemoryLayout { LayoutXYZ, LayoutZYX };

/// @brief Base class for Dense which is defined below.
/// @note The constructor of this class is protected to prevent direct
/// instantiation.
template<typename ValueT, MemoryLayout Layout> class DenseBase;

/// @brief Partial template specialization of DenseBase.
/// @note ZYX is the memory-layout in VDB. It leads to nested
/// for-loops of the order x, y, z which we find to be the most intuitive.
template<typename ValueT>
class DenseBase<ValueT, LayoutZYX>
{
public:
    /// @brief Return the linear offset into this grid's value array given by
    /// unsigned coordinates (i, j, k), i.e., coordinates relative to
    /// the origin of this grid's bounding box.
    ///
    /// @warning The input coordinates are assume to be relative to
    /// the grid's origin, i.e. minimum of its index bounding box!
    inline size_t coordToOffset(size_t i, size_t j, size_t k) const { return i*mX + j*mY + k; }

    /// @brief Return the local coordinate corresponding to the specified linear offset.
    ///
    /// @warning The returned coordinate is relative to the origin of this
    /// grid's bounding box so add dense.origin() to get absolute coordinates.
    inline Coord offsetToLocalCoord(size_t n) const
    {
      const size_t x = n / mX;
      n -= mX*x;
      const size_t y = n / mY;
      return Coord(Coord::ValueType(x), Coord::ValueType(y), Coord::ValueType(n - mY*y));
    }

    /// @brief Return the stride of the array in the x direction ( = dimY*dimZ).
    /// @note This method is required by both CopyToDense and CopyFromDense.
    inline size_t xStride() const { return mX; }

    /// @brief Return the stride of the array in the y direction ( = dimZ).
    /// @note This method is required by both CopyToDense and CopyFromDense.
    inline size_t yStride() const { return mY; }

    /// @brief Return the stride of the array in the z direction ( = 1).
    /// @note This method is required by both CopyToDense and CopyFromDense.
    static size_t zStride() { return 1; }

protected:
    /// Protected constructor so as to prevent direct instantiation
    DenseBase(const CoordBBox& bbox) : mBBox(bbox), mY(bbox.dim()[2]), mX(mY*bbox.dim()[1]) {}

    const CoordBBox mBBox;//signed coordinates of the domain represented by the grid
    const size_t mY, mX;//strides in the y and x direction
};// end of DenseBase<ValueT, LayoutZYX>

/// @brief Partial template specialization of DenseBase.
/// @note This is the memory-layout employed in Houdini and Maya. It leads
/// to nested for-loops of the order z, y, x.
template<typename ValueT>
class DenseBase<ValueT, LayoutXYZ>
{
public:
    /// @brief Return the linear offset into this grid's value array given by
    /// unsigned coordinates (i, j, k), i.e., coordinates relative to
    /// the origin of this grid's bounding box.
    ///
    /// @warning The input coordinates are assume to be relative to
    /// the grid's origin, i.e. minimum of its index bounding box!
    inline size_t coordToOffset(size_t i, size_t j, size_t k) const { return i + j*mY + k*mZ; }

    /// @brief Return the index coordinate corresponding to the specified linear offset.
    ///
    /// @warning The returned coordinate is relative to the origin of this
    /// grid's bounding box so add dense.origin() to get absolute coordinates.
    inline Coord offsetToLocalCoord(size_t n) const
    {
        const size_t z = n / mZ;
        n -= mZ*z;
        const size_t y = n / mY;
        return Coord(Coord::ValueType(n - mY*y), Coord::ValueType(y), Coord::ValueType(z));
    }

    /// @brief Return the stride of the array in the x direction ( = 1).
    /// @note This method is required by both CopyToDense and CopyFromDense.
    static size_t xStride() { return 1; }

    /// @brief Return the stride of the array in the y direction ( = dimX).
    /// @note This method is required by both CopyToDense and CopyFromDense.
    inline size_t yStride() const { return mY; }

    /// @brief Return the stride of the array in the y direction ( = dimX*dimY).
    /// @note This method is required by both CopyToDense and CopyFromDense.
    inline size_t zStride() const { return mZ; }

protected:
    /// Protected constructor so as to prevent direct instantiation
    DenseBase(const CoordBBox& bbox) : mBBox(bbox), mY(bbox.dim()[0]), mZ(mY*bbox.dim()[1]) {}

    const CoordBBox mBBox;//signed coordinates of the domain represented by the grid
    const size_t mY, mZ;//strides in the y and z direction
};// end of DenseBase<ValueT, LayoutXYZ>

/// @brief Dense is a simple dense grid API used by the CopyToDense and
/// CopyFromDense classes defined below.
/// @details Use the Dense class to efficiently produce a dense in-memory
/// representation of an OpenVDB grid.  However, be aware that a dense grid
/// could have a memory footprint that is orders of magnitude larger than
/// the sparse grid from which it originates.
///
/// @note This class can be used as a simple wrapper for existing dense grid
/// classes if they provide access to the raw data array.
/// @note This implementation allows for the 3D memory layout to be
/// defined by the MemoryLayout template parameter (see above for definition).
/// The default memory layout is ZYX since that's the layout used by OpenVDB grids.
template<typename ValueT, MemoryLayout Layout = LayoutZYX>
class Dense : public DenseBase<ValueT, Layout>
{
public:
    using ValueType = ValueT;
    using BaseT = DenseBase<ValueT, Layout>;
    using Ptr = SharedPtr<Dense>;
    using ConstPtr = SharedPtr<const Dense>;

    /// @brief Construct a dense grid with a given range of coordinates.
    ///
    /// @param bbox  the bounding box of the (signed) coordinate range of this grid
    /// @throw ValueError if the bounding box is empty.
    /// @note The min and max coordinates of the bounding box are inclusive.
    Dense(const CoordBBox& bbox) : BaseT(bbox) { this->init(); }

    /// @brief Construct a dense grid with a given range of coordinates and initial value
    ///
    /// @param bbox  the bounding box of the (signed) coordinate range of this grid
    /// @param value the initial value of the grid.
    /// @throw ValueError if the bounding box is empty.
    /// @note The min and max coordinates of the bounding box are inclusive.
    Dense(const CoordBBox& bbox, const ValueT& value) : BaseT(bbox)
    {
        this->init();
        this->fill(value);
    }

    /// @brief Construct a dense grid that wraps an external array.
    ///
    /// @param bbox  the bounding box of the (signed) coordinate range of this grid
    /// @param data  a raw C-style array whose size is commensurate with
    ///     the coordinate domain of @a bbox
    ///
    /// @note The data array is assumed to have a stride of one in the @e z direction.
    /// @throw ValueError if the bounding box is empty.
    /// @note The min and max coordinates of the bounding box are inclusive.
    Dense(const CoordBBox& bbox, ValueT* data) : BaseT(bbox), mData(data)
    {
        if (BaseT::mBBox.empty()) {
            OPENVDB_THROW(ValueError, "can't construct a dense grid with an empty bounding box");
        }
    }

    /// @brief Construct a dense grid with a given origin and dimensions.
    ///
    /// @param dim  the desired dimensions of the grid
    /// @param min  the signed coordinates of the first voxel in the dense grid
    /// @throw ValueError if any of the dimensions are zero.
    /// @note The @a min coordinate is inclusive, and the max coordinate will be
    /// @a min + @a dim - 1.
    Dense(const Coord& dim, const Coord& min = Coord(0))
        : BaseT(CoordBBox(min, min+dim.offsetBy(-1)))
    {
        this->init();
    }

    /// @brief Return the memory layout for this grid (see above for definitions).
    static MemoryLayout memoryLayout() { return Layout; }

    /// @brief Return a raw pointer to this grid's value array.
    /// @note This method is required by CopyToDense.
    inline ValueT* data() { return mData; }

    /// @brief Return a raw pointer to this grid's value array.
    /// @note This method is required by CopyFromDense.
    inline const ValueT* data() const { return mData; }

    /// @brief Return the bounding box of the signed index domain of this grid.
    /// @note This method is required by both CopyToDense and CopyFromDense.
    inline const CoordBBox& bbox() const { return BaseT::mBBox; }

     /// Return the grid's origin in index coordinates.
    inline const Coord& origin() const { return BaseT::mBBox.min(); }

    /// @brief Return the number of voxels contained in this grid.
    inline Index64 valueCount() const { return BaseT::mBBox.volume(); }

    /// @brief Set the value of the voxel at the given array offset.
    inline void setValue(size_t offset, const ValueT& value) { mData[offset] = value; }

    /// @brief Return a const reference to the value of the voxel at the given array offset.
    const ValueT& getValue(size_t offset) const { return mData[offset]; }

    /// @brief Return a non-const reference to the value of the voxel at the given array offset.
    ValueT& getValue(size_t offset) { return mData[offset]; }

    /// @brief Set the value of the voxel at unsigned index coordinates (i, j, k).
    /// @note This is somewhat slower than using an array offset.
    inline void setValue(size_t i, size_t j, size_t k, const ValueT& value)
    {
        mData[BaseT::coordToOffset(i,j,k)] = value;
    }

    /// @brief Return a const reference to the value of the voxel
    /// at unsigned index coordinates (i, j, k).
    /// @note This is somewhat slower than using an array offset.
    inline const ValueT& getValue(size_t i, size_t j, size_t k) const
    {
        return mData[BaseT::coordToOffset(i,j,k)];
    }

    /// @brief Return a non-const reference to the value of the voxel
    /// at unsigned index coordinates (i, j, k).
    /// @note This is somewhat slower than using an array offset.
    inline ValueT& getValue(size_t i, size_t j, size_t k)
    {
        return mData[BaseT::coordToOffset(i,j,k)];
    }

    /// @brief Set the value of the voxel at the given signed coordinates.
    /// @note This is slower than using either an array offset or unsigned index coordinates.
    inline void setValue(const Coord& xyz, const ValueT& value)
    {
        mData[this->coordToOffset(xyz)] = value;
    }

    /// @brief Return a const reference to the value of the voxel at the given signed coordinates.
    /// @note This is slower than using either an array offset or unsigned index coordinates.
    inline const ValueT& getValue(const Coord& xyz) const
    {
        return mData[this->coordToOffset(xyz)];
    }

    /// @brief Return a non-const reference to the value of the voxel
    /// at the given signed coordinates.
    /// @note This is slower than using either an array offset or unsigned index coordinates.
    inline ValueT& getValue(const Coord& xyz)
    {
        return mData[this->coordToOffset(xyz)];
    }

    /// @brief Fill this grid with a constant value.
    inline void fill(const ValueT& value)
    {
        size_t size = this->valueCount();
        ValueT* a = mData;
        while(size--) *a++ = value;
    }

    /// @brief Return the linear offset into this grid's value array given by
    /// the specified signed coordinates, i.e., coordinates in the space of
    /// this grid's bounding box.
    ///
    /// @note This method reflects the fact that we assume the same
    /// layout of values as an OpenVDB grid, i.e., the fastest coordinate is @e z.
    inline size_t coordToOffset(const Coord& xyz) const
    {
        assert(BaseT::mBBox.isInside(xyz));
        return BaseT::coordToOffset(size_t(xyz[0]-BaseT::mBBox.min()[0]),
                                    size_t(xyz[1]-BaseT::mBBox.min()[1]),
                                    size_t(xyz[2]-BaseT::mBBox.min()[2]));
    }

    /// @brief Return the global coordinate corresponding to the specified linear offset.
    inline Coord offsetToCoord(size_t n) const
    {
      return this->offsetToLocalCoord(n) + BaseT::mBBox.min();
    }

    /// @brief Return the memory footprint of this Dense grid in bytes.
    inline Index64 memUsage() const
    {
        return sizeof(*this) + BaseT::mBBox.volume() * sizeof(ValueType);
    }

    /// @brief Output a human-readable description of this grid to the
    /// specified stream.
    void print(const std::string& name = "", std::ostream& os = std::cout) const
    {
        const Coord dim = BaseT::mBBox.dim();
        os << "Dense Grid";
        if (!name.empty()) os << " \"" << name << "\"";
        util::printBytes(os, this->memUsage(), ":\n  Memory footprint:     ");
        os << "  Dimensions of grid  :   " << dim[0] << " x " << dim[1] << " x " << dim[2] << "\n";
        os << "  Number of voxels:       " << util::formattedInt(this->valueCount()) << "\n";
        os << "  Bounding box of voxels: " << BaseT::mBBox << "\n";
        os << "  Memory layout:          " << (Layout == LayoutZYX ? "ZYX (" : "XYZ (dis")
           << "similar to VDB)\n";
    }

private:
    /// @brief Private method to initialize the dense value array.
    void init()
    {
        if (BaseT::mBBox.empty()) {
            OPENVDB_THROW(ValueError, "can't construct a dense grid with an empty bounding box");
        }
        mArray.reset(new ValueT[BaseT::mBBox.volume()]);
        mData = mArray.get();
    }

    std::unique_ptr<ValueT[]> mArray;
    ValueT* mData;//raw c-style pointer to values
};// end of Dense

////////////////////////////////////////


/// @brief Copy an OpenVDB tree into an existing dense grid.
///
/// @note Only voxels that intersect the dense grid's bounding box are copied
/// from the OpenVDB tree.  But both active and inactive voxels are copied,
/// so all existing values in the dense grid are overwritten, regardless of
/// the OpenVDB tree's topology.
template<typename _TreeT, typename _DenseT = Dense<typename _TreeT::ValueType> >
class CopyToDense
{
public:
    using DenseT = _DenseT;
    using TreeT = _TreeT;
    using ValueT = typename TreeT::ValueType;

    CopyToDense(const TreeT& tree, DenseT& dense)
        : mRoot(&(tree.root())), mDense(&dense) {}

    void copy(bool serial = false) const
    {
        if (serial) {
            mRoot->copyToDense(mDense->bbox(), *mDense);
        } else {
            tbb::parallel_for(mDense->bbox(), *this);
        }
    }

    /// @brief Public method called by tbb::parallel_for
    void operator()(const CoordBBox& bbox) const
    {
        mRoot->copyToDense(bbox, *mDense);
    }

private:
    const typename TreeT::RootNodeType* mRoot;
    DenseT* mDense;
};// CopyToDense


// Convenient wrapper function for the CopyToDense class
template<typename DenseT, typename GridOrTreeT>
void
copyToDense(const GridOrTreeT& sparse, DenseT& dense, bool serial)
{
    using Adapter = TreeAdapter<GridOrTreeT>;
    using TreeT = typename Adapter::TreeType;

    CopyToDense<TreeT, DenseT> op(Adapter::constTree(sparse), dense);
    op.copy(serial);
}


////////////////////////////////////////


/// @brief Copy the values from a dense grid into an OpenVDB tree.
///
/// @details Values in the dense grid that are within a tolerance of
/// the background value are truncated to inactive background voxels or tiles.
/// This allows the tree to form a sparse representation of the dense grid.
///
/// @note Since this class allocates leaf nodes concurrently it is recommended
/// to use a scalable implementation of @c new like the one provided by TBB,
/// rather than the mutex-protected standard library @c new.
template<typename _TreeT, typename _DenseT = Dense<typename _TreeT::ValueType> >
class CopyFromDense
{
public:
    using DenseT = _DenseT;
    using TreeT = _TreeT;
    using ValueT = typename TreeT::ValueType;
    using LeafT = typename TreeT::LeafNodeType;
    using AccessorT = tree::ValueAccessor<TreeT>;

    CopyFromDense(const DenseT& dense, TreeT& tree, const ValueT& tolerance)
        : mDense(&dense),
          mTree(&tree),
          mBlocks(nullptr),
          mTolerance(tolerance),
          mAccessor(tree.empty() ? nullptr : new AccessorT(tree))
    {
    }
    CopyFromDense(const CopyFromDense& other)
        : mDense(other.mDense),
          mTree(other.mTree),
          mBlocks(other.mBlocks),
          mTolerance(other.mTolerance),
          mAccessor(other.mAccessor.get() == nullptr ? nullptr : new AccessorT(*mTree))
    {
    }

    /// @brief Copy values from the dense grid to the sparse tree.
    void copy(bool serial = false)
    {
        mBlocks = new std::vector<Block>();
        const CoordBBox& bbox = mDense->bbox();
        // Pre-process: Construct a list of blocks aligned with (potential) leaf nodes
        for (CoordBBox sub=bbox; sub.min()[0] <= bbox.max()[0]; sub.min()[0] = sub.max()[0] + 1) {
            for (sub.min()[1] = bbox.min()[1]; sub.min()[1] <= bbox.max()[1];
                 sub.min()[1] = sub.max()[1] + 1)
            {
                for (sub.min()[2] = bbox.min()[2]; sub.min()[2] <= bbox.max()[2];
                     sub.min()[2] = sub.max()[2] + 1)
                {
                    sub.max() = Coord::minComponent(bbox.max(),
                        (sub.min()&(~(LeafT::DIM-1u))).offsetBy(LeafT::DIM-1u));
                    mBlocks->push_back(Block(sub));
                }
            }
        }

        // Multi-threaded process: Convert dense grid into leaf nodes and tiles
        if (serial) {
            (*this)(tbb::blocked_range<size_t>(0, mBlocks->size()));
        } else {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mBlocks->size()), *this);
        }

        // Post-process: Insert leaf nodes and tiles into the tree, and prune the tiles only!
        tree::ValueAccessor<TreeT> acc(*mTree);
        for (size_t m=0, size = mBlocks->size(); m<size; ++m) {
            Block& block = (*mBlocks)[m];
            if (block.leaf) {
                acc.addLeaf(block.leaf);
            } else if (block.tile.second) {//only background tiles are inactive
                acc.addTile(1, block.bbox.min(), block.tile.first, true);//leaf tile
            }
        }
        delete mBlocks;
        mBlocks = nullptr;

        tools::pruneTiles(*mTree, mTolerance);//multi-threaded
    }

    /// @brief Public method called by tbb::parallel_for
    /// @warning Never call this method directly!
    void operator()(const tbb::blocked_range<size_t> &r) const
    {
        assert(mBlocks);
        LeafT* leaf = new LeafT();

        for (size_t m=r.begin(), n=0, end = r.end(); m != end; ++m, ++n) {

            Block& block = (*mBlocks)[m];
            const CoordBBox &bbox = block.bbox;

            if (mAccessor.get() == nullptr) {//i.e. empty target tree
                leaf->fill(mTree->background(), false);
            } else {//account for existing leaf nodes in the target tree
                if (const LeafT* target = mAccessor->probeConstLeaf(bbox.min())) {
                    (*leaf) = (*target);
                } else {
                    ValueT value = zeroVal<ValueT>();
                    bool state = mAccessor->probeValue(bbox.min(), value);
                    leaf->fill(value, state);
                }
            }

            leaf->copyFromDense(bbox, *mDense, mTree->background(), mTolerance);

            if (!leaf->isConstant(block.tile.first, block.tile.second, mTolerance)) {
                leaf->setOrigin(bbox.min() & (~(LeafT::DIM - 1)));
                block.leaf = leaf;
                leaf = new LeafT();
            }
        }// loop over blocks

        delete leaf;
    }

private:
    struct Block {
        CoordBBox               bbox;
        LeafT*                  leaf;
        std::pair<ValueT, bool> tile;
        Block(const CoordBBox& b) : bbox(b), leaf(nullptr) {}
    };

    const DenseT*              mDense;
    TreeT*                     mTree;
    std::vector<Block>*        mBlocks;
    ValueT                     mTolerance;
    std::unique_ptr<AccessorT> mAccessor;
};// CopyFromDense


// Convenient wrapper function for the CopyFromDense class
template<typename DenseT, typename GridOrTreeT>
void
copyFromDense(const DenseT& dense, GridOrTreeT& sparse,
    const typename GridOrTreeT::ValueType& tolerance, bool serial)
{
    using Adapter = TreeAdapter<GridOrTreeT>;
    using TreeT = typename Adapter::TreeType;

    CopyFromDense<TreeT, DenseT> op(dense, Adapter::tree(sparse), tolerance);
    op.copy(serial);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_DENSE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
