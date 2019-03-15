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

/// @author Dan Bailey
///
/// @file points/PointDataGrid.h
///
/// @brief  Attribute-owned data structure for points. Point attributes are
///         stored in leaf nodes and ordered by voxel for fast random and
///         sequential access.

#ifndef OPENVDB_POINTS_POINT_DATA_GRID_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_DATA_GRID_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tools/PointIndexGrid.h>
#include "AttributeArray.h"
#include "AttributeArrayString.h"
#include "AttributeGroup.h"
#include "AttributeSet.h"
#include "StreamCompression.h"
#include <cstring> // std::memcpy
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits> // std::is_same
#include <utility> // std::pair, std::make_pair
#include <vector>

#include <boost/mpl/vector.hpp>//for boost::mpl::vector
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/back.hpp>

class TestPointDataLeaf;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace io
{

/// @brief openvdb::io::readCompressedValues specialized on PointDataIndex32 arrays to
/// ignore the value mask, use a larger block size and use 16-bit size instead of 64-bit
template<>
inline void
readCompressedValues(   std::istream& is, PointDataIndex32* destBuf, Index destCount,
                        const util::NodeMask<3>& /*valueMask*/, bool /*fromHalf*/)
{
    using compression::bloscDecompress;

    const bool seek = destBuf == nullptr;

    const size_t destBytes = destCount*sizeof(PointDataIndex32);
    const size_t maximumBytes = std::numeric_limits<uint16_t>::max();
    if (destBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot read more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    uint16_t bytes16;

    const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(is);

    if (seek && meta) {
        // buffer size temporarily stored in the StreamMetadata pass
        // to avoid having to perform an expensive disk read for 2-bytes
        bytes16 = static_cast<uint16_t>(meta->pass());
        // seek over size of the compressed buffer
        is.seekg(sizeof(uint16_t), std::ios_base::cur);
    }
    else {
        // otherwise read from disk
        is.read(reinterpret_cast<char*>(&bytes16), sizeof(uint16_t));
    }

    if (bytes16 == std::numeric_limits<uint16_t>::max()) {
        // read or seek uncompressed data
        if (seek) {
            is.seekg(destBytes, std::ios_base::cur);
        }
        else {
            is.read(reinterpret_cast<char*>(destBuf), destBytes);
        }
    }
    else {
        // read or seek uncompressed data
        if (seek) {
            is.seekg(int(bytes16), std::ios_base::cur);
        }
        else {
            // decompress into the destination buffer
            std::unique_ptr<char[]> bloscBuffer(new char[int(bytes16)]);
            is.read(bloscBuffer.get(), bytes16);
            std::unique_ptr<char[]> buffer = bloscDecompress(   bloscBuffer.get(),
                                                                destBytes,
                                                                /*resize=*/false);
            std::memcpy(destBuf, buffer.get(), destBytes);
        }
    }
}

/// @brief openvdb::io::writeCompressedValues specialized on PointDataIndex32 arrays to
/// ignore the value mask, use a larger block size and use 16-bit size instead of 64-bit
template<>
inline void
writeCompressedValues(  std::ostream& os, PointDataIndex32* srcBuf, Index srcCount,
                        const util::NodeMask<3>& /*valueMask*/,
                        const util::NodeMask<3>& /*childMask*/, bool /*toHalf*/)
{
    using compression::bloscCompress;

    const size_t srcBytes = srcCount*sizeof(PointDataIndex32);
    const size_t maximumBytes = std::numeric_limits<uint16_t>::max();
    if (srcBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot write more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    const char* charBuffer = reinterpret_cast<const char*>(srcBuf);

    size_t compressedBytes;
    std::unique_ptr<char[]> buffer = bloscCompress( charBuffer, srcBytes,
                                                    compressedBytes, /*resize=*/false);

    if (compressedBytes > 0) {
        auto bytes16 = static_cast<uint16_t>(compressedBytes); // clamp to 16-bit unsigned integer
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
        os.write(reinterpret_cast<const char*>(buffer.get()), compressedBytes);
    }
    else {
        auto bytes16 = static_cast<uint16_t>(maximumBytes); // max value indicates uncompressed
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
        os.write(reinterpret_cast<const char*>(srcBuf), srcBytes);
    }
}

template <typename T>
inline void
writeCompressedValuesSize(std::ostream& os, const T* srcBuf, Index srcCount)
{
    using compression::bloscCompressedSize;

    const size_t srcBytes = srcCount*sizeof(T);
    const size_t maximumBytes = std::numeric_limits<uint16_t>::max();
    if (srcBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot write more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    const char* charBuffer = reinterpret_cast<const char*>(srcBuf);

    // calculate voxel buffer size after compression
    size_t compressedBytes = bloscCompressedSize(charBuffer, srcBytes);

    if (compressedBytes > 0) {
        auto bytes16 = static_cast<uint16_t>(compressedBytes); // clamp to 16-bit unsigned integer
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
    }
    else {
        auto bytes16 = static_cast<uint16_t>(maximumBytes); // max value indicates uncompressed
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
    }
}

} // namespace io


// forward declaration
namespace tree {
    template<Index, typename> struct SameLeafConfig;
}


////////////////////////////////////////


namespace points {


// forward declaration
template<typename T, Index Log2Dim> class PointDataLeafNode;

/// @brief Point index tree configured to match the default VDB configurations.
using PointDataTree = tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode
    <PointDataLeafNode<PointDataIndex32, 3>, 4>, 5>>>;


/// @brief Point data grid.
using PointDataGrid = Grid<PointDataTree>;


/// @brief  Deep copy the descriptor across all leaf nodes.
///
/// @param  tree the PointDataTree.
///
/// @return the new descriptor.
///
/// @note This method will fail if the Descriptors in the tree are not all identical.
template <typename PointDataTreeT>
inline AttributeSet::Descriptor::Ptr
makeDescriptorUnique(PointDataTreeT& tree);


/// @brief  Toggle the streaming mode on all attributes in the tree to collapse the attributes
///         after deconstructing a bound AttributeHandle to each array. This results in better
///         memory efficiency when the data is streamed into another data structure
///         (typically for rendering).
///
/// @param  tree the PointDataTree.
/// @param  on @c true to enable streaming
///
/// @note   Multiple threads cannot safely access the same AttributeArray when using streaming.
template <typename PointDataTreeT>
inline void
setStreamingMode(PointDataTreeT& tree, bool on = true);


/// @brief  Sequentially pre-fetch all delayed-load voxel and attribute data from disk in order
///         to accelerate subsequent random access.
///
/// @param  tree                the PointDataTree.
/// @param  position            if enabled, prefetch the position attribute (default is on)
/// @param  otherAttributes     if enabled, prefetch all other attributes (default is on)
template <typename PointDataTreeT>
inline void
prefetch(PointDataTreeT& tree, bool position = true, bool otherAttributes = true);


////////////////////////////////////////


template <typename T, Index Log2Dim>
class PointDataLeafNode : public tree::LeafNode<T, Log2Dim>, io::MultiPass {

public:
    using LeafNodeType  = PointDataLeafNode<T, Log2Dim>;
    using Ptr           = std::shared_ptr<PointDataLeafNode>;

    using ValueType     = T;
    using ValueTypePair = std::pair<ValueType, ValueType>;
    using IndexArray    = std::vector<ValueType>;

    using Descriptor    = AttributeSet::Descriptor;

    ////////////////////////////////////////

    // The following methods had to be copied from the LeafNode class
    // to make the derived PointDataLeafNode class compatible with the tree structure.

    using BaseLeaf      = tree::LeafNode<T, Log2Dim>;
    using NodeMaskType  = util::NodeMask<Log2Dim>;

    using BaseLeaf::LOG2DIM;
    using BaseLeaf::TOTAL;
    using BaseLeaf::DIM;
    using BaseLeaf::NUM_VALUES;
    using BaseLeaf::NUM_VOXELS;
    using BaseLeaf::SIZE;
    using BaseLeaf::LEVEL;

    /// Default constructor
    PointDataLeafNode()
        : mAttributeSet(new AttributeSet) { }

    ~PointDataLeafNode() = default;

    /// Construct using deep copy of other PointDataLeafNode
    explicit PointDataLeafNode(const PointDataLeafNode& other)
        : BaseLeaf(other)
        , mAttributeSet(new AttributeSet(*other.mAttributeSet)) { }

    /// Construct using supplied origin, value and active status
    explicit
    PointDataLeafNode(const Coord& coords, const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(coords, zeroVal<T>(), active)
        , mAttributeSet(new AttributeSet) { assertNonModifiableUnlessZero(value); }

    /// Construct using supplied origin, value and active status
    /// use attribute map from another PointDataLeafNode
    PointDataLeafNode(const PointDataLeafNode& other, const Coord& coords,
        const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(coords, zeroVal<T>(), active)
        , mAttributeSet(new AttributeSet(*other.mAttributeSet))
    {
        assertNonModifiableUnlessZero(value);
    }

    // Copy-construct from a PointIndexLeafNode with the same configuration but a different ValueType.
    template<typename OtherValueType>
    PointDataLeafNode(const tools::PointIndexLeafNode<OtherValueType, Log2Dim>& other)
        : BaseLeaf(other)
        , mAttributeSet(new AttributeSet) { }

    // Copy-construct from a LeafNode with the same configuration but a different ValueType.
    // Used for topology copies - explicitly sets the value (background) to zeroVal
    template <typename ValueType>
    PointDataLeafNode(const tree::LeafNode<ValueType, Log2Dim>& other, const T& value, TopologyCopy)
        : BaseLeaf(other, zeroVal<T>(), TopologyCopy())
        , mAttributeSet(new AttributeSet) { assertNonModifiableUnlessZero(value); }

    // Copy-construct from a LeafNode with the same configuration but a different ValueType.
    // Used for topology copies - explicitly sets the on and off value (background) to zeroVal
    template <typename ValueType>
    PointDataLeafNode(const tree::LeafNode<ValueType, Log2Dim>& other, const T& /*offValue*/, const T& /*onValue*/, TopologyCopy)
        : BaseLeaf(other, zeroVal<T>(), zeroVal<T>(), TopologyCopy())
        , mAttributeSet(new AttributeSet) { }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    PointDataLeafNode(PartialCreate, const Coord& coords,
        const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(PartialCreate(), coords, value, active)
        , mAttributeSet(new AttributeSet) { assertNonModifiableUnlessZero(value); }
#endif

public:

    /// Retrieve the attribute set.
    const AttributeSet& attributeSet() const { return *mAttributeSet; }

    /// @brief Create a new attribute set. Existing attributes will be removed.
    void initializeAttributes(const Descriptor::Ptr& descriptor, const Index arrayLength);
    /// @brief Clear the attribute set.
    void clearAttributes(const bool updateValueMask = true);

    /// @brief Returns @c true if an attribute with this index exists.
    /// @param pos Index of the attribute
    bool hasAttribute(const size_t pos) const;
    /// @brief Returns @c true if an attribute with this name exists.
    /// @param attributeName    Name of the attribute
    bool hasAttribute(const Name& attributeName) const;

    /// @brief Append an attribute to the leaf.
    /// @param expected Existing descriptor is expected to match this parameter.
    /// @param replacement New descriptor to replace the existing one.
    /// @param pos Index of the new attribute in the descriptor replacement.
    /// @param strideOrTotalSize Stride of the attribute array (if constantStride), total size otherwise
    /// @param constantStride if @c false, stride is interpreted as total size of the array
    AttributeArray::Ptr appendAttribute(const Descriptor& expected, Descriptor::Ptr& replacement,
                                        const size_t pos, const Index strideOrTotalSize = 1,
                                        const bool constantStride = true);

    /// @brief Drop list of attributes.
    /// @param pos vector of attribute indices to drop
    /// @param expected Existing descriptor is expected to match this parameter.
    /// @param replacement New descriptor to replace the existing one.
    void dropAttributes(const std::vector<size_t>& pos,
                        const Descriptor& expected, Descriptor::Ptr& replacement);
    /// @brief Reorder attribute set.
    /// @param replacement New descriptor to replace the existing one.
    void reorderAttributes(const Descriptor::Ptr& replacement);
    /// @brief Rename attributes in attribute set (order must remain the same).
    /// @param expected Existing descriptor is expected to match this parameter.
    /// @param replacement New descriptor to replace the existing one.
    void renameAttributes(const Descriptor& expected, Descriptor::Ptr& replacement);
    /// @brief Compact all attributes in attribute set.
    void compactAttributes();

    /// @brief Replace the underlying attribute set with the given @a attributeSet.
    /// @details This leaf will assume ownership of the given attribute set. The descriptors must
    /// match and the voxel offsets values will need updating if the point order is different.
    /// @throws ValueError if @a allowMismatchingDescriptors is @c false and the descriptors
    /// do not match
    void replaceAttributeSet(AttributeSet* attributeSet, bool allowMismatchingDescriptors = false);

    /// @brief Replace the descriptor with a new one
    /// The new Descriptor must exactly match the old one
    void resetDescriptor(const Descriptor::Ptr& replacement);

    /// @brief Sets all of the voxel offset values on this leaf, from the given vector
    /// of @a offsets. If @a updateValueMask is true, then the active value mask will
    /// be updated so voxels with points are active and empty voxels are inactive.
    void setOffsets(const std::vector<ValueType>& offsets, const bool updateValueMask = true);

    /// @brief Throws an error if the voxel values on this leaf are not monotonically
    /// increasing or within the bounds of the attribute arrays
    void validateOffsets() const;

    /// @brief Read-write attribute array reference from index
    /// {
    AttributeArray& attributeArray(const size_t pos);
    const AttributeArray& attributeArray(const size_t pos) const;
    const AttributeArray& constAttributeArray(const size_t pos) const;
    /// }
    /// @brief Read-write attribute array reference from name
    /// {
    AttributeArray& attributeArray(const Name& attributeName);
    const AttributeArray& attributeArray(const Name& attributeName) const;
    const AttributeArray& constAttributeArray(const Name& attributeName) const;
    /// }

    /// @brief Read-only group handle from group index
    GroupHandle groupHandle(const AttributeSet::Descriptor::GroupIndex& index) const;
    /// @brief Read-only group handle from group name
    GroupHandle groupHandle(const Name& group) const;
    /// @brief Read-write group handle from group index
    GroupWriteHandle groupWriteHandle(const AttributeSet::Descriptor::GroupIndex& index);
    /// @brief Read-write group handle from group name
    GroupWriteHandle groupWriteHandle(const Name& name);

    /// @brief Compute the total point count for the leaf
    Index64 pointCount() const;
    /// @brief Compute the total active (on) point count for the leaf
    Index64 onPointCount() const;
    /// @brief Compute the total inactive (off) point count for the leaf
    Index64 offPointCount() const;
    /// @brief Compute the point count in a specific group for the leaf
    Index64 groupPointCount(const Name& groupName) const;

    /// @brief Activate voxels with non-zero points, deactivate voxels with zero points.
    void updateValueMask();

    ////////////////////////////////////////

    void setOffsetOn(Index offset, const ValueType& val);
    void setOffsetOnly(Index offset, const ValueType& val);

    /// @brief Return @c true if the given node (which may have a different @c ValueType
    /// than this node) has the same active value topology as this node.
    template<typename OtherType, Index OtherLog2Dim>
    bool hasSameTopology(const PointDataLeafNode<OtherType, OtherLog2Dim>* other) const {
        return BaseLeaf::hasSameTopology(other);
    }

    /// Check for buffer, state and origin equivalence first.
    /// If this returns true, do a deeper comparison on the attribute set to check
    bool operator==(const PointDataLeafNode& other) const {
        if(BaseLeaf::operator==(other) != true) return false;
        return (*this->mAttributeSet == *other.mAttributeSet);
    }

    bool operator!=(const PointDataLeafNode& other) const { return !(other == *this); }

    void addLeaf(PointDataLeafNode*) {}
    template<typename AccessorT>
    void addLeafAndCache(PointDataLeafNode*, AccessorT&) {}

    //@{
    /// @brief Return a pointer to this node.
    PointDataLeafNode* touchLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointDataLeafNode* touchLeafAndCache(const Coord&, AccessorT&) { return this; }

    template<typename NodeT, typename AccessorT>
    NodeT* probeNodeAndCache(const Coord&, AccessorT&)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(std::is_same<NodeT,PointDataLeafNode>::value)) return nullptr;
        return reinterpret_cast<NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    PointDataLeafNode* probeLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointDataLeafNode* probeLeafAndCache(const Coord&, AccessorT&) { return this; }
    //@}

    //@{
    /// @brief Return a @const pointer to this node.
    const PointDataLeafNode* probeConstLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const PointDataLeafNode* probeConstLeafAndCache(const Coord&, AccessorT&) const { return this; }
    template<typename AccessorT>
    const PointDataLeafNode* probeLeafAndCache(const Coord&, AccessorT&) const { return this; }
    const PointDataLeafNode* probeLeaf(const Coord&) const { return this; }
    template<typename NodeT, typename AccessorT>
    const NodeT* probeConstNodeAndCache(const Coord&, AccessorT&) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(std::is_same<NodeT,PointDataLeafNode>::value)) return nullptr;
        return reinterpret_cast<const NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}

    // I/O methods

    void readTopology(std::istream& is, bool fromHalf = false);
    void writeTopology(std::ostream& os, bool toHalf = false) const;

    Index buffers() const;

    void readBuffers(std::istream& is, bool fromHalf = false);
    void readBuffers(std::istream& is, const CoordBBox&, bool fromHalf = false);
    void writeBuffers(std::ostream& os, bool toHalf = false) const;


    Index64 memUsage() const;

    void evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels = true) const;

    /// @brief Return the bounding box of this node, i.e., the full index space
    /// spanned by this leaf node.
    CoordBBox getNodeBoundingBox() const;

    ////////////////////////////////////////

    // Disable all write methods to avoid unintentional changes
    // to the point-array offsets.

    void assertNonmodifiable() {
        assert(false && "Cannot modify voxel values in a PointDataTree.");
    }

    // some methods silently ignore attempts to modify the
    // point-array offsets if a zero value is used

    void assertNonModifiableUnlessZero(const ValueType& value) {
        if (value != zeroVal<T>()) this->assertNonmodifiable();
    }

    void setActiveState(const Coord& xyz, bool on) { BaseLeaf::setActiveState(xyz, on); }
    void setActiveState(Index offset, bool on) { BaseLeaf::setActiveState(offset, on); }

    void setValueOnly(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOnly(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOff(const Coord& xyz) { BaseLeaf::setValueOff(xyz); }
    void setValueOff(Index offset) { BaseLeaf::setValueOff(offset); }

    void setValueOff(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOff(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOn(const Coord& xyz) { BaseLeaf::setValueOn(xyz); }
    void setValueOn(Index offset) {  BaseLeaf::setValueOn(offset); }

    void setValueOn(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOn(Index, const ValueType&) { assertNonmodifiable(); }

    void setValue(const Coord&, const ValueType&) { assertNonmodifiable(); }

    void setValuesOn() { BaseLeaf::setValuesOn(); }
    void setValuesOff() { BaseLeaf::setValuesOff(); }

    template<typename ModifyOp>
    void modifyValue(Index, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValue(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    // clipping is not yet supported
    void clip(const CoordBBox&, const ValueType& value) { assertNonModifiableUnlessZero(value); }

    void fill(const CoordBBox&, const ValueType&, bool);
    void fill(const ValueType& value) { assertNonModifiableUnlessZero(value); }
    void fill(const ValueType&, bool);

    template<typename AccessorT>
    void setValueOnlyAndCache(const Coord&, const ValueType&, AccessorT&) {assertNonmodifiable();}

    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndActiveStateAndCache(const Coord&, const ModifyOp&, AccessorT&) {
        assertNonmodifiable();
    }

    template<typename AccessorT>
    void setValueOffAndCache(const Coord&, const ValueType&, AccessorT&) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setActiveStateAndCache(const Coord& xyz, bool on, AccessorT& parent) {
        BaseLeaf::setActiveStateAndCache(xyz, on, parent);
    }

    void resetBackground(const ValueType&, const ValueType& newBackground) {
        assertNonModifiableUnlessZero(newBackground);
    }

    void signedFloodFill(const ValueType&) { assertNonmodifiable(); }
    void signedFloodFill(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void negate() { assertNonmodifiable(); }

    friend class ::TestPointDataLeaf;

    using ValueOn   = typename BaseLeaf::ValueOn;
    using ValueOff  = typename BaseLeaf::ValueOff;
    using ValueAll  = typename BaseLeaf::ValueAll;

private:
    std::unique_ptr<AttributeSet> mAttributeSet;
    uint16_t mVoxelBufferSize = 0;

protected:
    using ChildOn           = typename BaseLeaf::ChildOn;
    using ChildOff          = typename BaseLeaf::ChildOff;
    using ChildAll          = typename BaseLeaf::ChildAll;

    using MaskOnIterator    = typename NodeMaskType::OnIterator;
    using MaskOffIterator   = typename NodeMaskType::OffIterator;
    using MaskDenseIterator = typename NodeMaskType::DenseIterator;

    // During topology-only construction, access is needed
    // to protected/private members of other template instances.
    template<typename, Index> friend class PointDataLeafNode;

    friend class tree::IteratorBase<MaskOnIterator, PointDataLeafNode>;
    friend class tree::IteratorBase<MaskOffIterator, PointDataLeafNode>;
    friend class tree::IteratorBase<MaskDenseIterator, PointDataLeafNode>;

public:
    /// @brief Leaf value voxel iterator
    ValueVoxelCIter beginValueVoxel(const Coord& ijk) const;

public:

#if defined(_MSC_VER) && (_MSC_VER < 1914)
    using ValueOnIter = typename BaseLeaf::ValueIter<
        MaskOnIterator, PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOnCIter = typename BaseLeaf::ValueIter<
        MaskOnIterator, const PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOffIter = typename BaseLeaf::ValueIter<
        MaskOffIterator, PointDataLeafNode, const ValueType, ValueOff>;
    using ValueOffCIter = typename BaseLeaf::ValueIter<
        MaskOffIterator,const PointDataLeafNode,const ValueType,ValueOff>;
    using ValueAllIter = typename BaseLeaf::ValueIter<
        MaskDenseIterator, PointDataLeafNode, const ValueType, ValueAll>;
    using ValueAllCIter = typename BaseLeaf::ValueIter<
        MaskDenseIterator,const PointDataLeafNode,const ValueType,ValueAll>;
    using ChildOnIter = typename BaseLeaf::ChildIter<
        MaskOnIterator, PointDataLeafNode, ChildOn>;
    using ChildOnCIter = typename BaseLeaf::ChildIter<
        MaskOnIterator, const PointDataLeafNode, ChildOn>;
    using ChildOffIter = typename BaseLeaf::ChildIter<
        MaskOffIterator, PointDataLeafNode, ChildOff>;
    using ChildOffCIter = typename BaseLeaf::ChildIter<
        MaskOffIterator, const PointDataLeafNode, ChildOff>;
    using ChildAllIter = typename BaseLeaf::DenseIter<
        PointDataLeafNode, ValueType, ChildAll>;
    using ChildAllCIter = typename BaseLeaf::DenseIter<
        const PointDataLeafNode, const ValueType, ChildAll>;
#else
    using ValueOnIter = typename BaseLeaf::template ValueIter<
        MaskOnIterator, PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOnCIter = typename BaseLeaf::template ValueIter<
        MaskOnIterator, const PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOffIter = typename BaseLeaf::template ValueIter<
        MaskOffIterator, PointDataLeafNode, const ValueType, ValueOff>;
    using ValueOffCIter = typename BaseLeaf::template ValueIter<
        MaskOffIterator,const PointDataLeafNode,const ValueType,ValueOff>;
    using ValueAllIter = typename BaseLeaf::template ValueIter<
        MaskDenseIterator, PointDataLeafNode, const ValueType, ValueAll>;
    using ValueAllCIter = typename BaseLeaf::template ValueIter<
        MaskDenseIterator,const PointDataLeafNode,const ValueType,ValueAll>;
    using ChildOnIter = typename BaseLeaf::template ChildIter<
        MaskOnIterator, PointDataLeafNode, ChildOn>;
    using ChildOnCIter = typename BaseLeaf::template ChildIter<
        MaskOnIterator, const PointDataLeafNode, ChildOn>;
    using ChildOffIter = typename BaseLeaf::template ChildIter<
        MaskOffIterator, PointDataLeafNode, ChildOff>;
    using ChildOffCIter = typename BaseLeaf::template ChildIter<
        MaskOffIterator, const PointDataLeafNode, ChildOff>;
    using ChildAllIter = typename BaseLeaf::template DenseIter<
        PointDataLeafNode, ValueType, ChildAll>;
    using ChildAllCIter = typename BaseLeaf::template DenseIter<
        const PointDataLeafNode, const ValueType, ChildAll>;
#endif

    using IndexVoxelIter    = IndexIter<ValueVoxelCIter, NullFilter>;
    using IndexAllIter      = IndexIter<ValueAllCIter, NullFilter>;
    using IndexOnIter       = IndexIter<ValueOnCIter, NullFilter>;
    using IndexOffIter      = IndexIter<ValueOffCIter, NullFilter>;

    /// @brief Leaf index iterator
    IndexAllIter beginIndexAll() const
    {
	NullFilter filter;
	return this->beginIndex<ValueAllCIter, NullFilter>(filter);
    }
    IndexOnIter beginIndexOn() const
    {
	NullFilter filter;
	return this->beginIndex<ValueOnCIter, NullFilter>(filter);
    }
    IndexOffIter beginIndexOff() const
    {
	NullFilter filter;
	return this->beginIndex<ValueOffCIter, NullFilter>(filter);
    }

    template<typename IterT, typename FilterT>
    IndexIter<IterT, FilterT> beginIndex(const FilterT& filter) const;

    /// @brief Filtered leaf index iterator
    template<typename FilterT>
    IndexIter<ValueAllCIter, FilterT> beginIndexAll(const FilterT& filter) const
    {
	return this->beginIndex<ValueAllCIter, FilterT>(filter);
    }
    template<typename FilterT>
    IndexIter<ValueOnCIter, FilterT> beginIndexOn(const FilterT& filter) const
    {
	return this->beginIndex<ValueOnCIter, FilterT>(filter);
    }
    template<typename FilterT>
    IndexIter<ValueOffCIter, FilterT> beginIndexOff(const FilterT& filter) const
    {
	return this->beginIndex<ValueOffCIter, FilterT>(filter);
    }

    /// @brief Leaf index iterator from voxel
    IndexVoxelIter beginIndexVoxel(const Coord& ijk) const;

    /// @brief Filtered leaf index iterator from voxel
    template<typename FilterT>
    IndexIter<ValueVoxelCIter, FilterT> beginIndexVoxel(const Coord& ijk, const FilterT& filter) const;

#define VMASK_ this->getValueMask()
    ValueOnCIter  cbeginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnCIter   beginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnIter    beginValueOn()        { return ValueOnIter(VMASK_.beginOn(), this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffIter   beginValueOff()       { return ValueOffIter(VMASK_.beginOff(), this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllIter   beginValueAll()       { return ValueAllIter(VMASK_.beginDense(), this); }

    ValueOnCIter  cendValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnCIter   endValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnIter    endValueOn()          { return ValueOnIter(VMASK_.endOn(), this); }
    ValueOffCIter cendValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffCIter  endValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffIter   endValueOff()         { return ValueOffIter(VMASK_.endOff(), this); }
    ValueAllCIter cendValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllCIter  endValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllIter   endValueAll()         { return ValueAllIter(VMASK_.endDense(), this); }

    ChildOnCIter  cbeginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   beginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    beginChildOn()        { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cbeginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  beginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   beginChildOff()       { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cbeginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllCIter  beginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllIter   beginChildAll()       { return ChildAllIter(VMASK_.beginDense(), this); }

    ChildOnCIter  cendChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   endChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    endChildOn()          { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cendChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  endChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   endChildOff()         { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cendChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllCIter  endChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllIter   endChildAll()         { return ChildAllIter(VMASK_.endDense(), this); }
#undef VMASK_
}; // struct PointDataLeafNode

////////////////////////////////////////

// PointDataLeafNode implementation

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::initializeAttributes(const Descriptor::Ptr& descriptor, const Index arrayLength)
{
    if (descriptor->size() != 1 ||
        descriptor->find("P") == AttributeSet::INVALID_POS ||
        descriptor->valueType(0) != typeNameAsString<Vec3f>())
    {
        OPENVDB_THROW(IndexError, "Initializing attributes only allowed with one Vec3f position attribute.");
    }

    mAttributeSet.reset(new AttributeSet(descriptor, arrayLength));
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::clearAttributes(const bool updateValueMask)
{
    mAttributeSet.reset(new AttributeSet(*mAttributeSet, 0));

    // zero voxel values

    this->buffer().fill(ValueType(0));

    // if updateValueMask, also de-activate all voxels

    if (updateValueMask)    this->setValuesOff();
}

template<typename T, Index Log2Dim>
inline bool
PointDataLeafNode<T, Log2Dim>::hasAttribute(const size_t pos) const
{
    return pos < mAttributeSet->size();
}

template<typename T, Index Log2Dim>
inline bool
PointDataLeafNode<T, Log2Dim>::hasAttribute(const Name& attributeName) const
{
    const size_t pos = mAttributeSet->find(attributeName);
    return pos != AttributeSet::INVALID_POS;
}

template<typename T, Index Log2Dim>
inline AttributeArray::Ptr
PointDataLeafNode<T, Log2Dim>::appendAttribute( const Descriptor& expected, Descriptor::Ptr& replacement,
                                                const size_t pos, const Index strideOrTotalSize,
                                                const bool constantStride)
{
    return mAttributeSet->appendAttribute(expected, replacement, pos, strideOrTotalSize, constantStride);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::dropAttributes(const std::vector<size_t>& pos,
                    const Descriptor& expected, Descriptor::Ptr& replacement)
{
    mAttributeSet->dropAttributes(pos, expected, replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::reorderAttributes(const Descriptor::Ptr& replacement)
{
    mAttributeSet->reorderAttributes(replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::renameAttributes(const Descriptor& expected, Descriptor::Ptr& replacement)
{
    mAttributeSet->renameAttributes(expected, replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::compactAttributes()
{
    for (size_t i = 0; i < mAttributeSet->size(); i++) {
        AttributeArray* array = mAttributeSet->get(i);
        array->compact();
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::replaceAttributeSet(AttributeSet* attributeSet, bool allowMismatchingDescriptors)
{
    if (!attributeSet) {
        OPENVDB_THROW(ValueError, "Cannot replace with a null attribute set");
    }

    if (!allowMismatchingDescriptors && mAttributeSet->descriptor() != attributeSet->descriptor()) {
        OPENVDB_THROW(ValueError, "Attribute set descriptors are not equal.");
    }

    mAttributeSet.reset(attributeSet);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::resetDescriptor(const Descriptor::Ptr& replacement)
{
    mAttributeSet->resetDescriptor(replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsets(const std::vector<ValueType>& offsets, const bool updateValueMask)
{
    if (offsets.size() != LeafNodeType::NUM_VALUES) {
        OPENVDB_THROW(ValueError, "Offset vector size doesn't match number of voxels.")
    }

    for (Index index = 0; index < offsets.size(); ++index) {
        setOffsetOnly(index, offsets[index]);
    }

    if (updateValueMask) this->updateValueMask();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::validateOffsets() const
{
    // Ensure all of the offset values are monotonically increasing
    for (Index index = 1; index < BaseLeaf::SIZE; ++index) {
        if (this->getValue(index-1) > this->getValue(index)) {
            OPENVDB_THROW(ValueError, "Voxel offset values are not monotonically increasing");
        }
    }

    // Ensure all attribute arrays are of equal length
    for (size_t attributeIndex = 1; attributeIndex < mAttributeSet->size(); ++attributeIndex ) {
        if (mAttributeSet->getConst(attributeIndex-1)->size() != mAttributeSet->getConst(attributeIndex)->size()) {
            OPENVDB_THROW(ValueError, "Attribute arrays have inconsistent length");
        }
    }

    // Ensure the last voxel's offset value matches the size of each attribute array
    if (mAttributeSet->size() > 0 && this->getValue(BaseLeaf::SIZE-1) != mAttributeSet->getConst(0)->size()) {
        OPENVDB_THROW(ValueError, "Last voxel offset value does not match attribute array length");
    }
}

template<typename T, Index Log2Dim>
inline AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const size_t pos)
{
    if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range - " << pos);
    return *mAttributeSet->get(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const size_t pos) const
{
    if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range - " << pos);
    return *mAttributeSet->getConst(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::constAttributeArray(const size_t pos) const
{
    return this->attributeArray(pos);
}

template<typename T, Index Log2Dim>
inline AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const Name& attributeName)
{
    const size_t pos = mAttributeSet->find(attributeName);
    if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found - " << attributeName);
    return *mAttributeSet->get(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const Name& attributeName) const
{
    const size_t pos = mAttributeSet->find(attributeName);
    if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found - " << attributeName);
    return *mAttributeSet->getConst(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::constAttributeArray(const Name& attributeName) const
{
    return this->attributeArray(attributeName);
}

template<typename T, Index Log2Dim>
inline GroupHandle
PointDataLeafNode<T, Log2Dim>::groupHandle(const AttributeSet::Descriptor::GroupIndex& index) const
{
    const AttributeArray& array = this->attributeArray(index.first);
    assert(isGroup(array));

    const GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

    return GroupHandle(groupArray, index.second);
}

template<typename T, Index Log2Dim>
inline GroupHandle
PointDataLeafNode<T, Log2Dim>::groupHandle(const Name& name) const
{
    const AttributeSet::Descriptor::GroupIndex index = this->attributeSet().groupIndex(name);
    return this->groupHandle(index);
}

template<typename T, Index Log2Dim>
inline GroupWriteHandle
PointDataLeafNode<T, Log2Dim>::groupWriteHandle(const AttributeSet::Descriptor::GroupIndex& index)
{
    AttributeArray& array = this->attributeArray(index.first);
    assert(isGroup(array));

    GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

    return GroupWriteHandle(groupArray, index.second);
}

template<typename T, Index Log2Dim>
inline GroupWriteHandle
PointDataLeafNode<T, Log2Dim>::groupWriteHandle(const Name& name)
{
    const AttributeSet::Descriptor::GroupIndex index = this->attributeSet().groupIndex(name);
    return this->groupWriteHandle(index);
}

template<typename T, Index Log2Dim>
template<typename ValueIterT, typename FilterT>
inline IndexIter<ValueIterT, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndex(const FilterT& filter) const
{
    // generate no-op iterator if filter evaluates no indices

    if (filter.state() == index::NONE) {
        return IndexIter<ValueIterT, FilterT>(ValueIterT(), filter);
    }

    // copy filter to ensure thread-safety

    FilterT newFilter(filter);
    newFilter.reset(*this);

    using IterTraitsT = tree::IterTraits<LeafNodeType, ValueIterT>;

    // construct the value iterator and reset the filter to use this leaf

    ValueIterT valueIter = IterTraitsT::begin(*this);

    return IndexIter<ValueIterT, FilterT>(valueIter, newFilter);
}

template<typename T, Index Log2Dim>
inline ValueVoxelCIter
PointDataLeafNode<T, Log2Dim>::beginValueVoxel(const Coord& ijk) const
{
    const Index index = LeafNodeType::coordToOffset(ijk);
    assert(index < BaseLeaf::SIZE);
    const ValueType end = this->getValue(index);
    const ValueType start = (index == 0) ? ValueType(0) : this->getValue(index - 1);
    return ValueVoxelCIter(start, end);
}

template<typename T, Index Log2Dim>
inline typename PointDataLeafNode<T, Log2Dim>::IndexVoxelIter
PointDataLeafNode<T, Log2Dim>::beginIndexVoxel(const Coord& ijk) const
{
    ValueVoxelCIter iter = this->beginValueVoxel(ijk);
    return IndexVoxelIter(iter, NullFilter());
}

template<typename T, Index Log2Dim>
template<typename FilterT>
inline IndexIter<ValueVoxelCIter, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndexVoxel(const Coord& ijk, const FilterT& filter) const
{
    ValueVoxelCIter iter = this->beginValueVoxel(ijk);
    FilterT newFilter(filter);
    newFilter.reset(*this);
    return IndexIter<ValueVoxelCIter, FilterT>(iter, newFilter);
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::pointCount() const
{
    return this->getLastValue();
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::onPointCount() const
{
    if (this->isEmpty())        return 0;
    else if (this->isDense())   return this->pointCount();
    return iterCount(this->beginIndexOn());
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::offPointCount() const
{
    if (this->isEmpty())        return this->pointCount();
    else if (this->isDense())   return 0;
    return iterCount(this->beginIndexOff());
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::groupPointCount(const Name& groupName) const
{
    if (!this->attributeSet().descriptor().hasGroup(groupName)) {
        return Index64(0);
    }
    GroupFilter filter(groupName, this->attributeSet());
    if (filter.state() == index::ALL) {
        return this->pointCount();
    } else {
        return iterCount(this->beginIndexAll(filter));
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::updateValueMask()
{
    ValueType start = 0, end = 0;
    for (Index n = 0; n < LeafNodeType::NUM_VALUES; n++) {
        end = this->getValue(n);
        this->setValueMask(n, (end - start) > 0);
        start = end;
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsetOn(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
    this->setValueMaskOn(offset);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsetOnly(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readTopology(std::istream& is, bool fromHalf)
{
    BaseLeaf::readTopology(is, fromHalf);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::writeTopology(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeTopology(os, toHalf);
}

template<typename T, Index Log2Dim>
inline Index
PointDataLeafNode<T, Log2Dim>::buffers() const
{
    return Index(   /*voxel buffer sizes*/          1 +
                    /*voxel buffers*/               1 +
                    /*attribute metadata*/          1 +
                    /*attribute uniform values*/    mAttributeSet->size() +
                    /*attribute buffers*/           mAttributeSet->size() +
                    /*cleanup*/                     1);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    this->readBuffers(is, CoordBBox::inf(), fromHalf);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& /*bbox*/, bool fromHalf)
{
    struct Local
    {
        static void destroyPagedStream(const io::StreamMetadata::AuxDataMap& auxData, const Index index)
        {
            // if paged stream exists, delete it
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(it);
            }
        }

        static compression::PagedInputStream& getOrInsertPagedStream(   const io::StreamMetadata::AuxDataMap& auxData,
                                                                        const Index index)
        {
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                return *(boost::any_cast<compression::PagedInputStream::Ptr>(it->second));
            }
            else {
                compression::PagedInputStream::Ptr pagedStream = std::make_shared<compression::PagedInputStream>();
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[key] = pagedStream;
                return *pagedStream;
            }
        }

        static bool hasMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            return itMatching != auxData.end();
        }

        static void clearMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            std::string descriptorKey("descriptorPtr");
            auto itMatching = auxData.find(matchingKey);
            auto itDescriptor = auxData.find(descriptorKey);
            if (itMatching != auxData.end())    (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itMatching);
            if (itDescriptor != auxData.end())  (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itDescriptor);
        }

        static void insertDescriptor(   const io::StreamMetadata::AuxDataMap& auxData,
                                        const Descriptor::Ptr descriptor)
        {
            std::string descriptorKey("descriptorPtr");
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            if (itMatching == auxData.end()) {
                // if matching bool is not found, insert "true" and the descriptor
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[matchingKey] = true;
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[descriptorKey] = descriptor;
            }
        }

        static AttributeSet::Descriptor::Ptr retrieveMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string descriptorKey("descriptorPtr");
            auto itDescriptor = auxData.find(descriptorKey);
            assert(itDescriptor != auxData.end());
            const Descriptor::Ptr descriptor = boost::any_cast<AttributeSet::Descriptor::Ptr>(itDescriptor->second);
            return descriptor;
        }
    };

    const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(is);

    if (!meta) {
        OPENVDB_THROW(IoError, "Cannot read in a PointDataLeaf without StreamMetadata.");
    }

    const Index pass(static_cast<uint16_t>(meta->pass()));
    const Index maximumPass(static_cast<uint16_t>(meta->pass() >> 16));

    const Index attributes = (maximumPass - 4) / 2;

    if (pass == 0) {
        // pass 0 - voxel data sizes
        is.read(reinterpret_cast<char*>(&mVoxelBufferSize), sizeof(uint16_t));
        Local::clearMatchingDescriptor(meta->auxData());
    }
    else if (pass == 1) {
        // pass 1 - descriptor and attribute metadata
        if (Local::hasMatchingDescriptor(meta->auxData())) {
            AttributeSet::Descriptor::Ptr descriptor = Local::retrieveMatchingDescriptor(meta->auxData());
            mAttributeSet->resetDescriptor(descriptor, /*allowMismatchingDescriptors=*/true);
        }
        else {
            uint8_t header;
            is.read(reinterpret_cast<char*>(&header), sizeof(uint8_t));
            mAttributeSet->readDescriptor(is);
            if (header & uint8_t(1)) {
                AttributeSet::DescriptorPtr descriptor = mAttributeSet->descriptorPtr();
                Local::insertDescriptor(meta->auxData(), descriptor);
            }
            // a forwards-compatibility mechanism for future use,
            // if a 0x2 bit is set, read and skip over a specific number of bytes
            if (header & uint8_t(2)) {
                uint64_t bytesToSkip;
                is.read(reinterpret_cast<char*>(&bytesToSkip), sizeof(uint64_t));
                if (bytesToSkip > uint64_t(0)) {
                    auto metadata = io::getStreamMetadataPtr(is);
                    if (metadata && metadata->seekable()) {
                        is.seekg(bytesToSkip, std::ios_base::cur);
                    }
                    else {
                        std::vector<uint8_t> tempData(bytesToSkip);
                        is.read(reinterpret_cast<char*>(&tempData[0]), bytesToSkip);
                    }
                }
            }
            // this reader is only able to read headers with 0x1 and 0x2 bits set
            if (header > uint8_t(3)) {
                OPENVDB_THROW(IoError, "Unrecognised header flags in PointDataLeafNode");
            }
        }
        mAttributeSet->readMetadata(is);
    }
    else if (pass < (attributes + 2)) {
        // pass 2...n+2 - attribute uniform values
        const size_t attributeIndex = pass - 2;
        AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->get(attributeIndex) : nullptr;
        if (array) {
            compression::PagedInputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), static_cast<Index>(attributeIndex));
            pagedStream.setInputStream(is);
            pagedStream.setSizeOnly(true);
            array->readPagedBuffers(pagedStream);
        }
    }
    else if (pass == attributes + 2) {
        // pass n+2 - voxel data

        const Index passValue(meta->pass());

        // StreamMetadata pass variable used to temporarily store voxel buffer size
        io::StreamMetadata& nonConstMeta = const_cast<io::StreamMetadata&>(*meta);
        nonConstMeta.setPass(mVoxelBufferSize);

        // readBuffers() calls readCompressedValues specialization above
        BaseLeaf::readBuffers(is, fromHalf);

        // pass now reset to original value
        nonConstMeta.setPass(passValue);
    }
    else if (pass < (attributes*2 + 3)) {
        // pass n+2..2n+2 - attribute buffers
        const Index attributeIndex = pass - attributes - 3;
        AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->get(attributeIndex) : nullptr;
        if (array) {
            compression::PagedInputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), attributeIndex);
            pagedStream.setInputStream(is);
            pagedStream.setSizeOnly(false);
            array->readPagedBuffers(pagedStream);
        }
        // cleanup paged stream reference in auxiliary metadata
        if (pass > attributes + 3) {
            Local::destroyPagedStream(meta->auxData(), attributeIndex-1);
        }
    }
    else if (pass < buffers()) {
        // pass 2n+3 - cleanup last paged stream
        const Index attributeIndex = pass - attributes - 4;
        Local::destroyPagedStream(meta->auxData(), attributeIndex);
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    struct Local
    {
        static void destroyPagedStream(const io::StreamMetadata::AuxDataMap& auxData, const Index index)
        {
            // if paged stream exists, flush and delete it
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                compression::PagedOutputStream& stream = *(boost::any_cast<compression::PagedOutputStream::Ptr>(it->second));
                stream.flush();
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(it);
            }
        }

        static compression::PagedOutputStream& getOrInsertPagedStream(  const io::StreamMetadata::AuxDataMap& auxData,
                                                                        const Index index)
        {
            std::string key("paged:" + std::to_string(index));
            auto it = auxData.find(key);
            if (it != auxData.end()) {
                return *(boost::any_cast<compression::PagedOutputStream::Ptr>(it->second));
            }
            else {
                compression::PagedOutputStream::Ptr pagedStream = std::make_shared<compression::PagedOutputStream>();
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[key] = pagedStream;
                return *pagedStream;
            }
        }

        static void insertDescriptor(   const io::StreamMetadata::AuxDataMap& auxData,
                                        const Descriptor::Ptr descriptor)
        {
            std::string descriptorKey("descriptorPtr");
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            auto itDescriptor = auxData.find(descriptorKey);
            if (itMatching == auxData.end()) {
                // if matching bool is not found, insert "true" and the descriptor
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[matchingKey] = true;
                assert(itDescriptor == auxData.end());
                (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[descriptorKey] = descriptor;
            }
            else {
                // if matching bool is found and is false, early exit (a previous descriptor did not match)
                bool matching = boost::any_cast<bool>(itMatching->second);
                if (!matching)    return;
                assert(itDescriptor != auxData.end());
                // if matching bool is true, check whether the existing descriptor matches the current one and set
                // matching bool to false if not
                const Descriptor::Ptr existingDescriptor = boost::any_cast<AttributeSet::Descriptor::Ptr>(itDescriptor->second);
                if (*existingDescriptor != *descriptor) {
                    (const_cast<io::StreamMetadata::AuxDataMap&>(auxData))[matchingKey] = false;
                }
            }
        }

        static bool hasMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            auto itMatching = auxData.find(matchingKey);
            // if matching key is not found, no matching descriptor
            if (itMatching == auxData.end())                return false;
            // if matching key is found and is false, no matching descriptor
            if (!boost::any_cast<bool>(itMatching->second)) return false;
            return true;
        }

        static AttributeSet::Descriptor::Ptr retrieveMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string descriptorKey("descriptorPtr");
            auto itDescriptor = auxData.find(descriptorKey);
            // if matching key is true, however descriptor is not found, it has already been retrieved
            if (itDescriptor == auxData.end())              return nullptr;
            // otherwise remove it and return it
            const Descriptor::Ptr descriptor = boost::any_cast<AttributeSet::Descriptor::Ptr>(itDescriptor->second);
            (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itDescriptor);
            return descriptor;
        }

        static void clearMatchingDescriptor(const io::StreamMetadata::AuxDataMap& auxData)
        {
            std::string matchingKey("hasMatchingDescriptor");
            std::string descriptorKey("descriptorPtr");
            auto itMatching = auxData.find(matchingKey);
            auto itDescriptor = auxData.find(descriptorKey);
            if (itMatching != auxData.end())    (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itMatching);
            if (itDescriptor != auxData.end())  (const_cast<io::StreamMetadata::AuxDataMap&>(auxData)).erase(itDescriptor);
        }
    };

    const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(os);

    if (!meta) {
        OPENVDB_THROW(IoError, "Cannot write out a PointDataLeaf without StreamMetadata.");
    }

    const Index pass(static_cast<uint16_t>(meta->pass()));

    // leaf traversal analysis deduces the number of passes to perform for this leaf
    // then updates the leaf traversal value to ensure all passes will be written

    if (meta->countingPasses()) {
        const Index requiredPasses = this->buffers();
        if (requiredPasses > pass) {
            meta->setPass(requiredPasses);
        }
        return;
    }

    const Index maximumPass(static_cast<uint16_t>(meta->pass() >> 16));
    const Index attributes = (maximumPass - 4) / 2;

    if (pass == 0) {
        // pass 0 - voxel data sizes
        io::writeCompressedValuesSize(os, this->buffer().data(), SIZE);
        // track if descriptor is shared or not
        Local::insertDescriptor(meta->auxData(), mAttributeSet->descriptorPtr());
    }
    else if (pass == 1) {
        // pass 1 - descriptor and attribute metadata
        bool matchingDescriptor = Local::hasMatchingDescriptor(meta->auxData());
        if (matchingDescriptor) {
            AttributeSet::Descriptor::Ptr descriptor = Local::retrieveMatchingDescriptor(meta->auxData());
            if (descriptor) {
                // write a header to indicate a shared descriptor
                uint8_t header(1);
                os.write(reinterpret_cast<const char*>(&header), sizeof(uint8_t));
                mAttributeSet->writeDescriptor(os, /*transient=*/false);
            }
        }
        else {
            // write a header to indicate a non-shared descriptor
            uint8_t header(0);
            os.write(reinterpret_cast<const char*>(&header), sizeof(uint8_t));
            mAttributeSet->writeDescriptor(os, /*transient=*/false);
        }
        mAttributeSet->writeMetadata(os, /*transient=*/false, /*paged=*/true);
    }
    else if (pass < attributes + 2) {
        // pass 2...n+2 - attribute buffer sizes
        const Index attributeIndex = pass - 2;
        // destroy previous paged stream
        if (pass > 2) {
            Local::destroyPagedStream(meta->auxData(), attributeIndex-1);
        }
        const AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->getConst(attributeIndex) : nullptr;
        if (array) {
            compression::PagedOutputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), attributeIndex);
            pagedStream.setOutputStream(os);
            pagedStream.setSizeOnly(true);
            array->writePagedBuffers(pagedStream, /*outputTransient*/false);
        }
    }
    else if (pass == attributes + 2) {
        const Index attributeIndex = pass - 3;
        Local::destroyPagedStream(meta->auxData(), attributeIndex);
        // pass n+2 - voxel data
        BaseLeaf::writeBuffers(os, toHalf);
    }
    else if (pass < (attributes*2 + 3)) {
        // pass n+3...2n+3 - attribute buffers
        const Index attributeIndex = pass - attributes - 3;
        // destroy previous paged stream
        if (pass > attributes + 2) {
            Local::destroyPagedStream(meta->auxData(), attributeIndex-1);
        }
        const AttributeArray* array = attributeIndex < mAttributeSet->size() ?
            mAttributeSet->getConst(attributeIndex) : nullptr;
        if (array) {
            compression::PagedOutputStream& pagedStream =
                Local::getOrInsertPagedStream(meta->auxData(), attributeIndex);
            pagedStream.setOutputStream(os);
            pagedStream.setSizeOnly(false);
            array->writePagedBuffers(pagedStream, /*outputTransient*/false);
        }
    }
    else if (pass < buffers()) {
        Local::clearMatchingDescriptor(meta->auxData());
        // pass 2n+3 - cleanup last paged stream
        const Index attributeIndex = pass - attributes - 4;
        Local::destroyPagedStream(meta->auxData(), attributeIndex);
    }
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::memUsage() const
{
    return BaseLeaf::memUsage() + mAttributeSet->memUsage();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    BaseLeaf::evalActiveBoundingBox(bbox, visitVoxels);
}

template<typename T, Index Log2Dim>
inline CoordBBox
PointDataLeafNode<T, Log2Dim>::getNodeBoundingBox() const
{
    return BaseLeaf::getNodeBoundingBox();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    if (!this->allocate()) return;
#endif

    this->assertNonModifiableUnlessZero(value);

    // active state is permitted to be updated

    for (Int32 x = bbox.min().x(); x <= bbox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u)) << 2*Log2Dim;
        for (Int32 y = bbox.min().y(); y <= bbox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u)) << Log2Dim);
            for (Int32 z = bbox.min().z(); z <= bbox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                this->setValueMask(offset, active);
            }
        }
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::fill(const ValueType& value, bool active)
{
    this->assertNonModifiableUnlessZero(value);

    // active state is permitted to be updated

    if (active)                 this->setValuesOn();
    else                        this->setValuesOff();
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline AttributeSet::Descriptor::Ptr
makeDescriptorUnique(PointDataTreeT& tree)
{
    auto leafIter = tree.beginLeaf();
    if (!leafIter)  return nullptr;

    const AttributeSet::Descriptor& descriptor = leafIter->attributeSet().descriptor();
    auto newDescriptor = std::make_shared<AttributeSet::Descriptor>(descriptor);
    for (; leafIter; ++leafIter) {
        leafIter->resetDescriptor(newDescriptor);
    }

    return newDescriptor;
}


template <typename PointDataTreeT>
inline void
setStreamingMode(PointDataTreeT& tree, bool on)
{
    auto leafIter = tree.beginLeaf();
    for (; leafIter; ++leafIter) {
        for (size_t i = 0; i < leafIter->attributeSet().size(); i++) {
            leafIter->attributeArray(i).setStreaming(on);
        }
    }
}


template <typename PointDataTreeT>
inline void
prefetch(PointDataTreeT& tree, bool position, bool otherAttributes)
{
    // NOTE: the following is intentionally not multi-threaded, as the I/O
    // is faster if done in the order in which it is stored in the file

    auto leaf = tree.cbeginLeaf();
    if (!leaf)  return;

    const auto& attributeSet = leaf->attributeSet();

    // pre-fetch leaf data

    for ( ; leaf; ++leaf) {
        leaf->buffer().data();
    }

    // pre-fetch position attribute data (position will typically have index 0)

    size_t positionIndex = attributeSet.find("P");

    if (position && positionIndex != AttributeSet::INVALID_POS) {
        for (leaf = tree.cbeginLeaf(); leaf; ++leaf) {
            assert(leaf->hasAttribute(positionIndex));
            leaf->constAttributeArray(positionIndex).loadData();
        }
    }

    // pre-fetch other attribute data

    if (otherAttributes) {
        const size_t attributes = attributeSet.size();
        for (size_t attributeIndex = 0; attributeIndex < attributes; attributeIndex++) {
            if (attributeIndex == positionIndex)     continue;
            for (leaf = tree.cbeginLeaf(); leaf; ++leaf) {
                assert(leaf->hasAttribute(attributeIndex));
                leaf->constAttributeArray(attributeIndex).loadData();
            }
        }
    }
}


namespace internal {

/// @brief Global registration of point data-related types
/// @note This is called from @c openvdb::initialize, so there is
/// no need to call it directly.
void initialize();

/// @brief Global deregistration of point data-related types
/// @note This is called from @c openvdb::uninitialize, so there is
/// no need to call it directly.
void uninitialize();


/// @brief Recursive node chain which generates a boost::mpl::vector listing
/// value converted types of nodes to PointDataGrid nodes of the same configuration,
/// rooted at RootNodeType in reverse order, from LeafNode to RootNode.
/// See also TreeConverter<>.
template<typename HeadT, int HeadLevel>
struct PointDataNodeChain
{
    using SubtreeT = typename PointDataNodeChain<typename HeadT::ChildNodeType, HeadLevel-1>::Type;
    using RootNodeT = tree::RootNode<typename boost::mpl::back<SubtreeT>::type>;
    using Type = typename boost::mpl::push_back<SubtreeT, RootNodeT>::type;
};

// Specialization for internal nodes which require their embedded child type to
// be switched
template <typename ChildT, Index Log2Dim, int HeadLevel>
struct PointDataNodeChain<tree::InternalNode<ChildT, Log2Dim>, HeadLevel>
{
    using SubtreeT = typename PointDataNodeChain<ChildT, HeadLevel-1>::Type;
    using InternalNodeT = tree::InternalNode<typename boost::mpl::back<SubtreeT>::type, Log2Dim>;
    using Type = typename boost::mpl::push_back<SubtreeT, InternalNodeT>::type;
};

// Specialization for the last internal node of a node chain, expected
// to be templated on a leaf node
template <typename ChildT, Index Log2Dim>
struct PointDataNodeChain<tree::InternalNode<ChildT, Log2Dim>, /*HeadLevel=*/1>
{
    using LeafNodeT = PointDataLeafNode<PointDataIndex32, ChildT::LOG2DIM>;
    using InternalNodeT = tree::InternalNode<LeafNodeT, Log2Dim>;
    using Type = typename boost::mpl::vector<LeafNodeT, InternalNodeT>::type;
};

} // namespace internal


/// @brief Similiar to ValueConverter, but allows for tree configuration conversion
/// to a PointDataTree. ValueConverter<PointDataIndex32> cannot be used as a
/// PointDataLeafNode is not a specialization of LeafNode
template <typename TreeType>
struct TreeConverter {
    using RootNodeT = typename TreeType::RootNodeType;
    using NodeChainT = typename internal::PointDataNodeChain<RootNodeT, RootNodeT::LEVEL>::Type;
    using Type = tree::Tree<typename boost::mpl::back<NodeChainT>::type>;
};


} // namespace points


////////////////////////////////////////


namespace tree
{

/// Helper metafunction used to implement LeafNode::SameConfiguration
/// (which, as an inner class, can't be independently specialized)
template<Index Dim1, typename T2>
struct SameLeafConfig<Dim1, points::PointDataLeafNode<T2, Dim1>> { static const bool value = true; };

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_DATA_GRID_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
