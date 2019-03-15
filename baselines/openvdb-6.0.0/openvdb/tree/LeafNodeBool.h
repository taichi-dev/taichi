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

#ifndef OPENVDB_TREE_LEAF_NODE_BOOL_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_LEAF_NODE_BOOL_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/io/Compression.h> // for io::readData(), etc.
#include <openvdb/math/Math.h> // for math::isZero()
#include <openvdb/util/NodeMasks.h>
#include "LeafNode.h"
#include "Iterator.h"
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @brief LeafNode specialization for values of type bool that stores both
/// the active states and the values of (2^Log2Dim)^3 voxels as bit masks
template<Index Log2Dim>
class LeafNode<bool, Log2Dim>
{
public:
    using LeafNodeType = LeafNode<bool, Log2Dim>;
    using BuildType = bool;
    using ValueType = bool;
    using Buffer = LeafBuffer<ValueType, Log2Dim>;
    using NodeMaskType = util::NodeMask<Log2Dim>;
    using Ptr = SharedPtr<LeafNodeType>;

    // These static declarations must be on separate lines to avoid VC9 compiler errors.
    static const Index LOG2DIM    = Log2Dim;    // needed by parent nodes
    static const Index TOTAL      = Log2Dim;    // needed by parent nodes
    static const Index DIM        = 1 << TOTAL; // dimension along one coordinate direction
    static const Index NUM_VALUES = 1 << 3 * Log2Dim;
    static const Index NUM_VOXELS = NUM_VALUES; // total number of voxels represented by this node
    static const Index SIZE       = NUM_VALUES;
    static const Index LEVEL      = 0;          // level 0 = leaf

    /// @brief ValueConverter<T>::Type is the type of a LeafNode having the same
    /// dimensions as this node but a different value type, T.
    template<typename ValueType>
    struct ValueConverter { using Type = LeafNode<ValueType, Log2Dim>; };

    /// @brief SameConfiguration<OtherNodeType>::value is @c true if and only if
    /// OtherNodeType is the type of a LeafNode with the same dimensions as this node.
    template<typename OtherNodeType>
    struct SameConfiguration {
        static const bool value = SameLeafConfig<LOG2DIM, OtherNodeType>::value;
    };


    /// Default constructor
    LeafNode();

    /// Constructor
    /// @param xyz     the coordinates of a voxel that lies within the node
    /// @param value   the initial value for all of this node's voxels
    /// @param active  the active state to which to initialize all voxels
    explicit LeafNode(const Coord& xyz, bool value = false, bool active = false);

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// "Partial creation" constructor used during file input
    LeafNode(PartialCreate, const Coord& xyz, bool value = false, bool active = false);
#endif

    /// Deep copy constructor
    LeafNode(const LeafNode&);

    /// Deep assignment operator
    LeafNode& operator=(const LeafNode&) = default;

    /// Value conversion copy constructor
    template<typename OtherValueType>
    explicit LeafNode(const LeafNode<OtherValueType, Log2Dim>& other);

    /// Topology copy constructor
    template<typename ValueType>
    LeafNode(const LeafNode<ValueType, Log2Dim>& other, TopologyCopy);

    //@{
    /// @brief Topology copy constructor
    /// @note This variant exists mainly to enable template instantiation.
    template<typename ValueType>
    LeafNode(const LeafNode<ValueType, Log2Dim>& other, bool offValue, bool onValue, TopologyCopy);
    template<typename ValueType>
    LeafNode(const LeafNode<ValueType, Log2Dim>& other, bool background, TopologyCopy);
    //@}

    /// Destructor
    ~LeafNode();

    //
    // Statistics
    //
    /// Return log2 of the size of the buffer storage.
    static Index log2dim() { return Log2Dim; }
    /// Return the number of voxels in each dimension.
    static Index dim() { return DIM; }
    static Index size() { return SIZE; }
    static Index numValues() { return SIZE; }
    static Index getLevel() { return LEVEL; }
    static void getNodeLog2Dims(std::vector<Index>& dims) { dims.push_back(Log2Dim); }
    static Index getChildDim() { return 1; }

    static Index32 leafCount() { return 1; }
    static Index32 nonLeafCount() { return 0; }

    /// Return the number of active voxels.
    Index64 onVoxelCount() const { return mValueMask.countOn(); }
    /// Return the number of inactive voxels.
    Index64 offVoxelCount() const { return mValueMask.countOff(); }
    Index64 onLeafVoxelCount() const { return onVoxelCount(); }
    Index64 offLeafVoxelCount() const { return offVoxelCount(); }
    static Index64 onTileCount()  { return 0; }
    static Index64 offTileCount() { return 0; }

    /// Return @c true if this node has no active voxels.
    bool isEmpty() const { return mValueMask.isOff(); }
    /// Return @c true if this node only contains active voxels.
    bool isDense() const { return mValueMask.isOn(); }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// @brief Return @c true if memory for this node's buffer has been allocated.
    /// @details Currently, boolean leaf nodes don't support partial creation,
    /// so this always returns @c true.
    bool isAllocated() const { return true; }
    /// @brief Allocate memory for this node's buffer if it has not already been allocated.
    /// @details Currently, boolean leaf nodes don't support partial creation,
    /// so this has no effect.
    bool allocate() { return true; }
#endif

    /// Return the memory in bytes occupied by this node.
    Index64 memUsage() const;

    /// Expand the given bounding box so that it includes this leaf node's active voxels.
    /// If visitVoxels is false this LeafNode will be approximated as dense, i.e. with all
    /// voxels active. Else the individual active voxels are visited to produce a tight bbox.
    void evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels = true) const;

    /// @brief Return the bounding box of this node, i.e., the full index space
    /// spanned by this leaf node.
    CoordBBox getNodeBoundingBox() const { return CoordBBox::createCube(mOrigin, DIM); }

    /// Set the grid index coordinates of this node's local origin.
    void setOrigin(const Coord& origin) { mOrigin = origin; }
    //@{
    /// Return the grid index coordinates of this node's local origin.
    const Coord& origin() const { return mOrigin; }
    void getOrigin(Coord& origin) const { origin = mOrigin; }
    void getOrigin(Int32& x, Int32& y, Int32& z) const { mOrigin.asXYZ(x, y, z); }
    //@}

    /// Return the linear table offset of the given global or local coordinates.
    static Index coordToOffset(const Coord& xyz);
    /// @brief Return the local coordinates for a linear table offset,
    /// where offset 0 has coordinates (0, 0, 0).
    static Coord offsetToLocalCoord(Index n);
    /// Return the global coordinates for a linear table offset.
    Coord offsetToGlobalCoord(Index n) const;

    /// Return a string representation of this node.
    std::string str() const;

    /// @brief Return @c true if the given node (which may have a different @c ValueType
    /// than this node) has the same active value topology as this node.
    template<typename OtherType, Index OtherLog2Dim>
    bool hasSameTopology(const LeafNode<OtherType, OtherLog2Dim>* other) const;

    /// Check for buffer equivalence by value.
    bool operator==(const LeafNode&) const;
    bool operator!=(const LeafNode&) const;

    //
    // Buffer management
    //
    /// @brief Exchange this node's data buffer with the given data buffer
    /// without changing the active states of the values.
    void swap(Buffer& other) { mBuffer.swap(other); }
    const Buffer& buffer() const { return mBuffer; }
    Buffer& buffer() { return mBuffer; }

    //
    // I/O methods
    //
    /// Read in just the topology.
    void readTopology(std::istream&, bool fromHalf = false);
    /// Write out just the topology.
    void writeTopology(std::ostream&, bool toHalf = false) const;

    /// Read in the topology and the origin.
    void readBuffers(std::istream&, bool fromHalf = false);
    void readBuffers(std::istream& is, const CoordBBox&, bool fromHalf = false);
    /// Write out the topology and the origin.
    void writeBuffers(std::ostream&, bool toHalf = false) const;

    //
    // Accessor methods
    //
    /// Return the value of the voxel at the given coordinates.
    const bool& getValue(const Coord& xyz) const;
    /// Return the value of the voxel at the given offset.
    const bool& getValue(Index offset) const;

    /// @brief Return @c true if the voxel at the given coordinates is active.
    /// @param xyz       the coordinates of the voxel to be probed
    /// @param[out] val  the value of the voxel at the given coordinates
    bool probeValue(const Coord& xyz, bool& val) const;

    /// Return the level (0) at which leaf node values reside.
    static Index getValueLevel(const Coord&) { return LEVEL; }

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on);
    /// Set the active state of the voxel at the given offset but don't change its value.
    void setActiveState(Index offset, bool on) { assert(offset<SIZE); mValueMask.set(offset, on); }

    /// Set the value of the voxel at the given coordinates but don't change its active state.
    void setValueOnly(const Coord& xyz, bool val);
    /// Set the value of the voxel at the given offset but don't change its active state.
    void setValueOnly(Index offset, bool val) { assert(offset<SIZE); mBuffer.setValue(offset,val); }

    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz) { mValueMask.setOff(this->coordToOffset(xyz)); }
    /// Mark the voxel at the given offset as inactive but don't change its value.
    void setValueOff(Index offset) { assert(offset < SIZE); mValueMask.setOff(offset); }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, bool val);
    /// Set the value of the voxel at the given offset and mark the voxel as inactive.
    void setValueOff(Index offset, bool val);

    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz) { mValueMask.setOn(this->coordToOffset(xyz)); }
    /// Mark the voxel at the given offset as active but don't change its value.
    void setValueOn(Index offset) { assert(offset < SIZE); mValueMask.setOn(offset); }

    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValueOn(const Coord& xyz, bool val);
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, bool val) { this->setValueOn(xyz, val); }
    /// Set the value of the voxel at the given offset and mark the voxel as active.
    void setValueOn(Index offset, bool val);

    /// @brief Apply a functor to the value of the voxel at the given offset
    /// and mark the voxel as active.
    template<typename ModifyOp>
    void modifyValue(Index offset, const ModifyOp& op);
    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op);

    /// Apply a functor to the voxel at the given coordinates.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op);

    /// Mark all voxels as active but don't change their values.
    void setValuesOn() { mValueMask.setOn(); }
    /// Mark all voxels as inactive but don't change their values.
    void setValuesOff() { mValueMask.setOff(); }

    /// Return @c true if the voxel at the given coordinates is active.
    bool isValueOn(const Coord& xyz) const { return mValueMask.isOn(this->coordToOffset(xyz)); }
    /// Return @c true if the voxel at the given offset is active.
    bool isValueOn(Index offset) const { assert(offset < SIZE); return mValueMask.isOn(offset); }

    /// Return @c false since leaf nodes never contain tiles.
    static bool hasActiveTiles() { return false; }

    /// Set all voxels that lie outside the given axis-aligned box to the background.
    void clip(const CoordBBox&, bool background);

    /// Set all voxels within an axis-aligned box to the specified value and active state.
    void fill(const CoordBBox& bbox, bool value, bool active = true);
    /// Set all voxels within an axis-aligned box to the specified value and active state.
    void denseFill(const CoordBBox& bbox, bool val, bool on = true) { this->fill(bbox, val, on); }

    /// Set all voxels to the specified value but don't change their active states.
    void fill(const bool& value);
    /// Set all voxels to the specified value and active state.
    void fill(const bool& value, bool active);

    /// @brief Copy into a dense grid the values of the voxels that lie within
    /// a given bounding box.
    ///
    /// @param bbox   inclusive bounding box of the voxels to be copied into the dense grid
    /// @param dense  dense grid with a stride in @e z of one (see tools::Dense
    ///               in tools/Dense.h for the required API)
    ///
    /// @note @a bbox is assumed to be identical to or contained in the coordinate domains
    /// of both the dense grid and this node, i.e., no bounds checking is performed.
    /// @note Consider using tools::CopyToDense in tools/Dense.h
    /// instead of calling this method directly.
    template<typename DenseT>
    void copyToDense(const CoordBBox& bbox, DenseT& dense) const;

    /// @brief Copy from a dense grid into this node the values of the voxels
    /// that lie within a given bounding box.
    /// @details Only values that are different (by more than the given tolerance)
    /// from the background value will be active.  Other values are inactive
    /// and truncated to the background value.
    ///
    /// @param bbox        inclusive bounding box of the voxels to be copied into this node
    /// @param dense       dense grid with a stride in @e z of one (see tools::Dense
    ///                    in tools/Dense.h for the required API)
    /// @param background  background value of the tree that this node belongs to
    /// @param tolerance   tolerance within which a value equals the background value
    ///
    /// @note @a bbox is assumed to be identical to or contained in the coordinate domains
    /// of both the dense grid and this node, i.e., no bounds checking is performed.
    /// @note Consider using tools::CopyFromDense in tools/Dense.h
    /// instead of calling this method directly.
    template<typename DenseT>
    void copyFromDense(const CoordBBox& bbox, const DenseT& dense, bool background, bool tolerance);

    /// @brief Return the value of the voxel at the given coordinates.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    const bool& getValueAndCache(const Coord& xyz, AccessorT&) const {return this->getValue(xyz);}

    /// @brief Return @c true if the voxel at the given coordinates is active.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    bool isValueOnAndCache(const Coord& xyz, AccessorT&) const { return this->isValueOn(xyz); }

    /// @brief Change the value of the voxel at the given coordinates and mark it as active.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setValueAndCache(const Coord& xyz, bool val, AccessorT&) { this->setValueOn(xyz, val); }

    /// @brief Change the value of the voxel at the given coordinates
    /// but preserve its state.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setValueOnlyAndCache(const Coord& xyz, bool val, AccessorT&) {this->setValueOnly(xyz,val);}

    /// @brief Change the value of the voxel at the given coordinates and mark it as inactive.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setValueOffAndCache(const Coord& xyz, bool value, AccessorT&)
    {
        this->setValueOff(xyz, value);
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @note Used internally by ValueAccessor.
    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndCache(const Coord& xyz, const ModifyOp& op, AccessorT&)
    {
        this->modifyValue(xyz, op);
    }

    /// Apply a functor to the voxel at the given coordinates.
    /// @note Used internally by ValueAccessor.
    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndActiveStateAndCache(const Coord& xyz, const ModifyOp& op, AccessorT&)
    {
        this->modifyValueAndActiveState(xyz, op);
    }

    /// @brief Set the active state of the voxel at the given coordinates
    /// without changing its value.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setActiveStateAndCache(const Coord& xyz, bool on, AccessorT&)
    {
        this->setActiveState(xyz, on);
    }

    /// @brief Return @c true if the voxel at the given coordinates is active
    /// and return the voxel value in @a val.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    bool probeValueAndCache(const Coord& xyz, bool& val, AccessorT&) const
    {
        return this->probeValue(xyz, val);
    }

    /// @brief Return the LEVEL (=0) at which leaf node values reside.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    static Index getValueLevelAndCache(const Coord&, AccessorT&) { return LEVEL; }

    /// @brief Return a const reference to the first entry in the buffer.
    /// @note Since it's actually a reference to a static data member
    /// it should not be converted to a non-const pointer!
    const bool& getFirstValue() const { if (mValueMask.isOn(0)) return Buffer::sOn; else return Buffer::sOff; }
    /// @brief Return a const reference to the last entry in the buffer.
    /// @note Since it's actually a reference to a static data member
    /// it should not be converted to a non-const pointer!
    const bool& getLastValue() const { if (mValueMask.isOn(SIZE-1)) return Buffer::sOn; else return Buffer::sOff; }

    /// Return @c true if all of this node's voxels have the same active state
    /// and are equal to within the given tolerance, and return the value in
    /// @a constValue and the active state in @a state.
    bool isConstant(bool& constValue, bool& state, bool tolerance = 0) const;

    /// @brief Computes the median value of all the active and inactive voxels in this node.
    /// @return The median value.
    ///
    /// @details The median for boolean values is defined as the mode
    /// of the values, i.e. the value that occurs most often.
    bool medianAll() const;

    /// @brief Computes the median value of all the active voxels in this node.
    /// @return The number of active voxels.
    /// @param value Updated with the median value of all the active voxels.
    ///
    /// @details The median for boolean values is defined as the mode
    /// of the values, i.e. the value that occurs most often.
    Index medianOn(ValueType &value) const;

    /// @brief Computes the median value of all the inactive voxels in this node.
    /// @return The number of inactive voxels.
    /// @param value Updated with the median value of all the inactive voxels.
    ///
    /// @details The median for boolean values is defined as the mode
    /// of the values, i.e. the value that occurs most often.
    Index medianOff(ValueType &value) const;

    /// Return @c true if all of this node's values are inactive.
    bool isInactive() const { return mValueMask.isOff(); }

    void resetBackground(bool oldBackground, bool newBackground);

    void negate() { mBuffer.mData.toggle(); }

    template<MergePolicy Policy>
    void merge(const LeafNode& other, bool bg = false, bool otherBG = false);
    template<MergePolicy Policy> void merge(bool tileValue, bool tileActive);

    /// @brief No-op
    /// @details This function exists only to enable template instantiation.
    void voxelizeActiveTiles(bool = true) {}

    /// @brief Union this node's set of active values with the active values
    /// of the other node, whose @c ValueType may be different. So a
    /// resulting voxel will be active if either of the original voxels
    /// were active.
    ///
    /// @note This operation modifies only active states, not values.
    template<typename OtherType>
    void topologyUnion(const LeafNode<OtherType, Log2Dim>& other);

    /// @brief Intersect this node's set of active values with the active values
    /// of the other node, whose @c ValueType may be different. So a
    /// resulting voxel will be active only if both of the original voxels
    /// were active.
    ///
    /// @details The last dummy argument is required to match the signature
    /// for InternalNode::topologyIntersection.
    ///
    /// @note This operation modifies only active states, not
    /// values. Also note that this operation can result in all voxels
    /// being inactive so consider subsequnetly calling prune.
    template<typename OtherType>
    void topologyIntersection(const LeafNode<OtherType, Log2Dim>& other, const bool&);

    /// @brief Difference this node's set of active values with the active values
    /// of the other node, whose @c ValueType may be different. So a
    /// resulting voxel will be active only if the original voxel is
    /// active in this LeafNode and inactive in the other LeafNode.
    ///
    /// @details The last dummy argument is required to match the signature
    /// for InternalNode::topologyDifference.
    ///
    /// @note This operation modifies only active states, not values.
    /// Also, because it can deactivate all of this node's voxels,
    /// consider subsequently calling prune.
    template<typename OtherType>
    void topologyDifference(const LeafNode<OtherType, Log2Dim>& other, const bool&);

    template<typename CombineOp>
    void combine(const LeafNode& other, CombineOp& op);
    template<typename CombineOp>
    void combine(bool, bool valueIsActive, CombineOp& op);

    template<typename CombineOp, typename OtherType /*= bool*/>
    void combine2(const LeafNode& other, const OtherType&, bool valueIsActive, CombineOp&);
    template<typename CombineOp, typename OtherNodeT /*= LeafNode*/>
    void combine2(bool, const OtherNodeT& other, bool valueIsActive, CombineOp&);
    template<typename CombineOp, typename OtherNodeT /*= LeafNode*/>
    void combine2(const LeafNode& b0, const OtherNodeT& b1, CombineOp&);

    /// @brief Calls the templated functor BBoxOp with bounding box information.
    /// An additional level argument is provided to the callback.
    ///
    /// @note The bounding boxes are guarenteed to be non-overlapping.
    template<typename BBoxOp> void visitActiveBBox(BBoxOp&) const;

    template<typename VisitorOp> void visit(VisitorOp&);
    template<typename VisitorOp> void visit(VisitorOp&) const;

    template<typename OtherLeafNodeType, typename VisitorOp>
    void visit2Node(OtherLeafNodeType& other, VisitorOp&);
    template<typename OtherLeafNodeType, typename VisitorOp>
    void visit2Node(OtherLeafNodeType& other, VisitorOp&) const;
    template<typename IterT, typename VisitorOp>
    void visit2(IterT& otherIter, VisitorOp&, bool otherIsLHS = false);
    template<typename IterT, typename VisitorOp>
    void visit2(IterT& otherIter, VisitorOp&, bool otherIsLHS = false) const;

    //@{
    /// This function exists only to enable template instantiation.
    void prune(const ValueType& /*tolerance*/ = zeroVal<ValueType>()) {}
    void addLeaf(LeafNode*) {}
    template<typename AccessorT>
    void addLeafAndCache(LeafNode*, AccessorT&) {}
    template<typename NodeT>
    NodeT* stealNode(const Coord&, const ValueType&, bool) { return nullptr; }
    template<typename NodeT>
    NodeT* probeNode(const Coord&) { return nullptr; }
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord&) const { return nullptr; }
    template<typename ArrayT> void getNodes(ArrayT&) const {}
    template<typename ArrayT> void stealNodes(ArrayT&, const ValueType&, bool) {}
    //@}

    void addTile(Index level, const Coord&, bool val, bool active);
    void addTile(Index offset, bool val, bool active);
    template<typename AccessorT>
    void addTileAndCache(Index level, const Coord&, bool val, bool active, AccessorT&);

    //@{
    /// @brief Return a pointer to this node.
    LeafNode* touchLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    LeafNode* touchLeafAndCache(const Coord&, AccessorT&) { return this; }
    LeafNode* probeLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    LeafNode* probeLeafAndCache(const Coord&, AccessorT&) { return this; }
    template<typename NodeT, typename AccessorT>
    NodeT* probeNodeAndCache(const Coord&, AccessorT&)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(std::is_same<NodeT, LeafNode>::value)) return nullptr;
        return reinterpret_cast<NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}
    //@{
    /// @brief Return a @const pointer to this node.
    const LeafNode* probeLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const LeafNode* probeLeafAndCache(const Coord&, AccessorT&) const { return this; }
    const LeafNode* probeConstLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const LeafNode* probeConstLeafAndCache(const Coord&, AccessorT&) const { return this; }
    template<typename NodeT, typename AccessorT>
    const NodeT* probeConstNodeAndCache(const Coord&, AccessorT&) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(std::is_same<NodeT, LeafNode>::value)) return nullptr;
        return reinterpret_cast<const NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}

    //
    // Iterators
    //
protected:
    using MaskOnIter = typename NodeMaskType::OnIterator;
    using MaskOffIter = typename NodeMaskType::OffIterator;
    using MaskDenseIter = typename NodeMaskType::DenseIterator;

    template<typename MaskIterT, typename NodeT, typename ValueT>
    struct ValueIter:
        // Derives from SparseIteratorBase, but can also be used as a dense iterator,
        // if MaskIterT is a dense mask iterator type.
        public SparseIteratorBase<MaskIterT, ValueIter<MaskIterT, NodeT, ValueT>, NodeT, ValueT>
    {
        using BaseT = SparseIteratorBase<MaskIterT, ValueIter, NodeT, ValueT>;

        ValueIter() {}
        ValueIter(const MaskIterT& iter, NodeT* parent): BaseT(iter, parent) {}

        const bool& getItem(Index pos) const { return this->parent().getValue(pos); }
        const bool& getValue() const { return this->getItem(this->pos()); }

        // Note: setItem() can't be called on const iterators.
        void setItem(Index pos, bool value) const { this->parent().setValueOnly(pos, value); }
        // Note: setValue() can't be called on const iterators.
        void setValue(bool value) const { this->setItem(this->pos(), value); }

        // Note: modifyItem() can't be called on const iterators.
        template<typename ModifyOp>
        void modifyItem(Index n, const ModifyOp& op) const { this->parent().modifyValue(n, op); }
        // Note: modifyValue() can't be called on const iterators.
        template<typename ModifyOp>
        void modifyValue(const ModifyOp& op) const { this->modifyItem(this->pos(), op); }
    };

    /// Leaf nodes have no children, so their child iterators have no get/set accessors.
    template<typename MaskIterT, typename NodeT>
    struct ChildIter:
        public SparseIteratorBase<MaskIterT, ChildIter<MaskIterT, NodeT>, NodeT, bool>
    {
        ChildIter() {}
        ChildIter(const MaskIterT& iter, NodeT* parent): SparseIteratorBase<
            MaskIterT, ChildIter<MaskIterT, NodeT>, NodeT, bool>(iter, parent) {}
    };

    template<typename NodeT, typename ValueT>
    struct DenseIter: public DenseIteratorBase<
        MaskDenseIter, DenseIter<NodeT, ValueT>, NodeT, /*ChildT=*/void, ValueT>
    {
        using BaseT = DenseIteratorBase<MaskDenseIter, DenseIter, NodeT, void, ValueT>;
        using NonConstValueT = typename BaseT::NonConstValueType;

        DenseIter() {}
        DenseIter(const MaskDenseIter& iter, NodeT* parent): BaseT(iter, parent) {}

        bool getItem(Index pos, void*& child, NonConstValueT& value) const
        {
            value = this->parent().getValue(pos);
            child = nullptr;
            return false; // no child
        }

        // Note: setItem() can't be called on const iterators.
        //void setItem(Index pos, void* child) const {}

        // Note: unsetItem() can't be called on const iterators.
        void unsetItem(Index pos, const ValueT& val) const {this->parent().setValueOnly(pos, val);}
    };

public:
    using ValueOnIter = ValueIter<MaskOnIter, LeafNode, const bool>;
    using ValueOnCIter = ValueIter<MaskOnIter, const LeafNode, const bool>;
    using ValueOffIter = ValueIter<MaskOffIter, LeafNode, const bool>;
    using ValueOffCIter = ValueIter<MaskOffIter, const LeafNode, const bool>;
    using ValueAllIter = ValueIter<MaskDenseIter, LeafNode, const bool>;
    using ValueAllCIter = ValueIter<MaskDenseIter, const LeafNode, const bool>;
    using ChildOnIter = ChildIter<MaskOnIter, LeafNode>;
    using ChildOnCIter = ChildIter<MaskOnIter, const LeafNode>;
    using ChildOffIter = ChildIter<MaskOffIter, LeafNode>;
    using ChildOffCIter = ChildIter<MaskOffIter, const LeafNode>;
    using ChildAllIter = DenseIter<LeafNode, bool>;
    using ChildAllCIter = DenseIter<const LeafNode, const bool>;

    ValueOnCIter  cbeginValueOn() const { return ValueOnCIter(mValueMask.beginOn(), this); }
    ValueOnCIter   beginValueOn() const { return ValueOnCIter(mValueMask.beginOn(), this); }
    ValueOnIter    beginValueOn() { return ValueOnIter(mValueMask.beginOn(), this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(mValueMask.beginOff(), this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(mValueMask.beginOff(), this); }
    ValueOffIter   beginValueOff() { return ValueOffIter(mValueMask.beginOff(), this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(mValueMask.beginDense(), this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(mValueMask.beginDense(), this); }
    ValueAllIter   beginValueAll() { return ValueAllIter(mValueMask.beginDense(), this); }

    ValueOnCIter  cendValueOn() const { return ValueOnCIter(mValueMask.endOn(), this); }
    ValueOnCIter   endValueOn() const { return ValueOnCIter(mValueMask.endOn(), this); }
    ValueOnIter    endValueOn() { return ValueOnIter(mValueMask.endOn(), this); }
    ValueOffCIter cendValueOff() const { return ValueOffCIter(mValueMask.endOff(), this); }
    ValueOffCIter  endValueOff() const { return ValueOffCIter(mValueMask.endOff(), this); }
    ValueOffIter   endValueOff() { return ValueOffIter(mValueMask.endOff(), this); }
    ValueAllCIter cendValueAll() const { return ValueAllCIter(mValueMask.endDense(), this); }
    ValueAllCIter  endValueAll() const { return ValueAllCIter(mValueMask.endDense(), this); }
    ValueAllIter   endValueAll() { return ValueAllIter(mValueMask.endDense(), this); }

    // Note that [c]beginChildOn() and [c]beginChildOff() actually return end iterators,
    // because leaf nodes have no children.
    ChildOnCIter  cbeginChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnCIter   beginChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnIter    beginChildOn() { return ChildOnIter(mValueMask.endOn(), this); }
    ChildOffCIter cbeginChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffCIter  beginChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffIter   beginChildOff() { return ChildOffIter(mValueMask.endOff(), this); }
    ChildAllCIter cbeginChildAll() const { return ChildAllCIter(mValueMask.beginDense(), this); }
    ChildAllCIter  beginChildAll() const { return ChildAllCIter(mValueMask.beginDense(), this); }
    ChildAllIter   beginChildAll() { return ChildAllIter(mValueMask.beginDense(), this); }

    ChildOnCIter  cendChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnCIter   endChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnIter    endChildOn() { return ChildOnIter(mValueMask.endOn(), this); }
    ChildOffCIter cendChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffCIter  endChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffIter   endChildOff() { return ChildOffIter(mValueMask.endOff(), this); }
    ChildAllCIter cendChildAll() const { return ChildAllCIter(mValueMask.endDense(), this); }
    ChildAllCIter  endChildAll() const { return ChildAllCIter(mValueMask.endDense(), this); }
    ChildAllIter   endChildAll() { return ChildAllIter(mValueMask.endDense(), this); }

    //
    // Mask accessors
    //
    bool isValueMaskOn(Index n) const { return mValueMask.isOn(n); }
    bool isValueMaskOn() const { return mValueMask.isOn(); }
    bool isValueMaskOff(Index n) const { return mValueMask.isOff(n); }
    bool isValueMaskOff() const { return mValueMask.isOff(); }
    const NodeMaskType& getValueMask() const { return mValueMask; }
    const NodeMaskType& valueMask() const { return mValueMask; }
    NodeMaskType& getValueMask() { return mValueMask; }
    void setValueMask(const NodeMaskType& mask) { mValueMask = mask; }
    bool isChildMaskOn(Index) const { return false; } // leaf nodes have no children
    bool isChildMaskOff(Index) const { return true; }
    bool isChildMaskOff() const { return true; }
protected:
    void setValueMask(Index n, bool on) { mValueMask.set(n, on); }
    void setValueMaskOn(Index n)  { mValueMask.setOn(n); }
    void setValueMaskOff(Index n) { mValueMask.setOff(n); }

    /// Compute the origin of the leaf node that contains the voxel with the given coordinates.
    static void evalNodeOrigin(Coord& xyz) { xyz &= ~(DIM - 1); }

    template<typename NodeT, typename VisitorOp, typename ChildAllIterT>
    static inline void doVisit(NodeT&, VisitorOp&);

    template<typename NodeT, typename OtherNodeT, typename VisitorOp,
        typename ChildAllIterT, typename OtherChildAllIterT>
    static inline void doVisit2Node(NodeT& self, OtherNodeT& other, VisitorOp&);

    template<typename NodeT, typename VisitorOp,
        typename ChildAllIterT, typename OtherChildAllIterT>
    static inline void doVisit2(NodeT& self, OtherChildAllIterT&, VisitorOp&, bool otherIsLHS);


    /// Bitmask that determines which voxels are active
    NodeMaskType mValueMask;
    /// Bitmask representing the values of voxels
    Buffer mBuffer;
    /// Global grid index coordinates (x,y,z) of the local origin of this node
    Coord mOrigin;

private:
    /// @brief During topology-only construction, access is needed
    /// to protected/private members of other template instances.
    template<typename, Index> friend class LeafNode;

    friend struct ValueIter<MaskOnIter, LeafNode, bool>;
    friend struct ValueIter<MaskOffIter, LeafNode, bool>;
    friend struct ValueIter<MaskDenseIter, LeafNode, bool>;
    friend struct ValueIter<MaskOnIter, const LeafNode, bool>;
    friend struct ValueIter<MaskOffIter, const LeafNode, bool>;
    friend struct ValueIter<MaskDenseIter, const LeafNode, bool>;

    //@{
    /// Allow iterators to call mask accessor methods (see below).
    /// @todo Make mask accessors public?
    friend class IteratorBase<MaskOnIter, LeafNode>;
    friend class IteratorBase<MaskOffIter, LeafNode>;
    friend class IteratorBase<MaskDenseIter, LeafNode>;
    //@}

}; // class LeafNode<bool>


////////////////////////////////////////


template<Index Log2Dim>
inline
LeafNode<bool, Log2Dim>::LeafNode()
  : mOrigin(0, 0, 0)
{
}


template<Index Log2Dim>
inline
LeafNode<bool, Log2Dim>::LeafNode(const Coord& xyz, bool value, bool active)
    : mValueMask(active)
    , mBuffer(value)
    , mOrigin(xyz & (~(DIM - 1)))
{
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3
template<Index Log2Dim>
inline
LeafNode<bool, Log2Dim>::LeafNode(PartialCreate, const Coord& xyz, bool value, bool active)
    : mValueMask(active)
    , mBuffer(value)
    , mOrigin(xyz & (~(DIM - 1)))
{
    /// @todo For now, this is identical to the non-PartialCreate constructor.
    /// Consider modifying the Buffer class to allow it to be constructed
    /// without allocating a bitmask.
}
#endif


template<Index Log2Dim>
inline
LeafNode<bool, Log2Dim>::LeafNode(const LeafNode &other)
    : mValueMask(other.valueMask())
    , mBuffer(other.mBuffer)
    , mOrigin(other.mOrigin)
{
}


// Copy-construct from a leaf node with the same configuration but a different ValueType.
template<Index Log2Dim>
template<typename ValueT>
inline
LeafNode<bool, Log2Dim>::LeafNode(const LeafNode<ValueT, Log2Dim>& other)
    : mValueMask(other.valueMask())
    , mOrigin(other.origin())
{
    struct Local {
        /// @todo Consider using a value conversion functor passed as an argument instead.
        static inline bool convertValue(const ValueT& val) { return bool(val); }
    };

    for (Index i = 0; i < SIZE; ++i) {
         mBuffer.setValue(i, Local::convertValue(other.mBuffer[i]));
    }
}


template<Index Log2Dim>
template<typename ValueT>
inline
LeafNode<bool, Log2Dim>::LeafNode(const LeafNode<ValueT, Log2Dim>& other,
                                  bool background, TopologyCopy)
    : mValueMask(other.valueMask())
    , mBuffer(background)
    , mOrigin(other.origin())
{
}


template<Index Log2Dim>
template<typename ValueT>
inline
LeafNode<bool, Log2Dim>::LeafNode(const LeafNode<ValueT, Log2Dim>& other, TopologyCopy)
    : mValueMask(other.valueMask())
    , mBuffer(other.valueMask())// value = active state
    , mOrigin(other.origin())
{
}


template<Index Log2Dim>
template<typename ValueT>
inline
LeafNode<bool, Log2Dim>::LeafNode(const LeafNode<ValueT, Log2Dim>& other,
                                  bool offValue, bool onValue, TopologyCopy)
    : mValueMask(other.valueMask())
    , mBuffer(other.valueMask())
    , mOrigin(other.origin())
{
    if (offValue) { if (!onValue) mBuffer.mData.toggle(); else mBuffer.mData.setOn(); }
}


template<Index Log2Dim>
inline
LeafNode<bool, Log2Dim>::~LeafNode()
{
}


////////////////////////////////////////


template<Index Log2Dim>
inline Index64
LeafNode<bool, Log2Dim>::memUsage() const
{
    // Use sizeof(*this) to capture alignment-related padding
    return sizeof(*this);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    CoordBBox this_bbox = this->getNodeBoundingBox();
    if (bbox.isInside(this_bbox)) return;//this LeafNode is already enclosed in the bbox
    if (ValueOnCIter iter = this->cbeginValueOn()) {//any active values?
        if (visitVoxels) {//use voxel granularity?
            this_bbox.reset();
            for(; iter; ++iter) this_bbox.expand(this->offsetToLocalCoord(iter.pos()));
            this_bbox.translate(this->origin());
        }
        bbox.expand(this_bbox);
    }
}


template<Index Log2Dim>
template<typename OtherType, Index OtherLog2Dim>
inline bool
LeafNode<bool, Log2Dim>::hasSameTopology(const LeafNode<OtherType, OtherLog2Dim>* other) const
{
    assert(other);
    return (Log2Dim == OtherLog2Dim && mValueMask == other->getValueMask());
}


template<Index Log2Dim>
inline std::string
LeafNode<bool, Log2Dim>::str() const
{
    std::ostringstream ostr;
    ostr << "LeafNode @" << mOrigin << ": ";
    for (Index32 n = 0; n < SIZE; ++n) ostr << (mValueMask.isOn(n) ? '#' : '.');
    return ostr.str();
}


////////////////////////////////////////


template<Index Log2Dim>
inline Index
LeafNode<bool, Log2Dim>::coordToOffset(const Coord& xyz)
{
    assert ((xyz[0] & (DIM-1u)) < DIM && (xyz[1] & (DIM-1u)) < DIM && (xyz[2] & (DIM-1u)) < DIM);
    return ((xyz[0] & (DIM-1u)) << 2*Log2Dim)
         + ((xyz[1] & (DIM-1u)) << Log2Dim)
         +  (xyz[2] & (DIM-1u));
}


template<Index Log2Dim>
inline Coord
LeafNode<bool, Log2Dim>::offsetToLocalCoord(Index n)
{
    assert(n < (1 << 3*Log2Dim));
    Coord xyz;
    xyz.setX(n >> 2*Log2Dim);
    n &= ((1 << 2*Log2Dim) - 1);
    xyz.setY(n >> Log2Dim);
    xyz.setZ(n & ((1 << Log2Dim) - 1));
    return xyz;
}


template<Index Log2Dim>
inline Coord
LeafNode<bool, Log2Dim>::offsetToGlobalCoord(Index n) const
{
    return (this->offsetToLocalCoord(n) + this->origin());
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::readTopology(std::istream& is, bool /*fromHalf*/)
{
    mValueMask.load(is);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::writeTopology(std::ostream& os, bool /*toHalf*/) const
{
    mValueMask.save(os);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& clipBBox, bool fromHalf)
{
    // Boolean LeafNodes don't currently implement lazy loading.
    // Instead, load the full buffer, then clip it.

    this->readBuffers(is, fromHalf);

    // Get this tree's background value.
    bool background = false;
    if (const void* bgPtr = io::getGridBackgroundValuePtr(is)) {
        background = *static_cast<const bool*>(bgPtr);
    }
    this->clip(clipBBox, background);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::readBuffers(std::istream& is, bool /*fromHalf*/)
{
    // Read in the value mask.
    mValueMask.load(is);
    // Read in the origin.
    is.read(reinterpret_cast<char*>(&mOrigin), sizeof(Coord::ValueType) * 3);

    if (io::getFormatVersion(is) >= OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION) {
        // Read in the mask for the voxel values.
        mBuffer.mData.load(is);
    } else {
        // Older files stored one or more bool arrays.

        // Read in the number of buffers, which should now always be one.
        int8_t numBuffers = 0;
        is.read(reinterpret_cast<char*>(&numBuffers), sizeof(int8_t));

        // Read in the buffer.
        // (Note: prior to the bool leaf optimization, buffers were always compressed.)
        std::unique_ptr<bool[]> buf{new bool[SIZE]};
        io::readData<bool>(is, buf.get(), SIZE, /*isCompressed=*/true);

        // Transfer values to mBuffer.
        mBuffer.mData.setOff();
        for (Index i = 0; i < SIZE; ++i) {
            if (buf[i]) mBuffer.mData.setOn(i);
        }

        if (numBuffers > 1) {
            // Read in and discard auxiliary buffers that were created with
            // earlier versions of the library.
            for (int i = 1; i < numBuffers; ++i) {
                io::readData<bool>(is, buf.get(), SIZE, /*isCompressed=*/true);
            }
        }
    }
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::writeBuffers(std::ostream& os, bool /*toHalf*/) const
{
    // Write out the value mask.
    mValueMask.save(os);
    // Write out the origin.
    os.write(reinterpret_cast<const char*>(&mOrigin), sizeof(Coord::ValueType) * 3);
    // Write out the voxel values.
    mBuffer.mData.save(os);
}


////////////////////////////////////////


template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::operator==(const LeafNode& other) const
{
    return mOrigin == other.mOrigin &&
           mValueMask == other.valueMask() &&
           mBuffer == other.mBuffer;
}


template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::operator!=(const LeafNode& other) const
{
    return !(this->operator==(other));
}


////////////////////////////////////////


template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::isConstant(bool& constValue, bool& state, bool tolerance) const
{
    if (!mValueMask.isConstant(state)) return false;

    // Note: if tolerance is true (i.e., 1), then all boolean values compare equal.
    if (!tolerance && !(mBuffer.mData.isOn() || mBuffer.mData.isOff())) return false;

    constValue = mBuffer.mData.isOn();
    return true;
}

////////////////////////////////////////

template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::medianAll() const
{
    const Index countTrue = mBuffer.mData.countOn();
    return countTrue > (NUM_VALUES >> 1);
}

template<Index Log2Dim>
inline Index
LeafNode<bool, Log2Dim>::medianOn(bool& state) const
{
    const NodeMaskType tmp = mBuffer.mData & mValueMask;//both true and active
    const Index countTrueOn = tmp.countOn(), countOn = mValueMask.countOn();
    state = countTrueOn > (NUM_VALUES >> 1);
    return countOn;
}

template<Index Log2Dim>
inline Index
LeafNode<bool, Log2Dim>::medianOff(bool& state) const
{
    const NodeMaskType tmp = mBuffer.mData & (!mValueMask);//both true and inactive
    const Index countTrueOff = tmp.countOn(), countOff = mValueMask.countOff();
    state = countTrueOff > (NUM_VALUES >> 1);
    return countOff;
}

////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::addTile(Index /*level*/, const Coord& xyz, bool val, bool active)
{
    this->addTile(this->coordToOffset(xyz), val, active);
}

template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::addTile(Index offset, bool val, bool active)
{
    assert(offset < SIZE);
    this->setValueOnly(offset, val);
    this->setActiveState(offset, active);
}

template<Index Log2Dim>
template<typename AccessorT>
inline void
LeafNode<bool, Log2Dim>::addTileAndCache(Index level, const Coord& xyz,
    bool val, bool active, AccessorT&)
{
    this->addTile(level, xyz, val, active);
}


////////////////////////////////////////


template<Index Log2Dim>
inline const bool&
LeafNode<bool, Log2Dim>::getValue(const Coord& xyz) const
{
    // This *CANNOT* use operator ? because Visual C++
    if (mBuffer.mData.isOn(this->coordToOffset(xyz))) return Buffer::sOn; else return Buffer::sOff;
}


template<Index Log2Dim>
inline const bool&
LeafNode<bool, Log2Dim>::getValue(Index offset) const
{
    assert(offset < SIZE);
    // This *CANNOT* use operator ? for Windows
    if (mBuffer.mData.isOn(offset)) return Buffer::sOn; else return Buffer::sOff;
}


template<Index Log2Dim>
inline bool
LeafNode<bool, Log2Dim>::probeValue(const Coord& xyz, bool& val) const
{
    const Index offset = this->coordToOffset(xyz);
    val = mBuffer.mData.isOn(offset);
    return mValueMask.isOn(offset);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::setValueOn(const Coord& xyz, bool val)
{
    this->setValueOn(this->coordToOffset(xyz), val);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::setValueOn(Index offset, bool val)
{
    assert(offset < SIZE);
    mValueMask.setOn(offset);
    mBuffer.mData.set(offset, val);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::setValueOnly(const Coord& xyz, bool val)
{
    this->setValueOnly(this->coordToOffset(xyz), val);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::setActiveState(const Coord& xyz, bool on)
{
    mValueMask.set(this->coordToOffset(xyz), on);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::setValueOff(const Coord& xyz, bool val)
{
    this->setValueOff(this->coordToOffset(xyz), val);
}


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::setValueOff(Index offset, bool val)
{
    assert(offset < SIZE);
    mValueMask.setOff(offset);
    mBuffer.mData.set(offset, val);
}


template<Index Log2Dim>
template<typename ModifyOp>
inline void
LeafNode<bool, Log2Dim>::modifyValue(Index offset, const ModifyOp& op)
{
    bool val = mBuffer.mData.isOn(offset);
    op(val);
    mBuffer.mData.set(offset, val);
    mValueMask.setOn(offset);
}


template<Index Log2Dim>
template<typename ModifyOp>
inline void
LeafNode<bool, Log2Dim>::modifyValue(const Coord& xyz, const ModifyOp& op)
{
    this->modifyValue(this->coordToOffset(xyz), op);
}


template<Index Log2Dim>
template<typename ModifyOp>
inline void
LeafNode<bool, Log2Dim>::modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
{
    const Index offset = this->coordToOffset(xyz);
    bool val = mBuffer.mData.isOn(offset), state = mValueMask.isOn(offset);
    op(val, state);
    mBuffer.mData.set(offset, val);
    mValueMask.set(offset, state);
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::resetBackground(bool oldBackground, bool newBackground)
{
    if (newBackground != oldBackground) {
        // Flip mBuffer's background bits and zero its foreground bits.
        NodeMaskType bgMask = !(mBuffer.mData | mValueMask);
        // Overwrite mBuffer's background bits, leaving its foreground bits intact.
        mBuffer.mData = (mBuffer.mData & mValueMask) | bgMask;
    }
}


////////////////////////////////////////


template<Index Log2Dim>
template<MergePolicy Policy>
inline void
LeafNode<bool, Log2Dim>::merge(const LeafNode& other, bool /*bg*/, bool /*otherBG*/)
{
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    if (Policy == MERGE_NODES) return;
    for (typename NodeMaskType::OnIterator iter = other.valueMask().beginOn(); iter; ++iter) {
        const Index n = iter.pos();
        if (mValueMask.isOff(n)) {
            mBuffer.mData.set(n, other.mBuffer.mData.isOn(n));
            mValueMask.setOn(n);
        }
    }
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}

template<Index Log2Dim>
template<MergePolicy Policy>
inline void
LeafNode<bool, Log2Dim>::merge(bool tileValue, bool tileActive)
{
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    if (Policy != MERGE_ACTIVE_STATES_AND_NODES) return;
    if (!tileActive) return;
    // Replace all inactive values with the active tile value.
    if (tileValue) mBuffer.mData |= !mValueMask; // -0=>1, +0=>0, -1=>1, +1=>1 (-,+ = off,on)
    else mBuffer.mData &= mValueMask;            // -0=>0, +0=>0, -1=>0, +1=>1
    mValueMask.setOn();
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}


////////////////////////////////////////


template<Index Log2Dim>
template<typename OtherType>
inline void
LeafNode<bool, Log2Dim>::topologyUnion(const LeafNode<OtherType, Log2Dim>& other)
{
    mValueMask |= other.valueMask();
}


template<Index Log2Dim>
template<typename OtherType>
inline void
LeafNode<bool, Log2Dim>::topologyIntersection(const LeafNode<OtherType, Log2Dim>& other,
                                              const bool&)
{
    mValueMask &= other.valueMask();
}


template<Index Log2Dim>
template<typename OtherType>
inline void
LeafNode<bool, Log2Dim>::topologyDifference(const LeafNode<OtherType, Log2Dim>& other,
                                            const bool&)
{
    mValueMask &= !other.valueMask();
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::clip(const CoordBBox& clipBBox, bool background)
{
    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.  Fill it with background tiles.
        this->fill(nodeBBox, background, /*active=*/false);
    } else if (clipBBox.isInside(nodeBBox)) {
        // This node lies completely inside the clipping region.  Leave it intact.
        return;
    }

    // This node isn't completely contained inside the clipping region.
    // Set any voxels that lie outside the region to the background value.

    // Construct a boolean mask that is on inside the clipping region and off outside it.
    NodeMaskType mask;
    nodeBBox.intersect(clipBBox);
    Coord xyz;
    int &x = xyz.x(), &y = xyz.y(), &z = xyz.z();
    for (x = nodeBBox.min().x(); x <= nodeBBox.max().x(); ++x) {
        for (y = nodeBBox.min().y(); y <= nodeBBox.max().y(); ++y) {
            for (z = nodeBBox.min().z(); z <= nodeBBox.max().z(); ++z) {
                mask.setOn(static_cast<Index32>(this->coordToOffset(xyz)));
            }
        }
    }

    // Set voxels that lie in the inactive region of the mask (i.e., outside
    // the clipping region) to the background value.
    for (MaskOffIter maskIter = mask.beginOff(); maskIter; ++maskIter) {
        this->setValueOff(maskIter.pos(), background);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::fill(const CoordBBox& bbox, bool value, bool active)
{
    auto clippedBBox = this->getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    if (!clippedBBox) return;

    for (Int32 x = clippedBBox.min().x(); x <= clippedBBox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u))<<2*Log2Dim;
        for (Int32 y = clippedBBox.min().y(); y <= clippedBBox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u))<<  Log2Dim);
            for (Int32 z = clippedBBox.min().z(); z <= clippedBBox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                mValueMask.set(offset, active);
                mBuffer.mData.set(offset, value);
            }
        }
    }
}

template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::fill(const bool& value)
{
    mBuffer.fill(value);
}

template<Index Log2Dim>
inline void
LeafNode<bool, Log2Dim>::fill(const bool& value, bool active)
{
    mBuffer.fill(value);
    mValueMask.set(active);
}


////////////////////////////////////////


template<Index Log2Dim>
template<typename DenseT>
inline void
LeafNode<bool, Log2Dim>::copyToDense(const CoordBBox& bbox, DenseT& dense) const
{
    using DenseValueType = typename DenseT::ValueType;

    const size_t xStride = dense.xStride(), yStride = dense.yStride(), zStride = dense.zStride();
    const Coord& min = dense.bbox().min();
    DenseValueType* t0 = dense.data() + zStride * (bbox.min()[2] - min[2]); // target array
    const Int32 n0 = bbox.min()[2] & (DIM-1u);
    for (Int32 x = bbox.min()[0], ex = bbox.max()[0] + 1; x < ex; ++x) {
        DenseValueType* t1 = t0 + xStride * (x - min[0]);
        const Int32 n1 = n0 + ((x & (DIM-1u)) << 2*LOG2DIM);
        for (Int32 y = bbox.min()[1], ey = bbox.max()[1] + 1; y < ey; ++y) {
            DenseValueType* t2 = t1 + yStride * (y - min[1]);
            Int32 n2 = n1 + ((y & (DIM-1u)) << LOG2DIM);
            for (Int32 z = bbox.min()[2], ez = bbox.max()[2] + 1; z < ez; ++z, t2 += zStride) {
                *t2 = DenseValueType(mBuffer.mData.isOn(n2++));
            }
        }
    }
}


template<Index Log2Dim>
template<typename DenseT>
inline void
LeafNode<bool, Log2Dim>::copyFromDense(const CoordBBox& bbox, const DenseT& dense,
                                       bool background, bool tolerance)
{
    using DenseValueType = typename DenseT::ValueType;
    struct Local {
        inline static bool toBool(const DenseValueType& v) { return !math::isZero(v); }
    };

    const size_t xStride = dense.xStride(), yStride = dense.yStride(), zStride = dense.zStride();
    const Coord& min = dense.bbox().min();
    const DenseValueType* s0 = dense.data() + zStride * (bbox.min()[2] - min[2]); // source
    const Int32 n0 = bbox.min()[2] & (DIM-1u);
    for (Int32 x = bbox.min()[0], ex = bbox.max()[0] + 1; x < ex; ++x) {
        const DenseValueType* s1 = s0 + xStride * (x - min[0]);
        const Int32 n1 = n0 + ((x & (DIM-1u)) << 2*LOG2DIM);
        for (Int32 y = bbox.min()[1], ey = bbox.max()[1] + 1; y < ey; ++y) {
            const DenseValueType* s2 = s1 + yStride * (y - min[1]);
            Int32 n2 = n1 + ((y & (DIM-1u)) << LOG2DIM);
            for (Int32 z = bbox.min()[2], ez = bbox.max()[2]+1; z < ez; ++z, ++n2, s2 += zStride) {
                // Note: if tolerance is true (i.e., 1), then all boolean values compare equal.
                if (tolerance || (background == Local::toBool(*s2))) {
                    mValueMask.setOff(n2);
                    mBuffer.mData.set(n2, background);
                } else {
                    mValueMask.setOn(n2);
                    mBuffer.mData.set(n2, Local::toBool(*s2));
                }
            }
        }
    }
}


////////////////////////////////////////


template<Index Log2Dim>
template<typename CombineOp>
inline void
LeafNode<bool, Log2Dim>::combine(const LeafNode& other, CombineOp& op)
{
    CombineArgs<bool> args;
    for (Index i = 0; i < SIZE; ++i) {
        bool result = false, aVal = mBuffer.mData.isOn(i), bVal = other.mBuffer.mData.isOn(i);
        op(args.setARef(aVal)
            .setAIsActive(mValueMask.isOn(i))
            .setBRef(bVal)
            .setBIsActive(other.valueMask().isOn(i))
            .setResultRef(result));
        mValueMask.set(i, args.resultIsActive());
        mBuffer.mData.set(i, result);
    }
}


template<Index Log2Dim>
template<typename CombineOp>
inline void
LeafNode<bool, Log2Dim>::combine(bool value, bool valueIsActive, CombineOp& op)
{
    CombineArgs<bool> args;
    args.setBRef(value).setBIsActive(valueIsActive);
    for (Index i = 0; i < SIZE; ++i) {
        bool result = false, aVal = mBuffer.mData.isOn(i);
        op(args.setARef(aVal)
            .setAIsActive(mValueMask.isOn(i))
            .setResultRef(result));
        mValueMask.set(i, args.resultIsActive());
        mBuffer.mData.set(i, result);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
template<typename CombineOp, typename OtherType>
inline void
LeafNode<bool, Log2Dim>::combine2(const LeafNode& other, const OtherType& value,
    bool valueIsActive, CombineOp& op)
{
    CombineArgs<bool, OtherType> args;
    args.setBRef(value).setBIsActive(valueIsActive);
    for (Index i = 0; i < SIZE; ++i) {
        bool result = false, aVal = other.mBuffer.mData.isOn(i);
        op(args.setARef(aVal)
            .setAIsActive(other.valueMask().isOn(i))
            .setResultRef(result));
        mValueMask.set(i, args.resultIsActive());
        mBuffer.mData.set(i, result);
    }
}


template<Index Log2Dim>
template<typename CombineOp, typename OtherNodeT>
inline void
LeafNode<bool, Log2Dim>::combine2(bool value, const OtherNodeT& other,
    bool valueIsActive, CombineOp& op)
{
    CombineArgs<bool, typename OtherNodeT::ValueType> args;
    args.setARef(value).setAIsActive(valueIsActive);
    for (Index i = 0; i < SIZE; ++i) {
        bool result = false, bVal = other.mBuffer.mData.isOn(i);
        op(args.setBRef(bVal)
            .setBIsActive(other.valueMask().isOn(i))
            .setResultRef(result));
        mValueMask.set(i, args.resultIsActive());
        mBuffer.mData.set(i, result);
    }
}


template<Index Log2Dim>
template<typename CombineOp, typename OtherNodeT>
inline void
LeafNode<bool, Log2Dim>::combine2(const LeafNode& b0, const OtherNodeT& b1, CombineOp& op)
{
    CombineArgs<bool, typename OtherNodeT::ValueType> args;
    for (Index i = 0; i < SIZE; ++i) {
        // Default behavior: output voxel is active if either input voxel is active.
        mValueMask.set(i, b0.valueMask().isOn(i) || b1.valueMask().isOn(i));

        bool result = false, b0Val = b0.mBuffer.mData.isOn(i), b1Val = b1.mBuffer.mData.isOn(i);
        op(args.setARef(b0Val)
            .setAIsActive(b0.valueMask().isOn(i))
            .setBRef(b1Val)
            .setBIsActive(b1.valueMask().isOn(i))
            .setResultRef(result));
        mValueMask.set(i, args.resultIsActive());
        mBuffer.mData.set(i, result);
    }
}


////////////////////////////////////////

template<Index Log2Dim>
template<typename BBoxOp>
inline void
LeafNode<bool, Log2Dim>::visitActiveBBox(BBoxOp& op) const
{
    if (op.template descent<LEVEL>()) {
        for (ValueOnCIter i=this->cbeginValueOn(); i; ++i) {
#ifdef _MSC_VER
            op.operator()<LEVEL>(CoordBBox::createCube(i.getCoord(), 1));
#else
            op.template operator()<LEVEL>(CoordBBox::createCube(i.getCoord(), 1));
#endif
        }
    } else {
#ifdef _MSC_VER
        op.operator()<LEVEL>(this->getNodeBoundingBox());
#else
        op.template operator()<LEVEL>(this->getNodeBoundingBox());
#endif
    }
}


template<Index Log2Dim>
template<typename VisitorOp>
inline void
LeafNode<bool, Log2Dim>::visit(VisitorOp& op)
{
    doVisit<LeafNode, VisitorOp, ChildAllIter>(*this, op);
}


template<Index Log2Dim>
template<typename VisitorOp>
inline void
LeafNode<bool, Log2Dim>::visit(VisitorOp& op) const
{
    doVisit<const LeafNode, VisitorOp, ChildAllCIter>(*this, op);
}


template<Index Log2Dim>
template<typename NodeT, typename VisitorOp, typename ChildAllIterT>
inline void
LeafNode<bool, Log2Dim>::doVisit(NodeT& self, VisitorOp& op)
{
    for (ChildAllIterT iter = self.beginChildAll(); iter; ++iter) {
        op(iter);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
template<typename OtherLeafNodeType, typename VisitorOp>
inline void
LeafNode<bool, Log2Dim>::visit2Node(OtherLeafNodeType& other, VisitorOp& op)
{
    doVisit2Node<LeafNode, OtherLeafNodeType, VisitorOp, ChildAllIter,
        typename OtherLeafNodeType::ChildAllIter>(*this, other, op);
}


template<Index Log2Dim>
template<typename OtherLeafNodeType, typename VisitorOp>
inline void
LeafNode<bool, Log2Dim>::visit2Node(OtherLeafNodeType& other, VisitorOp& op) const
{
    doVisit2Node<const LeafNode, OtherLeafNodeType, VisitorOp, ChildAllCIter,
        typename OtherLeafNodeType::ChildAllCIter>(*this, other, op);
}


template<Index Log2Dim>
template<
    typename NodeT,
    typename OtherNodeT,
    typename VisitorOp,
    typename ChildAllIterT,
    typename OtherChildAllIterT>
inline void
LeafNode<bool, Log2Dim>::doVisit2Node(NodeT& self, OtherNodeT& other, VisitorOp& op)
{
    // Allow the two nodes to have different ValueTypes, but not different dimensions.
    static_assert(OtherNodeT::SIZE == NodeT::SIZE,
        "can't visit nodes of different sizes simultaneously");
    static_assert(OtherNodeT::LEVEL == NodeT::LEVEL,
        "can't visit nodes at different tree levels simultaneously");

    ChildAllIterT iter = self.beginChildAll();
    OtherChildAllIterT otherIter = other.beginChildAll();

    for ( ; iter && otherIter; ++iter, ++otherIter) {
        op(iter, otherIter);
    }
}


////////////////////////////////////////


template<Index Log2Dim>
template<typename IterT, typename VisitorOp>
inline void
LeafNode<bool, Log2Dim>::visit2(IterT& otherIter, VisitorOp& op, bool otherIsLHS)
{
    doVisit2<LeafNode, VisitorOp, ChildAllIter, IterT>(*this, otherIter, op, otherIsLHS);
}


template<Index Log2Dim>
template<typename IterT, typename VisitorOp>
inline void
LeafNode<bool, Log2Dim>::visit2(IterT& otherIter, VisitorOp& op, bool otherIsLHS) const
{
    doVisit2<const LeafNode, VisitorOp, ChildAllCIter, IterT>(*this, otherIter, op, otherIsLHS);
}


template<Index Log2Dim>
template<
    typename NodeT,
    typename VisitorOp,
    typename ChildAllIterT,
    typename OtherChildAllIterT>
inline void
LeafNode<bool, Log2Dim>::doVisit2(NodeT& self, OtherChildAllIterT& otherIter,
    VisitorOp& op, bool otherIsLHS)
{
    if (!otherIter) return;

    if (otherIsLHS) {
        for (ChildAllIterT iter = self.beginChildAll(); iter; ++iter) {
            op(otherIter, iter);
        }
    } else {
        for (ChildAllIterT iter = self.beginChildAll(); iter; ++iter) {
            op(iter, otherIter);
        }
    }
}

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_LEAF_NODE_BOOL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
