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

/// @file tree/Tree.h

#ifndef OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Metadata.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/BBox.h>
#include <openvdb/util/Formats.h>
#include <openvdb/util/logging.h>
#include <openvdb/Platform.h>
#include "RootNode.h"
#include "InternalNode.h"
#include "LeafNode.h"
#include "TreeIterator.h"
#include "ValueAccessor.h"
#include <tbb/atomic.h>
#include <tbb/concurrent_hash_map.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @brief Base class for typed trees
class OPENVDB_API TreeBase
{
public:
    using Ptr = SharedPtr<TreeBase>;
    using ConstPtr = SharedPtr<const TreeBase>;

    TreeBase() = default;
    TreeBase(const TreeBase&) = default;
    TreeBase& operator=(const TreeBase&) = delete; // disallow assignment
    virtual ~TreeBase() = default;

    /// Return the name of this tree's type.
    virtual const Name& type() const = 0;

    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d").
    virtual Name valueType() const = 0;

    /// Return a pointer to a deep copy of this tree
    virtual TreeBase::Ptr copy() const = 0;

    //
    // Tree methods
    //
    /// @brief Return this tree's background value wrapped as metadata.
    /// @note Query the metadata object for the value's type.
    virtual Metadata::Ptr getBackgroundValue() const { return Metadata::Ptr(); }

    /// @brief Return in @a bbox the axis-aligned bounding box of all
    /// leaf nodes and active tiles.
    /// @details This is faster than calling evalActiveVoxelBoundingBox,
    /// which visits the individual active voxels, and hence
    /// evalLeafBoundingBox produces a less tight, i.e. approximate, bbox.
    /// @return @c false if the bounding box is empty (in which case
    /// the bbox is set to its default value).
    virtual bool evalLeafBoundingBox(CoordBBox& bbox) const = 0;

    /// @brief Return in @a dim the dimensions of the axis-aligned bounding box
    /// of all leaf nodes.
    /// @return @c false if the bounding box is empty.
    virtual bool evalLeafDim(Coord& dim) const = 0;

    /// @brief Return in @a bbox the axis-aligned bounding box of all
    /// active voxels and tiles.
    /// @details This method produces a more accurate, i.e. tighter,
    /// bounding box than evalLeafBoundingBox which is approximate but
    /// faster.
    /// @return @c false if the bounding box is empty (in which case
    /// the bbox is set to its default value).
    virtual bool evalActiveVoxelBoundingBox(CoordBBox& bbox) const = 0;

    /// @brief Return in @a dim the dimensions of the axis-aligned bounding box of all
    /// active voxels.  This is a tighter bounding box than the leaf node bounding box.
    /// @return @c false if the bounding box is empty.
    virtual bool evalActiveVoxelDim(Coord& dim) const = 0;

    virtual void getIndexRange(CoordBBox& bbox) const = 0;

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// @brief Replace with background tiles any nodes whose voxel buffers
    /// have not yet been allocated.
    /// @details Typically, unallocated nodes are leaf nodes whose voxel buffers
    /// are not yet resident in memory because delayed loading is in effect.
    /// @sa readNonresidentBuffers, io::File::open
    virtual void clipUnallocatedNodes() = 0;
#endif
#if OPENVDB_ABI_VERSION_NUMBER >= 4
    /// Return the total number of unallocated leaf nodes residing in this tree.
    virtual Index32 unallocatedLeafCount() const = 0;
#endif


    //
    // Statistics
    //
    /// @brief Return the depth of this tree.
    ///
    /// A tree with only a root node and leaf nodes has depth 2, for example.
    virtual Index treeDepth() const = 0;
    /// Return the number of leaf nodes.
    virtual Index32 leafCount() const = 0;
    /// Return the number of non-leaf nodes.
    virtual Index32 nonLeafCount() const = 0;
    /// Return the number of active voxels stored in leaf nodes.
    virtual Index64 activeLeafVoxelCount() const = 0;
    /// Return the number of inactive voxels stored in leaf nodes.
    virtual Index64 inactiveLeafVoxelCount() const = 0;
    /// Return the total number of active voxels.
    virtual Index64 activeVoxelCount() const = 0;
    /// Return the number of inactive voxels within the bounding box of all active voxels.
    virtual Index64 inactiveVoxelCount() const = 0;
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// Return the total number of active tiles.
    virtual Index64 activeTileCount() const = 0;
#endif

    /// Return the total amount of memory in bytes occupied by this tree.
    virtual Index64 memUsage() const { return 0; }


    //
    // I/O methods
    //
    /// @brief Read the tree topology from a stream.
    ///
    /// This will read the tree structure and tile values, but not voxel data.
    virtual void readTopology(std::istream&, bool saveFloatAsHalf = false);
    /// @brief Write the tree topology to a stream.
    ///
    /// This will write the tree structure and tile values, but not voxel data.
    virtual void writeTopology(std::ostream&, bool saveFloatAsHalf = false) const;

    /// Read all data buffers for this tree.
    virtual void readBuffers(std::istream&, bool saveFloatAsHalf = false) = 0;
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// Read all of this tree's data buffers that intersect the given bounding box.
    virtual void readBuffers(std::istream&, const CoordBBox&, bool saveFloatAsHalf = false) = 0;
    /// @brief Read all of this tree's data buffers that are not yet resident in memory
    /// (because delayed loading is in effect).
    /// @details If this tree was read from a memory-mapped file, this operation
    /// disconnects the tree from the file.
    /// @sa clipUnallocatedNodes, io::File::open, io::MappedFile
    virtual void readNonresidentBuffers() const = 0;
#endif
    /// Write out all the data buffers for this tree.
    virtual void writeBuffers(std::ostream&, bool saveFloatAsHalf = false) const = 0;

    /// @brief Print statistics, memory usage and other information about this tree.
    /// @param os            a stream to which to write textual information
    /// @param verboseLevel  1: print tree configuration only;
    ///                      2: include node and voxel statistics;
    ///                      3: include memory usage;
    ///                      4: include minimum and maximum voxel values
    /// @warning @a verboseLevel 4 forces loading of any unallocated nodes.
    virtual void print(std::ostream& os = std::cout, int verboseLevel = 1) const;
};


////////////////////////////////////////


template<typename _RootNodeType>
class Tree: public TreeBase
{
public:
    using Ptr = SharedPtr<Tree>;
    using ConstPtr = SharedPtr<const Tree>;

    using RootNodeType = _RootNodeType;
    using ValueType = typename RootNodeType::ValueType;
    using BuildType = typename RootNodeType::BuildType;
    using LeafNodeType = typename RootNodeType::LeafNodeType;

    static const Index DEPTH = RootNodeType::LEVEL + 1;

    /// @brief ValueConverter<T>::Type is the type of a tree having the same
    /// hierarchy as this tree but a different value type, T.
    ///
    /// For example, FloatTree::ValueConverter<double>::Type is equivalent to DoubleTree.
    /// @note If the source tree type is a template argument, it might be necessary
    /// to write "typename SourceTree::template ValueConverter<T>::Type".
    template<typename OtherValueType>
    struct ValueConverter {
        using Type = Tree<typename RootNodeType::template ValueConverter<OtherValueType>::Type>;
    };


    Tree() {}

    Tree& operator=(const Tree&) = delete; // disallow assignment

    /// Deep copy constructor
    Tree(const Tree& other): TreeBase(other), mRoot(other.mRoot)
    {
    }

    /// @brief Value conversion deep copy constructor
    ///
    /// Deep copy a tree of the same configuration as this tree type but a different
    /// ValueType, casting the other tree's values to this tree's ValueType.
    /// @throw TypeError if the other tree's configuration doesn't match this tree's
    /// or if this tree's ValueType is not constructible from the other tree's ValueType.
    template<typename OtherRootType>
    explicit Tree(const Tree<OtherRootType>& other): TreeBase(other), mRoot(other.root())
    {
    }

    /// @brief Topology copy constructor from a tree of a different type
    ///
    /// Copy the structure, i.e., the active states of tiles and voxels, of another
    /// tree of a possibly different type, but don't copy any tile or voxel values.
    /// Instead, initialize tiles and voxels with the given active and inactive values.
    /// @param other          a tree having (possibly) a different ValueType
    /// @param inactiveValue  background value for this tree, and the value to which
    ///                       all inactive tiles and voxels are initialized
    /// @param activeValue    value to which active tiles and voxels are initialized
    /// @throw TypeError if the other tree's configuration doesn't match this tree's.
    template<typename OtherTreeType>
    Tree(const OtherTreeType& other,
        const ValueType& inactiveValue,
        const ValueType& activeValue,
        TopologyCopy):
        TreeBase(other),
        mRoot(other.root(), inactiveValue, activeValue, TopologyCopy())
    {
    }

    /// @brief Topology copy constructor from a tree of a different type
    ///
    /// @note This topology copy constructor is generally faster than
    /// the one that takes both a foreground and a background value.
    ///
    /// Copy the structure, i.e., the active states of tiles and voxels, of another
    /// tree of a possibly different type, but don't copy any tile or voxel values.
    /// Instead, initialize tiles and voxels with the given background value.
    /// @param other        a tree having (possibly) a different ValueType
    /// @param background   the value to which tiles and voxels are initialized
    /// @throw TypeError if the other tree's configuration doesn't match this tree's.
    template<typename OtherTreeType>
    Tree(const OtherTreeType& other, const ValueType& background, TopologyCopy):
        TreeBase(other),
        mRoot(other.root(), background, TopologyCopy())
    {
    }

    /// Empty tree constructor
    Tree(const ValueType& background): mRoot(background) {}

    ~Tree() override { this->clear(); releaseAllAccessors(); }

    /// Return a pointer to a deep copy of this tree
    TreeBase::Ptr copy() const override { return TreeBase::Ptr(new Tree(*this)); }

    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d")
    Name valueType() const override { return typeNameAsString<ValueType>(); }

    /// Return the name of this type of tree.
    static const Name& treeType();
    /// Return the name of this type of tree.
    const Name& type() const override { return this->treeType(); }

    bool operator==(const Tree&) const { OPENVDB_THROW(NotImplementedError, ""); }
    bool operator!=(const Tree&) const { OPENVDB_THROW(NotImplementedError, ""); }

    //@{
    /// Return this tree's root node.
    RootNodeType& root() { return mRoot; }
    const RootNodeType& root() const { return mRoot; }
    //@}


    //
    // Tree methods
    //
    /// @brief Return @c true if the given tree has the same node and active value
    /// topology as this tree, whether or not it has the same @c ValueType.
    template<typename OtherRootNodeType>
    bool hasSameTopology(const Tree<OtherRootNodeType>& other) const;

    bool evalLeafBoundingBox(CoordBBox& bbox) const override;
    bool evalActiveVoxelBoundingBox(CoordBBox& bbox) const override;
    bool evalActiveVoxelDim(Coord& dim) const override;
    bool evalLeafDim(Coord& dim) const override;

    /// @brief Traverse the type hierarchy of nodes, and return, in @a dims, a list
    /// of the Log2Dims of nodes in order from RootNode to LeafNode.
    /// @note Because RootNodes are resizable, the RootNode Log2Dim is 0 for all trees.
    static void getNodeLog2Dims(std::vector<Index>& dims);


    //
    // I/O methods
    //
    /// @brief Read the tree topology from a stream.
    ///
    /// This will read the tree structure and tile values, but not voxel data.
    void readTopology(std::istream&, bool saveFloatAsHalf = false) override;
    /// @brief Write the tree topology to a stream.
    ///
    /// This will write the tree structure and tile values, but not voxel data.
    void writeTopology(std::ostream&, bool saveFloatAsHalf = false) const override;
    /// Read all data buffers for this tree.
    void readBuffers(std::istream&, bool saveFloatAsHalf = false) override;
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// Read all of this tree's data buffers that intersect the given bounding box.
    void readBuffers(std::istream&, const CoordBBox&, bool saveFloatAsHalf = false) override;
    /// @brief Read all of this tree's data buffers that are not yet resident in memory
    /// (because delayed loading is in effect).
    /// @details If this tree was read from a memory-mapped file, this operation
    /// disconnects the tree from the file.
    /// @sa clipUnallocatedNodes, io::File::open, io::MappedFile
    void readNonresidentBuffers() const override;
#endif
    /// Write out all data buffers for this tree.
    void writeBuffers(std::ostream&, bool saveFloatAsHalf = false) const override;

    void print(std::ostream& os = std::cout, int verboseLevel = 1) const override;


    //
    // Statistics
    //
    /// @brief Return the depth of this tree.
    ///
    /// A tree with only a root node and leaf nodes has depth 2, for example.
    Index treeDepth() const override { return DEPTH; }
    /// Return the number of leaf nodes.
    Index32 leafCount() const override { return mRoot.leafCount(); }
    /// Return the number of non-leaf nodes.
    Index32 nonLeafCount() const override { return mRoot.nonLeafCount(); }
    /// Return the number of active voxels stored in leaf nodes.
    Index64 activeLeafVoxelCount() const override { return mRoot.onLeafVoxelCount(); }
    /// Return the number of inactive voxels stored in leaf nodes.
    Index64 inactiveLeafVoxelCount() const override { return mRoot.offLeafVoxelCount(); }
    /// Return the total number of active voxels.
    Index64 activeVoxelCount() const override { return mRoot.onVoxelCount(); }
    /// Return the number of inactive voxels within the bounding box of all active voxels.
    Index64 inactiveVoxelCount() const override;
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// Return the total number of active tiles.
    Index64 activeTileCount() const override { return mRoot.onTileCount(); }
#else
    Index64 activeTileCount() const { return mRoot.onTileCount(); }
#endif

    /// Return the minimum and maximum active values in this tree.
    void evalMinMax(ValueType &min, ValueType &max) const;

    Index64 memUsage() const override { return sizeof(*this) + mRoot.memUsage(); }


    //
    // Voxel access methods (using signed indexing)
    //
    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const;
    /// @brief Return the value of the voxel at the given coordinates
    /// and update the given accessor's node cache.
    template<typename AccessT> const ValueType& getValue(const Coord& xyz, AccessT&) const;

    /// @brief Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides.
    /// @details If (x, y, z) isn't explicitly represented in the tree (i.e., it is
    /// implicitly a background voxel), return -1.
    int getValueDepth(const Coord& xyz) const;

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on);
    /// Set the value of the voxel at the given coordinates but don't change its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value);
    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz);
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValueOn(const Coord& xyz, const ValueType& value);
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value);
    /// @brief Set the value of the voxel at the given coordinates, mark the voxel as active,
    /// and update the given accessor's node cache.
    template<typename AccessT> void setValue(const Coord& xyz, const ValueType& value, AccessT&);
    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz);
    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value);

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details Provided that the functor can be inlined, this is typically
    /// significantly faster than calling getValue() followed by setValueOn().
    /// @param xyz  the coordinates of a voxel whose value is to be modified
    /// @param op   a functor of the form <tt>void op(ValueType&) const</tt> that modifies
    ///             its argument in place
    /// @par Example:
    /// @code
    /// Coord xyz(1, 0, -2);
    /// // Multiply the value of a voxel by a constant and mark the voxel as active.
    /// floatTree.modifyValue(xyz, [](float& f) { f *= 0.25; }); // C++11
    /// // Set the value of a voxel to the maximum of its current value and 0.25,
    /// // and mark the voxel as active.
    /// floatTree.modifyValue(xyz, [](float& f) { f = std::max(f, 0.25f); }); // C++11
    /// @endcode
    /// @note The functor is not guaranteed to be called only once.
    /// @see tools::foreach()
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op);

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details Provided that the functor can be inlined, this is typically
    /// significantly faster than calling getValue() followed by setValue().
    /// @param xyz  the coordinates of a voxel to be modified
    /// @param op   a functor of the form <tt>void op(ValueType&, bool&) const</tt> that
    ///             modifies its arguments, a voxel's value and active state, in place
    /// @par Example:
    /// @code
    /// Coord xyz(1, 0, -2);
    /// // Multiply the value of a voxel by a constant and mark the voxel as inactive.
    /// floatTree.modifyValueAndActiveState(xyz,
    ///     [](float& f, bool& b) { f *= 0.25; b = false; }); // C++11
    /// // Set the value of a voxel to the maximum of its current value and 0.25,
    /// // but don't change the voxel's active state.
    /// floatTree.modifyValueAndActiveState(xyz,
    ///     [](float& f, bool&) { f = std::max(f, 0.25f); }); // C++11
    /// @endcode
    /// @note The functor is not guaranteed to be called only once.
    /// @see tools::foreach()
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op);

    /// @brief Get the value of the voxel at the given coordinates.
    /// @return @c true if the value is active.
    bool probeValue(const Coord& xyz, ValueType& value) const;

    /// Return @c true if the value at the given coordinates is active.
    bool isValueOn(const Coord& xyz) const { return mRoot.isValueOn(xyz); }
    /// Return @c true if the value at the given coordinates is inactive.
    bool isValueOff(const Coord& xyz) const { return !this->isValueOn(xyz); }
    /// Return @c true if this tree has any active tiles.
    bool hasActiveTiles() const { return mRoot.hasActiveTiles(); }

    /// Set all voxels that lie outside the given axis-aligned box to the background.
    void clip(const CoordBBox&);

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// @brief Replace with background tiles any nodes whose voxel buffers
    /// have not yet been allocated.
    /// @details Typically, unallocated nodes are leaf nodes whose voxel buffers
    /// are not yet resident in memory because delayed loading is in effect.
    /// @sa readNonresidentBuffers, io::File::open
    void clipUnallocatedNodes() override;
#endif
#if OPENVDB_ABI_VERSION_NUMBER >= 4
    /// Return the total number of unallocated leaf nodes residing in this tree.
    Index32 unallocatedLeafCount() const override;
#endif

    //@{
    /// @brief Set all voxels within a given axis-aligned box to a constant value.
    /// @param bbox    inclusive coordinates of opposite corners of an axis-aligned box
    /// @param value   the value to which to set voxels within the box
    /// @param active  if true, mark voxels within the box as active,
    ///                otherwise mark them as inactive
    /// @note This operation generates a sparse, but not always optimally sparse,
    /// representation of the filled box. Follow fill operations with a prune()
    /// operation for optimal sparseness.
    void sparseFill(const CoordBBox& bbox, const ValueType& value, bool active = true);
    void fill(const CoordBBox& bbox, const ValueType& value, bool active = true)
    {
        this->sparseFill(bbox, value, active);
    }
    //@}

    /// @brief Set all voxels within a given axis-aligned box to a constant value
    /// and ensure that those voxels are all represented at the leaf level.
    /// @param bbox    inclusive coordinates of opposite corners of an axis-aligned box.
    /// @param value   the value to which to set voxels within the box.
    /// @param active  if true, mark voxels within the box as active,
    ///                otherwise mark them as inactive.
    /// @sa voxelizeActiveTiles()
    void denseFill(const CoordBBox& bbox, const ValueType& value, bool active = true);

    /// @brief Densify active tiles, i.e., replace them with leaf-level active voxels.
    ///
    /// @param threaded if true, this operation is multi-threaded (over the internal nodes).
    ///
    /// @warning This method can explode the tree's memory footprint, especially if it
    /// contains active tiles at the upper levels (in particular the root level)!
    ///
    /// @sa denseFill()
    void voxelizeActiveTiles(bool threaded = true);

    /// @brief Reduce the memory footprint of this tree by replacing with tiles
    /// any nodes whose values are all the same (optionally to within a tolerance)
    /// and have the same active state.
    /// @warning Will soon be deprecated!
    void prune(const ValueType& tolerance = zeroVal<ValueType>())
    {
        this->clearAllAccessors();
        mRoot.prune(tolerance);
    }

    /// @brief Add the given leaf node to this tree, creating a new branch if necessary.
    /// If a leaf node with the same origin already exists, replace it.
    ///
    /// @warning Ownership of the leaf is transferred to the tree so
    /// the client code should not attempt to delete the leaf pointer!
    void addLeaf(LeafNodeType* leaf) { assert(leaf); mRoot.addLeaf(leaf); }

    /// @brief Add a tile containing voxel (x, y, z) at the specified tree level,
    /// creating a new branch if necessary.  Delete any existing lower-level nodes
    /// that contain (x, y, z).
    /// @note @a level must be less than this tree's depth.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool active);

    /// @brief Return a pointer to the node of type @c NodeT that contains voxel (x, y, z)
    /// and replace it with a tile of the specified value and state.
    /// If no such node exists, leave the tree unchanged and return @c nullptr.
    /// @note The caller takes ownership of the node and is responsible for deleting it.
    template<typename NodeT>
    NodeT* stealNode(const Coord& xyz, const ValueType& value, bool active);

    /// @brief Return a pointer to the leaf node that contains voxel (x, y, z).
    /// If no such node exists, create one that preserves the values and
    /// active states of all voxels.
    /// @details Use this method to preallocate a static tree topology over which to
    /// safely perform multithreaded processing.
    LeafNodeType* touchLeaf(const Coord& xyz);

    //@{
    /// @brief Return a pointer to the node of type @c NodeType that contains
    /// voxel (x, y, z).  If no such node exists, return @c nullptr.
    template<typename NodeType> NodeType* probeNode(const Coord& xyz);
    template<typename NodeType> const NodeType* probeConstNode(const Coord& xyz) const;
    template<typename NodeType> const NodeType* probeNode(const Coord& xyz) const;
    //@}

    //@{
    /// @brief Return a pointer to the leaf node that contains voxel (x, y, z).
    /// If no such node exists, return @c nullptr.
    LeafNodeType* probeLeaf(const Coord& xyz);
    const LeafNodeType* probeConstLeaf(const Coord& xyz) const;
    const LeafNodeType* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }
    //@}

    //@{
    /// @brief Adds all nodes of a certain type to a container with the following API:
    /// @code
    /// struct ArrayT {
    ///    using value_type = ...;             // the type of node to be added to the array
    ///    void push_back(value_type nodePtr); // add a node to the array
    /// };
    /// @endcode
    /// @details An example of a wrapper around a c-style array is:
    /// @code
    /// struct MyArray {
    ///    using value_type = LeafType*;
    ///    value_type* ptr;
    ///    MyArray(value_type* array) : ptr(array) {}
    ///    void push_back(value_type leaf) { *ptr++ = leaf; }
    ///};
    /// @endcode
    /// @details An example that constructs a list of pointer to all leaf nodes is:
    /// @code
    /// std::vector<const LeafNodeType*> array;//most std contains have the required API
    /// array.reserve(tree.leafCount());//this is a fast preallocation.
    /// tree.getNodes(array);
    /// @endcode
    template<typename ArrayT> void getNodes(ArrayT& array) { mRoot.getNodes(array); }
    template<typename ArrayT> void getNodes(ArrayT& array) const { mRoot.getNodes(array); }
    //@}

    /// @brief Steals all nodes of a certain type from the tree and
    /// adds them to a container with the following API:
    /// @code
    /// struct ArrayT {
    ///    using value_type = ...;             // the type of node to be added to the array
    ///    void push_back(value_type nodePtr); // add a node to the array
    /// };
    /// @endcode
    /// @details An example of a wrapper around a c-style array is:
    /// @code
    /// struct MyArray {
    ///    using value_type = LeafType*;
    ///    value_type* ptr;
    ///    MyArray(value_type* array) : ptr(array) {}
    ///    void push_back(value_type leaf) { *ptr++ = leaf; }
    ///};
    /// @endcode
    /// @details An example that constructs a list of pointer to all leaf nodes is:
    /// @code
    /// std::vector<const LeafNodeType*> array;//most std contains have the required API
    /// array.reserve(tree.leafCount());//this is a fast preallocation.
    /// tree.stealNodes(array);
    /// @endcode
    template<typename ArrayT>
    void stealNodes(ArrayT& array) { this->clearAllAccessors(); mRoot.stealNodes(array); }
    template<typename ArrayT>
    void stealNodes(ArrayT& array, const ValueType& value, bool state)
    {
        this->clearAllAccessors();
        mRoot.stealNodes(array, value, state);
    }

    //
    // Aux methods
    //
    /// @brief Return @c true if this tree contains no nodes other than
    /// the root node and no tiles other than background tiles.
    bool empty() const { return mRoot.empty(); }

    /// Remove all tiles from this tree and all nodes other than the root node.
    void clear();

    /// Clear all registered accessors.
    void clearAllAccessors();

    //@{
    /// @brief Register an accessor for this tree.  Registered accessors are
    /// automatically cleared whenever one of this tree's nodes is deleted.
    void attachAccessor(ValueAccessorBase<Tree, true>&) const;
    void attachAccessor(ValueAccessorBase<const Tree, true>&) const;
    //@}

    //@{
    /// Dummy implementations
    void attachAccessor(ValueAccessorBase<Tree, false>&) const {}
    void attachAccessor(ValueAccessorBase<const Tree, false>&) const {}
    //@}

    //@{
    /// Deregister an accessor so that it is no longer automatically cleared.
    void releaseAccessor(ValueAccessorBase<Tree, true>&) const;
    void releaseAccessor(ValueAccessorBase<const Tree, true>&) const;
    //@}

    //@{
    /// Dummy implementations
    void releaseAccessor(ValueAccessorBase<Tree, false>&) const {}
    void releaseAccessor(ValueAccessorBase<const Tree, false>&) const {}
    //@}

    /// @brief Return this tree's background value wrapped as metadata.
    /// @note Query the metadata object for the value's type.
    Metadata::Ptr getBackgroundValue() const override;

    /// @brief Return this tree's background value.
    ///
    /// @note Use tools::changeBackground to efficiently modify the
    /// background values. Else use tree.root().setBackground, which
    /// is serial and hence slower.
    const ValueType& background() const { return mRoot.background(); }

    /// Min and max are both inclusive.
    void getIndexRange(CoordBBox& bbox) const override { mRoot.getIndexRange(bbox); }

    /// @brief Efficiently merge another tree into this tree using one of several schemes.
    /// @details This operation is primarily intended to combine trees that are mostly
    /// non-overlapping (for example, intermediate trees from computations that are
    /// parallelized across disjoint regions of space).
    /// @note This operation is not guaranteed to produce an optimally sparse tree.
    /// Follow merge() with prune() for optimal sparseness.
    /// @warning This operation always empties the other tree.
    void merge(Tree& other, MergePolicy = MERGE_ACTIVE_STATES);

    /// @brief Union this tree's set of active values with the active values
    /// of the other tree, whose @c ValueType may be different.
    /// @details The resulting state of a value is active if the corresponding value
    /// was already active OR if it is active in the other tree.  Also, a resulting
    /// value maps to a voxel if the corresponding value already mapped to a voxel
    /// OR if it is a voxel in the other tree.  Thus, a resulting value can only
    /// map to a tile if the corresponding value already mapped to a tile
    /// AND if it is a tile value in other tree.
    ///
    /// @note This operation modifies only active states, not values.
    /// Specifically, active tiles and voxels in this tree are not changed, and
    /// tiles or voxels that were inactive in this tree but active in the other tree
    /// are marked as active in this tree but left with their original values.
    template<typename OtherRootNodeType>
    void topologyUnion(const Tree<OtherRootNodeType>& other);

    /// @brief Intersects this tree's set of active values with the active values
    /// of the other tree, whose @c ValueType may be different.
    /// @details The resulting state of a value is active only if the corresponding
    /// value was already active AND if it is active in the other tree. Also, a
    /// resulting value maps to a voxel if the corresponding value
    /// already mapped to an active voxel in either of the two grids
    /// and it maps to an active tile or voxel in the other grid.
    ///
    /// @note This operation can delete branches in this grid if they
    /// overlap with inactive tiles in the other grid. Likewise active
    /// voxels can be turned into unactive voxels resulting in leaf
    /// nodes with no active values. Thus, it is recommended to
    /// subsequently call tools::pruneInactive.
    template<typename OtherRootNodeType>
    void topologyIntersection(const Tree<OtherRootNodeType>& other);

    /// @brief Difference this tree's set of active values with the active values
    /// of the other tree, whose @c ValueType may be different. So a
    /// resulting voxel will be active only if the original voxel is
    /// active in this tree and inactive in the other tree.
    ///
    /// @note This operation can delete branches in this grid if they
    /// overlap with active tiles in the other grid. Likewise active
    /// voxels can be turned into inactive voxels resulting in leaf
    /// nodes with no active values. Thus, it is recommended to
    /// subsequently call tools::pruneInactive.
    template<typename OtherRootNodeType>
    void topologyDifference(const Tree<OtherRootNodeType>& other);

    /// For a given function @c f, use sparse traversal to compute <tt>f(this, other)</tt>
    /// over all corresponding pairs of values (tile or voxel) of this tree and the other tree
    /// and store the result in this tree.
    /// This method is typically more space-efficient than the two-tree combine2(),
    /// since it moves rather than copies nodes from the other tree into this tree.
    /// @note This operation always empties the other tree.
    /// @param other  a tree of the same type as this tree
    /// @param op     a functor of the form <tt>void op(const T& a, const T& b, T& result)</tt>,
    ///               where @c T is this tree's @c ValueType, that computes
    ///               <tt>result = f(a, b)</tt>
    /// @param prune  if true, prune the resulting tree one branch at a time (this is usually
    ///               more space-efficient than pruning the entire tree in one pass)
    ///
    /// @par Example:
    ///     Compute the per-voxel difference between two floating-point trees,
    ///     @c aTree and @c bTree, and store the result in @c aTree (leaving @c bTree empty).
    /// @code
    /// {
    ///     struct Local {
    ///         static inline void diff(const float& a, const float& b, float& result) {
    ///             result = a - b;
    ///         }
    ///     };
    ///     aTree.combine(bTree, Local::diff);
    /// }
    /// @endcode
    ///
    /// @par Example:
    ///     Compute <tt>f * a + (1 - f) * b</tt> over all voxels of two floating-point trees,
    ///     @c aTree and @c bTree, and store the result in @c aTree (leaving @c bTree empty).
    /// @code
    /// namespace {
    ///     struct Blend {
    ///         Blend(float f): frac(f) {}
    ///         inline void operator()(const float& a, const float& b, float& result) const {
    ///             result = frac * a + (1.0 - frac) * b;
    ///         }
    ///         float frac;
    ///     };
    /// }
    /// {
    ///     aTree.combine(bTree, Blend(0.25)); // 0.25 * a + 0.75 * b
    /// }
    /// @endcode
    template<typename CombineOp>
    void combine(Tree& other, CombineOp& op, bool prune = false);
#ifndef _MSC_VER
    template<typename CombineOp>
    void combine(Tree& other, const CombineOp& op, bool prune = false);
#endif

    /// Like combine(), but with
    /// @param other  a tree of the same type as this tree
    /// @param op     a functor of the form <tt>void op(CombineArgs<ValueType>& args)</tt> that
    ///               computes <tt>args.setResult(f(args.a(), args.b()))</tt> and, optionally,
    ///               <tt>args.setResultIsActive(g(args.aIsActive(), args.bIsActive()))</tt>
    ///               for some functions @c f and @c g
    /// @param prune  if true, prune the resulting tree one branch at a time (this is usually
    ///               more space-efficient than pruning the entire tree in one pass)
    ///
    /// This variant passes not only the @em a and @em b values but also the active states
    /// of the @em a and @em b values to the functor, which may then return, by calling
    /// @c args.setResultIsActive(), a computed active state for the result value.
    /// By default, the result is active if either the @em a or the @em b value is active.
    ///
    /// @see openvdb/Types.h for the definition of the CombineArgs struct.
    ///
    /// @par Example:
    ///     Replace voxel values in floating-point @c aTree with corresponding values
    ///     from floating-point @c bTree (leaving @c bTree empty) wherever the @c bTree
    ///     values are larger.  Also, preserve the active states of any transferred values.
    /// @code
    /// {
    ///     struct Local {
    ///         static inline void max(CombineArgs<float>& args) {
    ///             if (args.b() > args.a()) {
    ///                 // Transfer the B value and its active state.
    ///                 args.setResult(args.b());
    ///                 args.setResultIsActive(args.bIsActive());
    ///             } else {
    ///                 // Preserve the A value and its active state.
    ///                 args.setResult(args.a());
    ///                 args.setResultIsActive(args.aIsActive());
    ///             }
    ///         }
    ///     };
    ///     aTree.combineExtended(bTree, Local::max);
    /// }
    /// @endcode
    template<typename ExtendedCombineOp>
    void combineExtended(Tree& other, ExtendedCombineOp& op, bool prune = false);
#ifndef _MSC_VER
    template<typename ExtendedCombineOp>
    void combineExtended(Tree& other, const ExtendedCombineOp& op, bool prune = false);
#endif

    /// For a given function @c f, use sparse traversal to compute <tt>f(a, b)</tt> over all
    /// corresponding pairs of values (tile or voxel) of trees A and B and store the result
    /// in this tree.
    /// @param a,b    two trees with the same configuration (levels and node dimensions)
    ///               as this tree but with the B tree possibly having a different value type
    /// @param op     a functor of the form <tt>void op(const T1& a, const T2& b, T1& result)</tt>,
    ///               where @c T1 is this tree's and the A tree's @c ValueType and @c T2 is the
    ///               B tree's @c ValueType, that computes <tt>result = f(a, b)</tt>
    /// @param prune  if true, prune the resulting tree one branch at a time (this is usually
    ///               more space-efficient than pruning the entire tree in one pass)
    ///
    /// @throw TypeError if the B tree's configuration doesn't match this tree's
    /// or if this tree's ValueType is not constructible from the B tree's ValueType.
    ///
    /// @par Example:
    ///     Compute the per-voxel difference between two floating-point trees,
    ///     @c aTree and @c bTree, and store the result in a third tree.
    /// @code
    /// {
    ///     struct Local {
    ///         static inline void diff(const float& a, const float& b, float& result) {
    ///             result = a - b;
    ///         }
    ///     };
    ///     FloatTree resultTree;
    ///     resultTree.combine2(aTree, bTree, Local::diff);
    /// }
    /// @endcode
    template<typename CombineOp, typename OtherTreeType /*= Tree*/>
    void combine2(const Tree& a, const OtherTreeType& b, CombineOp& op, bool prune = false);
#ifndef _MSC_VER
    template<typename CombineOp, typename OtherTreeType /*= Tree*/>
    void combine2(const Tree& a, const OtherTreeType& b, const CombineOp& op, bool prune = false);
#endif

    /// Like combine2(), but with
    /// @param a,b    two trees with the same configuration (levels and node dimensions)
    ///               as this tree but with the B tree possibly having a different value type
    /// @param op     a functor of the form <tt>void op(CombineArgs<T1, T2>& args)</tt>, where
    ///               @c T1 is this tree's and the A tree's @c ValueType and @c T2 is the B tree's
    ///               @c ValueType, that computes <tt>args.setResult(f(args.a(), args.b()))</tt>
    ///               and, optionally,
    ///               <tt>args.setResultIsActive(g(args.aIsActive(), args.bIsActive()))</tt>
    ///               for some functions @c f and @c g
    /// @param prune  if true, prune the resulting tree one branch at a time (this is usually
    ///               more space-efficient than pruning the entire tree in one pass)
    /// This variant passes not only the @em a and @em b values but also the active states
    /// of the @em a and @em b values to the functor, which may then return, by calling
    /// <tt>args.setResultIsActive()</tt>, a computed active state for the result value.
    /// By default, the result is active if either the @em a or the @em b value is active.
    ///
    /// @throw TypeError if the B tree's configuration doesn't match this tree's
    /// or if this tree's ValueType is not constructible from the B tree's ValueType.
    ///
    /// @see openvdb/Types.h for the definition of the CombineArgs struct.
    ///
    /// @par Example:
    ///     Compute the per-voxel maximum values of two single-precision floating-point trees,
    ///     @c aTree and @c bTree, and store the result in a third tree.  Set the active state
    ///     of each output value to that of the larger of the two input values.
    /// @code
    /// {
    ///     struct Local {
    ///         static inline void max(CombineArgs<float>& args) {
    ///             if (args.b() > args.a()) {
    ///                 // Transfer the B value and its active state.
    ///                 args.setResult(args.b());
    ///                 args.setResultIsActive(args.bIsActive());
    ///             } else {
    ///                 // Preserve the A value and its active state.
    ///                 args.setResult(args.a());
    ///                 args.setResultIsActive(args.aIsActive());
    ///             }
    ///         }
    ///     };
    ///     FloatTree aTree = ...;
    ///     FloatTree bTree = ...;
    ///     FloatTree resultTree;
    ///     resultTree.combine2Extended(aTree, bTree, Local::max);
    /// }
    /// @endcode
    ///
    /// @par Example:
    ///     Compute the per-voxel maximum values of a double-precision and a single-precision
    ///     floating-point tree, @c aTree and @c bTree, and store the result in a third,
    ///     double-precision tree.  Set the active state of each output value to that of
    ///     the larger of the two input values.
    /// @code
    /// {
    ///     struct Local {
    ///         static inline void max(CombineArgs<double, float>& args) {
    ///             if (args.b() > args.a()) {
    ///                 // Transfer the B value and its active state.
    ///                 args.setResult(args.b());
    ///                 args.setResultIsActive(args.bIsActive());
    ///             } else {
    ///                 // Preserve the A value and its active state.
    ///                 args.setResult(args.a());
    ///                 args.setResultIsActive(args.aIsActive());
    ///             }
    ///         }
    ///     };
    ///     DoubleTree aTree = ...;
    ///     FloatTree bTree = ...;
    ///     DoubleTree resultTree;
    ///     resultTree.combine2Extended(aTree, bTree, Local::max);
    /// }
    /// @endcode
    template<typename ExtendedCombineOp, typename OtherTreeType /*= Tree*/>
    void combine2Extended(const Tree& a, const OtherTreeType& b, ExtendedCombineOp& op,
        bool prune = false);
#ifndef _MSC_VER
    template<typename ExtendedCombineOp, typename OtherTreeType /*= Tree*/>
    void combine2Extended(const Tree& a, const OtherTreeType& b, const ExtendedCombineOp&,
        bool prune = false);
#endif

    /// @brief Use sparse traversal to call the given functor with bounding box
    /// information for all active tiles and leaf nodes or active voxels in the tree.
    ///
    /// @note The bounding boxes are guaranteed to be non-overlapping.
    /// @param op  a functor with a templated call operator of the form
    ///     <tt>template<Index LEVEL> void operator()(const CoordBBox& bbox)</tt>,
    ///     where <tt>bbox</tt> is the bounding box of either an active tile
    ///     (if @c LEVEL > 0), a leaf node or an active voxel.
    ///     The functor must also provide a templated method of the form
    ///     <tt>template<Index LEVEL> bool descent()</tt> that returns @c false
    ///     if bounding boxes below the specified tree level are not to be visited.
    ///     In such cases of early tree termination, a bounding box is instead
    ///     derived from each terminating child node.
    ///
    /// @par Example:
    ///     Visit and process all active tiles and leaf nodes in a tree, but don't
    ///     descend to the active voxels.  The smallest bounding boxes that will be
    ///     visited are those of leaf nodes or level-1 active tiles.
    /// @code
    /// {
    ///     struct ProcessTilesAndLeafNodes {
    ///         // Descend to leaf nodes, but no further.
    ///         template<Index LEVEL> inline bool descent() { return LEVEL > 0; }
    ///         // Use this version to descend to voxels:
    ///         //template<Index LEVEL> inline bool descent() { return true; }
    ///
    ///         template<Index LEVEL>
    ///         inline void operator()(const CoordBBox &bbox) {
    ///             if (LEVEL > 0) {
    ///                 // code to process an active tile
    ///             } else {
    ///                 // code to process a leaf node
    ///             }
    ///         }
    ///     };
    ///     ProcessTilesAndLeafNodes op;
    ///     aTree.visitActiveBBox(op);
    /// }
    /// @endcode
    /// @see openvdb/unittest/TestTree.cc for another example.
    template<typename BBoxOp> void visitActiveBBox(BBoxOp& op) const { mRoot.visitActiveBBox(op); }

    /// Traverse this tree in depth-first order, and at each node call the given functor
    /// with a @c DenseIterator (see Iterator.h) that points to either a child node or a
    /// tile value.  If the iterator points to a child node and the functor returns true,
    /// do not descend to the child node; instead, continue the traversal at the next
    /// iterator position.
    /// @param op  a functor of the form <tt>template<typename IterT> bool op(IterT&)</tt>,
    ///            where @c IterT is either a RootNode::ChildAllIter,
    ///            an InternalNode::ChildAllIter or a LeafNode::ChildAllIter
    ///
    /// @note There is no iterator that points to a RootNode, so to visit the root node,
    /// retrieve the @c parent() of a RootNode::ChildAllIter.
    ///
    /// @par Example:
    ///     Print information about the nodes and tiles of a tree, but not individual voxels.
    /// @code
    /// namespace {
    ///     template<typename TreeT>
    ///     struct PrintTreeVisitor
    ///     {
    ///         using RootT = typename TreeT::RootNodeType;
    ///         bool visitedRoot;
    ///
    ///         PrintTreeVisitor(): visitedRoot(false) {}
    ///
    ///         template<typename IterT>
    ///         inline bool operator()(IterT& iter)
    ///         {
    ///             if (!visitedRoot && iter.parent().getLevel() == RootT::LEVEL) {
    ///                 visitedRoot = true;
    ///                 std::cout << "Level-" << RootT::LEVEL << " node" << std::endl;
    ///             }
    ///             typename IterT::NonConstValueType value;
    ///             typename IterT::ChildNodeType* child = iter.probeChild(value);
    ///             if (child == nullptr) {
    ///                 std::cout << "Tile with value " << value << std::endl;
    ///                 return true; // no child to visit, so stop descending
    ///             }
    ///             std::cout << "Level-" << child->getLevel() << " node" << std::endl;
    ///             return (child->getLevel() == 0); // don't visit leaf nodes
    ///         }
    ///
    ///         // The generic method, above, calls iter.probeChild(), which is not defined
    ///         // for LeafNode::ChildAllIter.  These overloads ensure that the generic
    ///         // method template doesn't get instantiated for LeafNode iterators.
    ///         bool operator()(typename TreeT::LeafNodeType::ChildAllIter&) { return true; }
    ///         bool operator()(typename TreeT::LeafNodeType::ChildAllCIter&) { return true; }
    ///     };
    /// }
    /// {
    ///     PrintTreeVisitor visitor;
    ///     tree.visit(visitor);
    /// }
    /// @endcode
    template<typename VisitorOp> void visit(VisitorOp& op);
    template<typename VisitorOp> void visit(const VisitorOp& op);

    /// Like visit(), but using @c const iterators, i.e., with
    /// @param op  a functor of the form <tt>template<typename IterT> bool op(IterT&)</tt>,
    ///            where @c IterT is either a RootNode::ChildAllCIter,
    ///            an InternalNode::ChildAllCIter or a LeafNode::ChildAllCIter
    template<typename VisitorOp> void visit(VisitorOp& op) const;
    template<typename VisitorOp> void visit(const VisitorOp& op) const;

    /// Traverse this tree and another tree in depth-first order, and for corresponding
    /// subregions of index space call the given functor with two @c DenseIterators
    /// (see Iterator.h), each of which points to either a child node or a tile value
    /// of this tree and the other tree.  If the A iterator points to a child node
    /// and the functor returns a nonzero value with bit 0 set (e.g., 1), do not descend
    /// to the child node; instead, continue the traversal at the next A iterator position.
    /// Similarly, if the B iterator points to a child node and the functor returns a value
    /// with bit 1 set (e.g., 2), continue the traversal at the next B iterator position.
    /// @note The other tree must have the same index space and fan-out factors as
    /// this tree, but it may have a different @c ValueType and a different topology.
    /// @param other  a tree of the same type as this tree
    /// @param op     a functor of the form
    ///               <tt>template<class AIterT, class BIterT> int op(AIterT&, BIterT&)</tt>,
    ///               where @c AIterT and @c BIterT are any combination of a
    ///               RootNode::ChildAllIter, an InternalNode::ChildAllIter or a
    ///               LeafNode::ChildAllIter with an @c OtherTreeType::RootNode::ChildAllIter,
    ///               an @c OtherTreeType::InternalNode::ChildAllIter
    ///               or an @c OtherTreeType::LeafNode::ChildAllIter
    ///
    /// @par Example:
    ///     Given two trees of the same type, @c aTree and @c bTree, replace leaf nodes of
    ///     @c aTree with corresponding leaf nodes of @c bTree, leaving @c bTree partially empty.
    /// @code
    /// namespace {
    ///     template<typename AIterT, typename BIterT>
    ///     inline int stealLeafNodes(AIterT& aIter, BIterT& bIter)
    ///     {
    ///         typename AIterT::NonConstValueType aValue;
    ///         typename AIterT::ChildNodeType* aChild = aIter.probeChild(aValue);
    ///         typename BIterT::NonConstValueType bValue;
    ///         typename BIterT::ChildNodeType* bChild = bIter.probeChild(bValue);
    ///
    ///         const Index aLevel = aChild->getLevel(), bLevel = bChild->getLevel();
    ///         if (aChild && bChild && aLevel == 0 && bLevel == 0) { // both are leaf nodes
    ///             aIter.setChild(bChild); // give B's child to A
    ///             bIter.setValue(bValue); // replace B's child with a constant tile value
    ///         }
    ///         // Don't iterate over leaf node voxels of either A or B.
    ///         int skipBranch = (aLevel == 0) ? 1 : 0;
    ///         if (bLevel == 0) skipBranch = skipBranch | 2;
    ///         return skipBranch;
    ///     }
    /// }
    /// {
    ///     aTree.visit2(bTree, stealLeafNodes);
    /// }
    /// @endcode
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, VisitorOp& op);
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, const VisitorOp& op);

    /// Like visit2(), but using @c const iterators, i.e., with
    /// @param other  a tree of the same type as this tree
    /// @param op     a functor of the form
    ///               <tt>template<class AIterT, class BIterT> int op(AIterT&, BIterT&)</tt>,
    ///               where @c AIterT and @c BIterT are any combination of a
    ///               RootNode::ChildAllCIter, an InternalNode::ChildAllCIter
    ///               or a LeafNode::ChildAllCIter with an
    ///               @c OtherTreeType::RootNode::ChildAllCIter,
    ///               an @c OtherTreeType::InternalNode::ChildAllCIter
    ///               or an @c OtherTreeType::LeafNode::ChildAllCIter
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, VisitorOp& op) const;
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, const VisitorOp& op) const;


    //
    // Iteration
    //
    //@{
    /// Return an iterator over children of the root node.
    typename RootNodeType::ChildOnCIter  beginRootChildren() const { return mRoot.cbeginChildOn(); }
    typename RootNodeType::ChildOnCIter cbeginRootChildren() const { return mRoot.cbeginChildOn(); }
    typename RootNodeType::ChildOnIter   beginRootChildren() { return mRoot.beginChildOn(); }
    //@}

    //@{
    /// Return an iterator over non-child entries of the root node's table.
    typename RootNodeType::ChildOffCIter  beginRootTiles() const { return mRoot.cbeginChildOff(); }
    typename RootNodeType::ChildOffCIter cbeginRootTiles() const { return mRoot.cbeginChildOff(); }
    typename RootNodeType::ChildOffIter   beginRootTiles() { return mRoot.beginChildOff(); }
    //@}

    //@{
    /// Return an iterator over all entries of the root node's table.
    typename RootNodeType::ChildAllCIter  beginRootDense() const { return mRoot.cbeginChildAll(); }
    typename RootNodeType::ChildAllCIter cbeginRootDense() const { return mRoot.cbeginChildAll(); }
    typename RootNodeType::ChildAllIter   beginRootDense() { return mRoot.beginChildAll(); }
    //@}


    //@{
    /// Iterator over all nodes in this tree
    using NodeIter = NodeIteratorBase<Tree, typename RootNodeType::ChildOnIter>;
    using NodeCIter = NodeIteratorBase<const Tree, typename RootNodeType::ChildOnCIter>;
    //@}

    //@{
    /// Iterator over all leaf nodes in this tree
    using LeafIter = LeafIteratorBase<Tree, typename RootNodeType::ChildOnIter>;
    using LeafCIter = LeafIteratorBase<const Tree, typename RootNodeType::ChildOnCIter>;
    //@}

    //@{
    /// Return an iterator over all nodes in this tree.
    NodeIter   beginNode() { return NodeIter(*this); }
    NodeCIter  beginNode() const { return NodeCIter(*this); }
    NodeCIter cbeginNode() const { return NodeCIter(*this); }
    //@}

    //@{
    /// Return an iterator over all leaf nodes in this tree.
    LeafIter   beginLeaf() { return LeafIter(*this); }
    LeafCIter  beginLeaf() const { return LeafCIter(*this); }
    LeafCIter cbeginLeaf() const { return LeafCIter(*this); }
    //@}

    using ValueAllIter = TreeValueIteratorBase<Tree, typename RootNodeType::ValueAllIter>;
    using ValueAllCIter = TreeValueIteratorBase<const Tree, typename RootNodeType::ValueAllCIter>;
    using ValueOnIter = TreeValueIteratorBase<Tree, typename RootNodeType::ValueOnIter>;
    using ValueOnCIter = TreeValueIteratorBase<const Tree, typename RootNodeType::ValueOnCIter>;
    using ValueOffIter = TreeValueIteratorBase<Tree, typename RootNodeType::ValueOffIter>;
    using ValueOffCIter = TreeValueIteratorBase<const Tree, typename RootNodeType::ValueOffCIter>;

    //@{
    /// Return an iterator over all values (tile and voxel) across all nodes.
    ValueAllIter   beginValueAll() { return ValueAllIter(*this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(*this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(*this); }
    //@}
    //@{
    /// Return an iterator over active values (tile and voxel) across all nodes.
    ValueOnIter   beginValueOn() { return ValueOnIter(*this); }
    ValueOnCIter  beginValueOn() const { return ValueOnCIter(*this); }
    ValueOnCIter cbeginValueOn() const { return ValueOnCIter(*this); }
    //@}
    //@{
    /// Return an iterator over inactive values (tile and voxel) across all nodes.
    ValueOffIter   beginValueOff() { return ValueOffIter(*this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(*this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(*this); }
    //@}

    /// @brief Return an iterator of type @c IterT (for example, begin<ValueOnIter>() is
    /// equivalent to beginValueOn()).
    template<typename IterT> IterT begin();
    /// @brief Return a const iterator of type CIterT (for example, cbegin<ValueOnCIter>()
    /// is equivalent to cbeginValueOn()).
    template<typename CIterT> CIterT cbegin() const;


protected:
    using AccessorRegistry = tbb::concurrent_hash_map<ValueAccessorBase<Tree, true>*, bool>;
    using ConstAccessorRegistry = tbb::concurrent_hash_map<ValueAccessorBase<const Tree, true>*, bool>;

    /// @brief Notify all registered accessors, by calling ValueAccessor::release(),
    /// that this tree is about to be deleted.
    void releaseAllAccessors();

    // TBB body object used to deallocates nodes in parallel.
    template<typename NodeType>
    struct DeallocateNodes {
        DeallocateNodes(std::vector<NodeType*>& nodes)
            : mNodes(nodes.empty() ? nullptr : &nodes.front()) { }
        void operator()(const tbb::blocked_range<size_t>& range) const {
            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                delete mNodes[n]; mNodes[n] = nullptr;
            }
        }
        NodeType ** const mNodes;
    };

    //
    // Data members
    //
    RootNodeType mRoot; // root node of the tree
    mutable AccessorRegistry mAccessorRegistry;
    mutable ConstAccessorRegistry mConstAccessorRegistry;

    static tbb::atomic<const Name*> sTreeTypeName;
}; // end of Tree class

template<typename _RootNodeType>
tbb::atomic<const Name*> Tree<_RootNodeType>::sTreeTypeName;


/// @brief Tree3<T, N1, N2>::Type is the type of a three-level tree
/// (Root, Internal, Leaf) with value type T and
/// internal and leaf node log dimensions N1 and N2, respectively.
/// @note This is NOT the standard tree configuration (Tree4 is).
template<typename T, Index N1=4, Index N2=3>
struct Tree3 {
    using Type = Tree<RootNode<InternalNode<LeafNode<T, N2>, N1>>>;
};


/// @brief Tree4<T, N1, N2, N3>::Type is the type of a four-level tree
/// (Root, Internal, Internal, Leaf) with value type T and
/// internal and leaf node log dimensions N1, N2 and N3, respectively.
/// @note This is the standard tree configuration.
template<typename T, Index N1=5, Index N2=4, Index N3=3>
struct Tree4 {
    using Type = Tree<RootNode<InternalNode<InternalNode<LeafNode<T, N3>, N2>, N1>>>;
};

/// @brief Tree5<T, N1, N2, N3, N4>::Type is the type of a five-level tree
/// (Root, Internal, Internal, Internal, Leaf) with value type T and
/// internal and leaf node log dimensions N1, N2, N3 and N4, respectively.
/// @note This is NOT the standard tree configuration (Tree4 is).
template<typename T, Index N1=6, Index N2=5, Index N3=4, Index N4=3>
struct Tree5 {
    using Type =
        Tree<RootNode<InternalNode<InternalNode<InternalNode<LeafNode<T, N4>, N3>, N2>, N1>>>;
};


////////////////////////////////////////


inline void
TreeBase::readTopology(std::istream& is, bool /*saveFloatAsHalf*/)
{
    int32_t bufferCount;
    is.read(reinterpret_cast<char*>(&bufferCount), sizeof(int32_t));
    if (bufferCount != 1) OPENVDB_LOG_WARN("multi-buffer trees are no longer supported");
}


inline void
TreeBase::writeTopology(std::ostream& os, bool /*saveFloatAsHalf*/) const
{
    int32_t bufferCount = 1;
    os.write(reinterpret_cast<char*>(&bufferCount), sizeof(int32_t));
}


inline void
TreeBase::print(std::ostream& os, int /*verboseLevel*/) const
{
    os << "    Tree Type: " << type()
       << "    Active Voxel Count: " << activeVoxelCount() << std::endl
#if OPENVDB_ABI_VERSION_NUMBER >= 3
       << "    Active tile Count: " << activeTileCount() << std::endl
#endif
       << "    Inactive Voxel Count: " << inactiveVoxelCount() << std::endl
       << "    Leaf Node Count: " << leafCount() << std::endl
       << "    Non-leaf Node Count: " << nonLeafCount() << std::endl;
}


////////////////////////////////////////


//
// Type traits for tree iterators
//

/// @brief TreeIterTraits provides, for all tree iterators, a begin(tree) function
/// that returns an iterator over a tree of arbitrary type.
template<typename TreeT, typename IterT> struct TreeIterTraits;

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOnIter> {
    static typename TreeT::RootNodeType::ChildOnIter begin(TreeT& tree) {
        return tree.beginRootChildren();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOnCIter> {
    static typename TreeT::RootNodeType::ChildOnCIter begin(const TreeT& tree) {
        return tree.cbeginRootChildren();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOffIter> {
    static typename TreeT::RootNodeType::ChildOffIter begin(TreeT& tree) {
        return tree.beginRootTiles();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOffCIter> {
    static typename TreeT::RootNodeType::ChildOffCIter begin(const TreeT& tree) {
        return tree.cbeginRootTiles();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildAllIter> {
    static typename TreeT::RootNodeType::ChildAllIter begin(TreeT& tree) {
        return tree.beginRootDense();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildAllCIter> {
    static typename TreeT::RootNodeType::ChildAllCIter begin(const TreeT& tree) {
        return tree.cbeginRootDense();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::NodeIter> {
    static typename TreeT::NodeIter begin(TreeT& tree) { return tree.beginNode(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::NodeCIter> {
    static typename TreeT::NodeCIter begin(const TreeT& tree) { return tree.cbeginNode(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::LeafIter> {
    static typename TreeT::LeafIter begin(TreeT& tree) { return tree.beginLeaf(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::LeafCIter> {
    static typename TreeT::LeafCIter begin(const TreeT& tree) { return tree.cbeginLeaf(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOnIter> {
    static typename TreeT::ValueOnIter begin(TreeT& tree) { return tree.beginValueOn(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOnCIter> {
    static typename TreeT::ValueOnCIter begin(const TreeT& tree) { return tree.cbeginValueOn(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOffIter> {
    static typename TreeT::ValueOffIter begin(TreeT& tree) { return tree.beginValueOff(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOffCIter> {
    static typename TreeT::ValueOffCIter begin(const TreeT& tree) { return tree.cbeginValueOff(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueAllIter> {
    static typename TreeT::ValueAllIter begin(TreeT& tree) { return tree.beginValueAll(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueAllCIter> {
    static typename TreeT::ValueAllCIter begin(const TreeT& tree) { return tree.cbeginValueAll(); }
};


template<typename RootNodeType>
template<typename IterT>
inline IterT
Tree<RootNodeType>::begin()
{
    return TreeIterTraits<Tree, IterT>::begin(*this);
}


template<typename RootNodeType>
template<typename IterT>
inline IterT
Tree<RootNodeType>::cbegin() const
{
    return TreeIterTraits<Tree, IterT>::begin(*this);
}


////////////////////////////////////////


template<typename RootNodeType>
void
Tree<RootNodeType>::readTopology(std::istream& is, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    TreeBase::readTopology(is, saveFloatAsHalf);
    mRoot.readTopology(is, saveFloatAsHalf);
}


template<typename RootNodeType>
void
Tree<RootNodeType>::writeTopology(std::ostream& os, bool saveFloatAsHalf) const
{
    TreeBase::writeTopology(os, saveFloatAsHalf);
    mRoot.writeTopology(os, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::readBuffers(std::istream &is, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    mRoot.readBuffers(is, saveFloatAsHalf);
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3

template<typename RootNodeType>
inline void
Tree<RootNodeType>::readBuffers(std::istream &is, const CoordBBox& bbox, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    mRoot.readBuffers(is, bbox, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::readNonresidentBuffers() const
{
    for (LeafCIter it = this->cbeginLeaf(); it; ++it) {
        // Retrieving the value of a leaf voxel forces loading of the leaf node's voxel buffer.
        it->getValue(Index(0));
    }
}

#endif


template<typename RootNodeType>
inline void
Tree<RootNodeType>::writeBuffers(std::ostream &os, bool saveFloatAsHalf) const
{
    mRoot.writeBuffers(os, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clear()
{
    std::vector<LeafNodeType*> leafnodes;
    this->stealNodes(leafnodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafnodes.size()),
        DeallocateNodes<LeafNodeType>(leafnodes));

    std::vector<typename RootNodeType::ChildNodeType*> internalNodes;
    this->stealNodes(internalNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalNodes.size()),
        DeallocateNodes<typename RootNodeType::ChildNodeType>(internalNodes));

    mRoot.clear();

    this->clearAllAccessors();
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::attachAccessor(ValueAccessorBase<Tree, true>& accessor) const
{
    typename AccessorRegistry::accessor a;
    mAccessorRegistry.insert(a, &accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::attachAccessor(ValueAccessorBase<const Tree, true>& accessor) const
{
    typename ConstAccessorRegistry::accessor a;
    mConstAccessorRegistry.insert(a, &accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAccessor(ValueAccessorBase<Tree, true>& accessor) const
{
    mAccessorRegistry.erase(&accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAccessor(ValueAccessorBase<const Tree, true>& accessor) const
{
    mConstAccessorRegistry.erase(&accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clearAllAccessors()
{
    for (typename AccessorRegistry::iterator it = mAccessorRegistry.begin();
        it != mAccessorRegistry.end(); ++it)
    {
        if (it->first) it->first->clear();
    }

    for (typename ConstAccessorRegistry::iterator it = mConstAccessorRegistry.begin();
        it != mConstAccessorRegistry.end(); ++it)
    {
        if (it->first) it->first->clear();
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAllAccessors()
{
    mAccessorRegistry.erase(nullptr);
    for (typename AccessorRegistry::iterator it = mAccessorRegistry.begin();
        it != mAccessorRegistry.end(); ++it)
    {
        it->first->release();
    }
    mAccessorRegistry.clear();

    mAccessorRegistry.erase(nullptr);
    for (typename ConstAccessorRegistry::iterator it = mConstAccessorRegistry.begin();
        it != mConstAccessorRegistry.end(); ++it)
    {
        it->first->release();
    }
    mConstAccessorRegistry.clear();
}


////////////////////////////////////////


template<typename RootNodeType>
inline const typename RootNodeType::ValueType&
Tree<RootNodeType>::getValue(const Coord& xyz) const
{
    return mRoot.getValue(xyz);
}


template<typename RootNodeType>
template<typename AccessT>
inline const typename RootNodeType::ValueType&
Tree<RootNodeType>::getValue(const Coord& xyz, AccessT& accessor) const
{
    return accessor.getValue(xyz);
}


template<typename RootNodeType>
inline int
Tree<RootNodeType>::getValueDepth(const Coord& xyz) const
{
    return mRoot.getValueDepth(xyz);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOff(const Coord& xyz)
{
    mRoot.setValueOff(xyz);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOff(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOff(xyz, value);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setActiveState(const Coord& xyz, bool on)
{
    mRoot.setActiveState(xyz, on);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValue(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOn(xyz, value);
}

template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOnly(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOnly(xyz, value);
}

template<typename RootNodeType>
template<typename AccessT>
inline void
Tree<RootNodeType>::setValue(const Coord& xyz, const ValueType& value, AccessT& accessor)
{
    accessor.setValue(xyz, value);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOn(const Coord& xyz)
{
    mRoot.setActiveState(xyz, true);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOn(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOn(xyz, value);
}


template<typename RootNodeType>
template<typename ModifyOp>
inline void
Tree<RootNodeType>::modifyValue(const Coord& xyz, const ModifyOp& op)
{
    mRoot.modifyValue(xyz, op);
}


template<typename RootNodeType>
template<typename ModifyOp>
inline void
Tree<RootNodeType>::modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
{
    mRoot.modifyValueAndActiveState(xyz, op);
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::probeValue(const Coord& xyz, ValueType& value) const
{
    return mRoot.probeValue(xyz, value);
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::addTile(Index level, const Coord& xyz,
                            const ValueType& value, bool active)
{
    mRoot.addTile(level, xyz, value, active);
}


template<typename RootNodeType>
template<typename NodeT>
inline NodeT*
Tree<RootNodeType>::stealNode(const Coord& xyz, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.template stealNode<NodeT>(xyz, value, active);
}


template<typename RootNodeType>
inline typename RootNodeType::LeafNodeType*
Tree<RootNodeType>::touchLeaf(const Coord& xyz)
{
    return mRoot.touchLeaf(xyz);
}


template<typename RootNodeType>
inline typename RootNodeType::LeafNodeType*
Tree<RootNodeType>::probeLeaf(const Coord& xyz)
{
    return mRoot.probeLeaf(xyz);
}


template<typename RootNodeType>
inline const typename RootNodeType::LeafNodeType*
Tree<RootNodeType>::probeConstLeaf(const Coord& xyz) const
{
    return mRoot.probeConstLeaf(xyz);
}


template<typename RootNodeType>
template<typename NodeType>
inline NodeType*
Tree<RootNodeType>::probeNode(const Coord& xyz)
{
    return mRoot.template probeNode<NodeType>(xyz);
}


template<typename RootNodeType>
template<typename NodeType>
inline const NodeType*
Tree<RootNodeType>::probeNode(const Coord& xyz) const
{
    return this->template probeConstNode<NodeType>(xyz);
}


template<typename RootNodeType>
template<typename NodeType>
inline const NodeType*
Tree<RootNodeType>::probeConstNode(const Coord& xyz) const
{
    return mRoot.template probeConstNode<NodeType>(xyz);
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clip(const CoordBBox& bbox)
{
    this->clearAllAccessors();
    return mRoot.clip(bbox);
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3
template<typename RootNodeType>
inline void
Tree<RootNodeType>::clipUnallocatedNodes()
{
    this->clearAllAccessors();
    for (LeafIter it = this->beginLeaf(); it; ) {
        const LeafNodeType* leaf = it.getLeaf();
        ++it; // advance the iterator before deleting the leaf node
        if (!leaf->isAllocated()) {
            this->addTile(/*level=*/0, leaf->origin(), this->background(), /*active=*/false);
        }
    }
}
#endif

#if OPENVDB_ABI_VERSION_NUMBER >= 4
template<typename RootNodeType>
inline Index32
Tree<RootNodeType>::unallocatedLeafCount() const
{
    Index32 sum = 0;
    for (auto it = this->cbeginLeaf(); it; ++it) if (!it->isAllocated()) ++sum;
    return sum;
}
#endif


template<typename RootNodeType>
inline void
Tree<RootNodeType>::sparseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.sparseFill(bbox, value, active);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::denseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.denseFill(bbox, value, active);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::voxelizeActiveTiles(bool threaded)
{
    this->clearAllAccessors();
    mRoot.voxelizeActiveTiles(threaded);
}


template<typename RootNodeType>
Metadata::Ptr
Tree<RootNodeType>::getBackgroundValue() const
{
    Metadata::Ptr result;
    if (Metadata::isRegisteredType(valueType())) {
        using MetadataT = TypedMetadata<ValueType>;
        result = Metadata::createMetadata(valueType());
        if (result->typeName() == MetadataT::staticTypeName()) {
            MetadataT* m = static_cast<MetadataT*>(result.get());
            m->value() = mRoot.background();
        }
    }
    return result;
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::merge(Tree& other, MergePolicy policy)
{
    this->clearAllAccessors();
    other.clearAllAccessors();
    switch (policy) {
        case MERGE_ACTIVE_STATES:
            mRoot.template merge<MERGE_ACTIVE_STATES>(other.mRoot); break;
        case MERGE_NODES:
            mRoot.template merge<MERGE_NODES>(other.mRoot); break;
        case MERGE_ACTIVE_STATES_AND_NODES:
            mRoot.template merge<MERGE_ACTIVE_STATES_AND_NODES>(other.mRoot); break;
    }
}


template<typename RootNodeType>
template<typename OtherRootNodeType>
inline void
Tree<RootNodeType>::topologyUnion(const Tree<OtherRootNodeType>& other)
{
    this->clearAllAccessors();
    mRoot.topologyUnion(other.root());
}

template<typename RootNodeType>
template<typename OtherRootNodeType>
inline void
Tree<RootNodeType>::topologyIntersection(const Tree<OtherRootNodeType>& other)
{
    this->clearAllAccessors();
    mRoot.topologyIntersection(other.root());
}

template<typename RootNodeType>
template<typename OtherRootNodeType>
inline void
Tree<RootNodeType>::topologyDifference(const Tree<OtherRootNodeType>& other)
{
    this->clearAllAccessors();
    mRoot.topologyDifference(other.root());
}

////////////////////////////////////////


/// @brief Helper class to adapt a three-argument (a, b, result) CombineOp functor
/// into a single-argument functor that accepts a CombineArgs struct
template<typename AValueT, typename CombineOp, typename BValueT = AValueT>
struct CombineOpAdapter
{
    CombineOpAdapter(CombineOp& _op): op(_op) {}

    void operator()(CombineArgs<AValueT, BValueT>& args) const {
        op(args.a(), args.b(), args.result());
    }

    CombineOp& op;
};


template<typename RootNodeType>
template<typename CombineOp>
inline void
Tree<RootNodeType>::combine(Tree& other, CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, CombineOp> extendedOp(op);
    this->combineExtended(other, extendedOp, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.combine(bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename CombineOp>
inline void
Tree<RootNodeType>::combine(Tree& other, const CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, const CombineOp> extendedOp(op);
    this->combineExtended(other, extendedOp, prune);
}
#endif


template<typename RootNodeType>
template<typename ExtendedCombineOp>
inline void
Tree<RootNodeType>::combineExtended(Tree& other, ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.combine(other.root(), op, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.combineExtended(bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename ExtendedCombineOp>
inline void
Tree<RootNodeType>::combineExtended(Tree& other, const ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.template combine<const ExtendedCombineOp>(other.mRoot, op, prune);
}
#endif


template<typename RootNodeType>
template<typename CombineOp, typename OtherTreeType>
inline void
Tree<RootNodeType>::combine2(const Tree& a, const OtherTreeType& b, CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, CombineOp, typename OtherTreeType::ValueType> extendedOp(op);
    this->combine2Extended(a, b, extendedOp, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.combine2(aTree, bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename CombineOp, typename OtherTreeType>
inline void
Tree<RootNodeType>::combine2(const Tree& a, const OtherTreeType& b, const CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, const CombineOp, typename OtherTreeType::ValueType> extendedOp(op);
    this->combine2Extended(a, b, extendedOp, prune);
}
#endif


template<typename RootNodeType>
template<typename ExtendedCombineOp, typename OtherTreeType>
inline void
Tree<RootNodeType>::combine2Extended(const Tree& a, const OtherTreeType& b,
    ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.combine2(a.root(), b.root(), op, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like the following, where the functor argument is a temporary:
/// <tt>tree.combine2Extended(aTree, bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename ExtendedCombineOp, typename OtherTreeType>
inline void
Tree<RootNodeType>::combine2Extended(const Tree& a, const OtherTreeType& b,
    const ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.template combine2<const ExtendedCombineOp>(a.root(), b.root(), op, prune);
}
#endif


////////////////////////////////////////


template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(VisitorOp& op)
{
    this->clearAllAccessors();
    mRoot.template visit<VisitorOp>(op);
}


template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(VisitorOp& op) const
{
    mRoot.template visit<VisitorOp>(op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.visit(MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(const VisitorOp& op)
{
    this->clearAllAccessors();
    mRoot.template visit<const VisitorOp>(op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.visit(MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(const VisitorOp& op) const
{
    mRoot.template visit<const VisitorOp>(op);
}


////////////////////////////////////////


template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, VisitorOp& op)
{
    this->clearAllAccessors();
    using OtherRootNodeType = typename OtherTreeType::RootNodeType;
    mRoot.template visit2<OtherRootNodeType, VisitorOp>(other.root(), op);
}


template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, VisitorOp& op) const
{
    using OtherRootNodeType = typename OtherTreeType::RootNodeType;
    mRoot.template visit2<OtherRootNodeType, VisitorOp>(other.root(), op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.visit2(bTree, MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, const VisitorOp& op)
{
    this->clearAllAccessors();
    using OtherRootNodeType = typename OtherTreeType::RootNodeType;
    mRoot.template visit2<OtherRootNodeType, const VisitorOp>(other.root(), op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.visit2(bTree, MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, const VisitorOp& op) const
{
    using OtherRootNodeType = typename OtherTreeType::RootNodeType;
    mRoot.template visit2<OtherRootNodeType, const VisitorOp>(other.root(), op);
}


////////////////////////////////////////


template<typename RootNodeType>
inline const Name&
Tree<RootNodeType>::treeType()
{
    if (sTreeTypeName == nullptr) {
        std::vector<Index> dims;
        Tree::getNodeLog2Dims(dims);
        std::ostringstream ostr;
        ostr << "Tree_" << typeNameAsString<BuildType>();
        for (size_t i = 1, N = dims.size(); i < N; ++i) { // start from 1 to skip the RootNode
            ostr << "_" << dims[i];
        }
        Name* s = new Name(ostr.str());
        if (sTreeTypeName.compare_and_swap(s, nullptr) != nullptr) delete s;
    }
    return *sTreeTypeName;
}


template<typename RootNodeType>
template<typename OtherRootNodeType>
inline bool
Tree<RootNodeType>::hasSameTopology(const Tree<OtherRootNodeType>& other) const
{
    return mRoot.hasSameTopology(other.root());
}


template<typename RootNodeType>
Index64
Tree<RootNodeType>::inactiveVoxelCount() const
{
    Coord dim(0, 0, 0);
    this->evalActiveVoxelDim(dim);
    const Index64
        totalVoxels = dim.x() * dim.y() * dim.z(),
        activeVoxels = this->activeVoxelCount();
    assert(totalVoxels >= activeVoxels);
    return totalVoxels - activeVoxels;
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalLeafBoundingBox(CoordBBox& bbox) const
{
    bbox.reset(); // default invalid bbox

    if (this->empty()) return false;  // empty

    mRoot.evalActiveBoundingBox(bbox, false);

    return true;// not empty
}

template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalActiveVoxelBoundingBox(CoordBBox& bbox) const
{
    bbox.reset(); // default invalid bbox

    if (this->empty()) return false;  // empty

    mRoot.evalActiveBoundingBox(bbox, true);

    return true;// not empty
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalActiveVoxelDim(Coord& dim) const
{
    CoordBBox bbox;
    bool notEmpty = this->evalActiveVoxelBoundingBox(bbox);
    dim = bbox.extents();
    return notEmpty;
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalLeafDim(Coord& dim) const
{
    CoordBBox bbox;
    bool notEmpty = this->evalLeafBoundingBox(bbox);
    dim = bbox.extents();
    return notEmpty;
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::evalMinMax(ValueType& minVal, ValueType& maxVal) const
{
    /// @todo optimize
    minVal = maxVal = zeroVal<ValueType>();
    if (ValueOnCIter iter = this->cbeginValueOn()) {
        minVal = maxVal = *iter;
        for (++iter; iter; ++iter) {
            const ValueType& val = *iter;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::getNodeLog2Dims(std::vector<Index>& dims)
{
    dims.clear();
    RootNodeType::getNodeLog2Dims(dims);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::print(std::ostream& os, int verboseLevel) const
{
    if (verboseLevel <= 0) return;

    /// @todo Consider using boost::io::ios_precision_saver instead.
    struct OnExit {
        std::ostream& os;
        std::streamsize savedPrecision;
        OnExit(std::ostream& _os): os(_os), savedPrecision(os.precision()) {}
        ~OnExit() { os.precision(savedPrecision); }
    };
    OnExit restorePrecision(os);

    std::vector<Index> dims;
    Tree::getNodeLog2Dims(dims);

    os << "Information about Tree:\n"
        << "  Type: " << this->type() << "\n";

    os << "  Configuration:\n";

    if (verboseLevel <= 1) {
        // Print node types and sizes.
        os << "    Root(" << mRoot.getTableSize() << ")";
        if (dims.size() > 1) {
            for (size_t i = 1, N = dims.size() - 1; i < N; ++i) {
                os << ", Internal(" << (1 << dims[i]) << "^3)";
            }
            os << ", Leaf(" << (1 << *dims.rbegin()) << "^3)\n";
        }
        os << "  Background value: " << mRoot.background() << "\n";
        return;
    }

    // The following is tree information that is expensive to extract.

    ValueType minVal = zeroVal<ValueType>(), maxVal = zeroVal<ValueType>();
    if (verboseLevel > 3) {
        // This forces loading of all non-resident nodes.
        this->evalMinMax(minVal, maxVal);
    }

    std::vector<Index64> nodeCount(dims.size());
    for (NodeCIter it = cbeginNode(); it; ++it) ++(nodeCount[it.getDepth()]);

    Index64 totalNodeCount = 0;
    for (size_t i = 0; i < nodeCount.size(); ++i) totalNodeCount += nodeCount[i];

    // Print node types, counts and sizes.
    os << "    Root(1 x " << mRoot.getTableSize() << ")";
    if (dims.size() > 1) {
        for (size_t i = 1, N = dims.size() - 1; i < N; ++i) {
            os << ", Internal(" << util::formattedInt(nodeCount[i]);
            os << " x " << (1 << dims[i]) << "^3)";
        }
        os << ", Leaf(" << util::formattedInt(*nodeCount.rbegin());
        os << " x " << (1 << *dims.rbegin()) << "^3)\n";
    }
    os << "  Background value: " << mRoot.background() << "\n";

    // Statistics of topology and values

    if (verboseLevel > 3) {
        os << "  Min value: " << minVal << "\n";
        os << "  Max value: " << maxVal << "\n";
    }

    const Index64
        leafCount = *nodeCount.rbegin(),
        numActiveVoxels = this->activeVoxelCount(),
        numActiveLeafVoxels = this->activeLeafVoxelCount(),
        numActiveTiles = this->activeTileCount();

    os << "  Number of active voxels:       " << util::formattedInt(numActiveVoxels) << "\n";
    os << "  Number of active tiles:        " << util::formattedInt(numActiveTiles) << "\n";

    Coord dim(0, 0, 0);
    Index64 totalVoxels = 0;
    if (numActiveVoxels) { // nonempty
        CoordBBox bbox;
        this->evalActiveVoxelBoundingBox(bbox);
        dim = bbox.extents();
        totalVoxels = dim.x() * uint64_t(dim.y()) * dim.z();

        os << "  Bounding box of active voxels: " << bbox << "\n";
        os << "  Dimensions of active voxels:   "
            << dim[0] << " x " << dim[1] << " x " << dim[2] << "\n";

        const double activeRatio = (100.0 * double(numActiveVoxels)) / double(totalVoxels);
        os << "  Percentage of active voxels:   " << std::setprecision(3) << activeRatio << "%\n";

        if (leafCount > 0) {
            const double fillRatio = (100.0 * double(numActiveLeafVoxels))
                / (double(leafCount) * double(LeafNodeType::NUM_VOXELS));
            os << "  Average leaf node fill ratio:  " << fillRatio << "%\n";
        }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
        if (verboseLevel > 2) {
            Index64 sum = 0;// count the number of unallocated leaf nodes
            for (auto it = this->cbeginLeaf(); it; ++it) if (!it->isAllocated()) ++sum;
            os << "  Number of unallocated nodes:   "
               << util::formattedInt(sum) << " ("
               << (100.0 * double(sum) / double(totalNodeCount)) << "%)\n";
        }
#endif
    } else {
        os << "  Tree is empty!\n";
    }
    os << std::flush;

    if (verboseLevel == 2) return;

    // Memory footprint in bytes
    const Index64
        actualMem = this->memUsage(),
        denseMem = sizeof(ValueType) * totalVoxels,
        voxelsMem = sizeof(ValueType) * numActiveLeafVoxels;
            ///< @todo not accurate for BoolTree (and probably should count tile values)

    os << "Memory footprint:\n";
    util::printBytes(os, actualMem, "  Actual:             ");
    util::printBytes(os, voxelsMem, "  Active leaf voxels: ");

    if (numActiveVoxels) {
        util::printBytes(os, denseMem, "  Dense equivalent:   ");
        os << "  Actual footprint is " << (100.0 * double(actualMem) / double(denseMem))
            << "% of an equivalent dense volume\n";
        os << "  Leaf voxel footprint is " << (100.0 * double(voxelsMem) / double(actualMem))
           << "% of actual footprint\n";
    }
}

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
