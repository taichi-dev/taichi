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

#ifndef OPENVDB_GRID_HAS_BEEN_INCLUDED
#define OPENVDB_GRID_HAS_BEEN_INCLUDED

#include "Exceptions.h"
#include "MetaMap.h"
#include "Types.h"
#include "io/io.h"
#include "math/Transform.h"
#include "tree/Tree.h"
#include "util/logging.h"
#include "util/Name.h"
#include <cassert>
#include <iostream>
#include <set>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

using TreeBase = tree::TreeBase;

template<typename> class Grid; // forward declaration


/// @brief Create a new grid of type @c GridType with a given background value.
///
/// @note Calling createGrid<GridType>(background) is equivalent to calling
/// GridType::create(background).
template<typename GridType>
inline typename GridType::Ptr createGrid(const typename GridType::ValueType& background);


/// @brief Create a new grid of type @c GridType with background value zero.
///
/// @note Calling createGrid<GridType>() is equivalent to calling GridType::create().
template<typename GridType>
inline typename GridType::Ptr createGrid();


/// @brief Create a new grid of the appropriate type that wraps the given tree.
///
/// @note This function can be called without specifying the template argument,
/// i.e., as createGrid(tree).
template<typename TreePtrType>
inline typename Grid<typename TreePtrType::element_type>::Ptr createGrid(TreePtrType);


/// @brief Create a new grid of type @c GridType classified as a "Level Set",
/// i.e., a narrow-band level set.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
///
/// @param voxelSize  the size of a voxel in world units
/// @param halfWidth  the half width of the narrow band in voxel units
///
/// @details The voxel size and the narrow band half width define the grid's
/// background value as halfWidth*voxelWidth.  The transform is linear
/// with a uniform scaling only corresponding to the specified voxel size.
///
/// @note It is generally advisable to specify a half-width of the narrow band
/// that is larger than one voxel unit, otherwise zero crossings are not guaranteed.
template<typename GridType>
typename GridType::Ptr createLevelSet(
    Real voxelSize = 1.0, Real halfWidth = LEVEL_SET_HALF_WIDTH);


////////////////////////////////////////


/// @brief Abstract base class for typed grids
class OPENVDB_API GridBase: public MetaMap
{
public:
    using Ptr      = SharedPtr<GridBase>;
    using ConstPtr = SharedPtr<const GridBase>;

    using GridFactory = Ptr (*)();


    ~GridBase() override {}

#if OPENVDB_ABI_VERSION_NUMBER <= 3
    /// @brief Return a new grid of the same type as this grid and whose
    /// metadata and transform are deep copies of this grid's.
    virtual GridBase::Ptr copyGrid(CopyPolicy treePolicy = CP_SHARE) const = 0;
#else
    //@{
    /// @brief Return a new grid of the same type as this grid whose metadata and
    /// transform are deep copies of this grid's and whose tree is shared with this grid.
    virtual GridBase::Ptr copyGrid() = 0;
    virtual GridBase::ConstPtr copyGrid() const = 0;
    //@}
    /// @brief Return a new grid of the same type as this grid whose metadata and
    /// transform are deep copies of this grid's and whose tree is default-constructed.
    virtual GridBase::Ptr copyGridWithNewTree() const = 0;
#endif
    /// Return a new grid whose metadata, transform and tree are deep copies of this grid's.
    virtual GridBase::Ptr deepCopyGrid() const = 0;


    //
    // Registry methods
    //
    /// Create a new grid of the given (registered) type.
    static Ptr createGrid(const Name& type);

    /// Return @c true if the given grid type name is registered.
    static bool isRegistered(const Name &type);

    /// Clear the grid type registry.
    static void clearRegistry();


    //
    // Grid type methods
    //
    /// Return the name of this grid's type.
    virtual Name type() const = 0;
    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d").
    virtual Name valueType() const = 0;

    /// Return @c true if this grid is of the same type as the template parameter.
    template<typename GridType>
    bool isType() const { return (this->type() == GridType::gridType()); }

    //@{
    /// @brief Return the result of downcasting a GridBase pointer to a Grid pointer
    /// of the specified type, or return a null pointer if the types are incompatible.
    template<typename GridType>
    static typename GridType::Ptr grid(const GridBase::Ptr&);
    template<typename GridType>
    static typename GridType::ConstPtr grid(const GridBase::ConstPtr&);
    template<typename GridType>
    static typename GridType::ConstPtr constGrid(const GridBase::Ptr&);
    template<typename GridType>
    static typename GridType::ConstPtr constGrid(const GridBase::ConstPtr&);
    //@}

    //@{
    /// @brief Return a pointer to this grid's tree, which might be
    /// shared with other grids.  The pointer is guaranteed to be non-null.
    TreeBase::Ptr baseTreePtr();
    TreeBase::ConstPtr baseTreePtr() const { return this->constBaseTreePtr(); }
    virtual TreeBase::ConstPtr constBaseTreePtr() const = 0;
    //@}

    //@{
    /// @brief Return a reference to this grid's tree, which might be
    /// shared with other grids.
    /// @note Calling setTree() on this grid invalidates all references
    /// previously returned by this method.
    TreeBase& baseTree() { return const_cast<TreeBase&>(this->constBaseTree()); }
    const TreeBase& baseTree() const { return this->constBaseTree(); }
    const TreeBase& constBaseTree() const { return *(this->constBaseTreePtr()); }
    //@}

    /// @brief Associate the given tree with this grid, in place of its existing tree.
    /// @throw ValueError if the tree pointer is null
    /// @throw TypeError if the tree is not of the appropriate type
    /// @note Invalidates all references previously returned by baseTree()
    /// or constBaseTree().
    virtual void setTree(TreeBase::Ptr) = 0;

    /// Set a new tree with the same background value as the previous tree.
    virtual void newTree() = 0;

    /// Return @c true if this grid contains only background voxels.
    virtual bool empty() const = 0;
    /// Empty this grid, setting all voxels to the background.
    virtual void clear() = 0;

    /// @brief Reduce the memory footprint of this grid by increasing its sparseness
    /// either losslessly (@a tolerance = 0) or lossily (@a tolerance > 0).
    /// @details With @a tolerance > 0, sparsify regions where voxels have the same
    /// active state and have values that differ by no more than the tolerance
    /// (converted to this grid's value type).
    virtual void pruneGrid(float tolerance = 0.0) = 0;

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// @brief Clip this grid to the given world-space bounding box.
    /// @details Voxels that lie outside the bounding box are set to the background.
    /// @warning Clipping a level set will likely produce a grid that is
    /// no longer a valid level set.
    void clipGrid(const BBoxd&);

    /// @brief Clip this grid to the given index-space bounding box.
    /// @details Voxels that lie outside the bounding box are set to the background.
    /// @warning Clipping a level set will likely produce a grid that is
    /// no longer a valid level set.
    virtual void clip(const CoordBBox&) = 0;
#endif


    //
    // Metadata
    //
    /// Return this grid's user-specified name.
    std::string getName() const;
    /// Specify a name for this grid.
    void setName(const std::string&);

    /// Return the user-specified description of this grid's creator.
    std::string getCreator() const;
    /// Provide a description of this grid's creator.
    void setCreator(const std::string&);

    /// @brief Return @c true if this grid should be written out with floating-point
    /// voxel values (including components of vectors) quantized to 16 bits.
    bool saveFloatAsHalf() const;
    void setSaveFloatAsHalf(bool);

    /// Return the class of volumetric data (level set, fog volume, etc.) stored in this grid.
    GridClass getGridClass() const;
    /// Specify the class of volumetric data (level set, fog volume, etc.) stored in this grid.
    void setGridClass(GridClass);
    /// Remove the setting specifying the class of this grid's volumetric data.
    void clearGridClass();

    /// Return the metadata string value for the given class of volumetric data.
    static std::string gridClassToString(GridClass);
    /// Return a formatted string version of the grid class.
    static std::string gridClassToMenuName(GridClass);
    /// @brief Return the class of volumetric data specified by the given string.
    /// @details If the string is not one of the ones returned by gridClassToString(),
    /// return @c GRID_UNKNOWN.
    static GridClass stringToGridClass(const std::string&);

    /// @brief Return the type of vector data (invariant, covariant, etc.) stored
    /// in this grid, assuming that this grid contains a vector-valued tree.
    VecType getVectorType() const;
    /// @brief Specify the type of vector data (invariant, covariant, etc.) stored
    /// in this grid, assuming that this grid contains a vector-valued tree.
    void setVectorType(VecType);
    /// Remove the setting specifying the type of vector data stored in this grid.
    void clearVectorType();

    /// Return the metadata string value for the given type of vector data.
    static std::string vecTypeToString(VecType);
    /// Return a string listing examples of the given type of vector data
    /// (e.g., "Gradient/Normal", given VEC_COVARIANT).
    static std::string vecTypeExamples(VecType);
    /// @brief Return a string describing how the given type of vector data is affected
    /// by transformations (e.g., "Does not transform", given VEC_INVARIANT).
    static std::string vecTypeDescription(VecType);
    static VecType stringToVecType(const std::string&);

    /// Return @c true if this grid's voxel values are in world space and should be
    /// affected by transformations, @c false if they are in local space and should
    /// not be affected by transformations.
    bool isInWorldSpace() const;
    /// Specify whether this grid's voxel values are in world space or in local space.
    void setIsInWorldSpace(bool);

    // Standard metadata field names
    // (These fields should normally not be accessed directly, but rather
    // via the accessor methods above, when available.)
    // Note: Visual C++ requires these declarations to be separate statements.
    static const char* const META_GRID_CLASS;
    static const char* const META_GRID_CREATOR;
    static const char* const META_GRID_NAME;
    static const char* const META_SAVE_HALF_FLOAT;
    static const char* const META_IS_LOCAL_SPACE;
    static const char* const META_VECTOR_TYPE;
    static const char* const META_FILE_BBOX_MIN;
    static const char* const META_FILE_BBOX_MAX;
    static const char* const META_FILE_COMPRESSION;
    static const char* const META_FILE_MEM_BYTES;
    static const char* const META_FILE_VOXEL_COUNT;


    //
    // Statistics
    //
    /// Return the number of active voxels.
    virtual Index64 activeVoxelCount() const = 0;

    /// Return the axis-aligned bounding box of all active voxels. If
    /// the grid is empty a default bbox is returned.
    virtual CoordBBox evalActiveVoxelBoundingBox() const = 0;

    /// Return the dimensions of the axis-aligned bounding box of all active voxels.
    virtual Coord evalActiveVoxelDim() const = 0;

    /// Return the number of bytes of memory used by this grid.
    virtual Index64 memUsage() const = 0;

    /// @brief Add metadata to this grid comprising the current values
    /// of statistics like the active voxel count and bounding box.
    /// @note This metadata is not automatically kept up-to-date with
    /// changes to this grid.
    void addStatsMetadata();
    /// @brief Return a new MetaMap containing just the metadata that
    /// was added to this grid with addStatsMetadata().
    /// @details If addStatsMetadata() was never called on this grid,
    /// return an empty MetaMap.
    MetaMap::Ptr getStatsMetadata() const;


    //
    // Transform methods
    //
    //@{
    /// @brief Return a pointer to this grid's transform, which might be
    /// shared with other grids.
    math::Transform::Ptr transformPtr() { return mTransform; }
    math::Transform::ConstPtr transformPtr() const { return mTransform; }
    math::Transform::ConstPtr constTransformPtr() const { return mTransform; }
    //@}
    //@{
    /// @brief Return a reference to this grid's transform, which might be
    /// shared with other grids.
    /// @note Calling setTransform() on this grid invalidates all references
    /// previously returned by this method.
    math::Transform& transform() { return *mTransform; }
    const math::Transform& transform() const { return *mTransform; }
    const math::Transform& constTransform() const { return *mTransform; }
    //@}
    /// @brief Associate the given transform with this grid, in place of
    /// its existing transform.
    /// @throw ValueError if the transform pointer is null
    /// @note Invalidates all references previously returned by transform()
    /// or constTransform().
    void setTransform(math::Transform::Ptr);

    /// Return the size of this grid's voxels.
    Vec3d voxelSize() const { return transform().voxelSize(); }
    /// @brief Return the size of this grid's voxel at position (x, y, z).
    /// @note Frustum and perspective transforms have position-dependent voxel size.
    Vec3d voxelSize(const Vec3d& xyz) const { return transform().voxelSize(xyz); }
    /// Return true if the voxels in world space are uniformly sized cubes
    bool hasUniformVoxels() const { return mTransform->hasUniformScale(); }
    //@{
    /// Apply this grid's transform to the given coordinates.
    Vec3d indexToWorld(const Vec3d& xyz) const { return transform().indexToWorld(xyz); }
    Vec3d indexToWorld(const Coord& ijk) const { return transform().indexToWorld(ijk); }
    //@}
    /// Apply the inverse of this grid's transform to the given coordinates.
    Vec3d worldToIndex(const Vec3d& xyz) const { return transform().worldToIndex(xyz); }


    //
    // I/O methods
    //
    /// @brief Read the grid topology from a stream.
    /// This will read only the grid structure, not the actual data buffers.
    virtual void readTopology(std::istream&) = 0;
    /// @brief Write the grid topology to a stream.
    /// This will write only the grid structure, not the actual data buffers.
    virtual void writeTopology(std::ostream&) const = 0;

    /// Read all data buffers for this grid.
    virtual void readBuffers(std::istream&) = 0;
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// Read all of this grid's data buffers that intersect the given index-space bounding box.
    virtual void readBuffers(std::istream&, const CoordBBox&) = 0;
    /// @brief Read all of this grid's data buffers that are not yet resident in memory
    /// (because delayed loading is in effect).
    /// @details If this grid was read from a memory-mapped file, this operation
    /// disconnects the grid from the file.
    /// @sa io::File::open, io::MappedFile
    virtual void readNonresidentBuffers() const = 0;
#endif
    /// Write out all data buffers for this grid.
    virtual void writeBuffers(std::ostream&) const = 0;

    /// Read in the transform for this grid.
    void readTransform(std::istream& is) { transform().read(is); }
    /// Write out the transform for this grid.
    void writeTransform(std::ostream& os) const { transform().write(os); }

    /// Output a human-readable description of this grid.
    virtual void print(std::ostream& = std::cout, int verboseLevel = 1) const = 0;


protected:
    /// @brief Initialize with an identity linear transform.
    GridBase(): mTransform(math::Transform::createLinearTransform()) {}

    /// @brief Deep copy another grid's metadata and transform.
    GridBase(const GridBase& other): MetaMap(other), mTransform(other.mTransform->copy()) {}

#if OPENVDB_ABI_VERSION_NUMBER <= 3
    /// @brief Copy another grid's metadata but share its transform.
    GridBase(const GridBase& other, ShallowCopy): MetaMap(other), mTransform(other.mTransform) {}
#else
    /// @brief Copy another grid's metadata but share its transform.
    GridBase(GridBase& other, ShallowCopy): MetaMap(other), mTransform(other.mTransform) {}
#endif

    /// Register a grid type along with a factory function.
    static void registerGrid(const Name& type, GridFactory);
    /// Remove a grid type from the registry.
    static void unregisterGrid(const Name& type);


private:
    math::Transform::Ptr mTransform;
}; // class GridBase


////////////////////////////////////////


using GridPtrVec       = std::vector<GridBase::Ptr>;
using GridPtrVecIter   = GridPtrVec::iterator;
using GridPtrVecCIter  = GridPtrVec::const_iterator;
using GridPtrVecPtr    = SharedPtr<GridPtrVec>;

using GridCPtrVec      = std::vector<GridBase::ConstPtr>;
using GridCPtrVecIter  = GridCPtrVec::iterator;
using GridCPtrVecCIter = GridCPtrVec::const_iterator;
using GridCPtrVecPtr   = SharedPtr<GridCPtrVec>;

using GridPtrSet       = std::set<GridBase::Ptr>;
using GridPtrSetIter   = GridPtrSet::iterator;
using GridPtrSetCIter  = GridPtrSet::const_iterator;
using GridPtrSetPtr    = SharedPtr<GridPtrSet>;

using GridCPtrSet      = std::set<GridBase::ConstPtr>;
using GridCPtrSetIter  = GridCPtrSet::iterator;
using GridCPtrSetCIter = GridCPtrSet::const_iterator;
using GridCPtrSetPtr   = SharedPtr<GridCPtrSet>;


/// @brief Predicate functor that returns @c true for grids that have a specified name
struct OPENVDB_API GridNamePred
{
    GridNamePred(const Name& _name): name(_name) {}
    bool operator()(const GridBase::ConstPtr& g) const { return g && g->getName() == name; }
    Name name;
};

/// Return the first grid in the given container whose name is @a name.
template<typename GridPtrContainerT>
inline typename GridPtrContainerT::value_type
findGridByName(const GridPtrContainerT& container, const Name& name)
{
    using GridPtrT = typename GridPtrContainerT::value_type;
    typename GridPtrContainerT::const_iterator it =
        std::find_if(container.begin(), container.end(), GridNamePred(name));
    return (it == container.end() ? GridPtrT() : *it);
}

/// Return the first grid in the given map whose name is @a name.
template<typename KeyT, typename GridPtrT>
inline GridPtrT
findGridByName(const std::map<KeyT, GridPtrT>& container, const Name& name)
{
    using GridPtrMapT = std::map<KeyT, GridPtrT>;
    for (typename GridPtrMapT::const_iterator it = container.begin(), end = container.end();
        it != end; ++it)
    {
        const GridPtrT& grid = it->second;
        if (grid && grid->getName() == name) return grid;
    }
    return GridPtrT();
}
//@}


////////////////////////////////////////


/// @brief Container class that associates a tree with a transform and metadata
template<typename _TreeType>
class Grid: public GridBase
{
public:
    using Ptr                 = SharedPtr<Grid>;
    using ConstPtr            = SharedPtr<const Grid>;

    using TreeType            = _TreeType;
    using TreePtrType         = typename _TreeType::Ptr;
    using ConstTreePtrType    = typename _TreeType::ConstPtr;
    using ValueType           = typename _TreeType::ValueType;
    using BuildType           = typename _TreeType::BuildType;

    using ValueOnIter         = typename _TreeType::ValueOnIter;
    using ValueOnCIter        = typename _TreeType::ValueOnCIter;
    using ValueOffIter        = typename _TreeType::ValueOffIter;
    using ValueOffCIter       = typename _TreeType::ValueOffCIter;
    using ValueAllIter        = typename _TreeType::ValueAllIter;
    using ValueAllCIter       = typename _TreeType::ValueAllCIter;

    using Accessor            = typename tree::ValueAccessor<_TreeType, true>;
    using ConstAccessor       = typename tree::ValueAccessor<const _TreeType, true>;
    using UnsafeAccessor      = typename tree::ValueAccessor<_TreeType, false>;
    using ConstUnsafeAccessor = typename tree::ValueAccessor<const _TreeType, false>;

    /// @brief ValueConverter<T>::Type is the type of a grid having the same
    /// hierarchy as this grid but a different value type, T.
    ///
    /// For example, FloatGrid::ValueConverter<double>::Type is equivalent to DoubleGrid.
    /// @note If the source grid type is a template argument, it might be necessary
    /// to write "typename SourceGrid::template ValueConverter<T>::Type".
    template<typename OtherValueType>
    struct ValueConverter {
        using Type = Grid<typename TreeType::template ValueConverter<OtherValueType>::Type>;
    };

    /// Return a new grid with the given background value.
    static Ptr create(const ValueType& background);
    /// Return a new grid with background value zero.
    static Ptr create();
    /// @brief Return a new grid that contains the given tree.
    /// @throw ValueError if the tree pointer is null
    static Ptr create(TreePtrType);
    /// @brief Return a new, empty grid with the same transform and metadata as the
    /// given grid and with background value zero.
    static Ptr create(const GridBase& other);


    /// Construct a new grid with background value zero.
    Grid();
    /// Construct a new grid with the given background value.
    explicit Grid(const ValueType& background);
    /// @brief Construct a new grid that shares the given tree and associates with it
    /// an identity linear transform.
    /// @throw ValueError if the tree pointer is null
    explicit Grid(TreePtrType);
    /// Deep copy another grid's metadata, transform and tree.
    Grid(const Grid&);
    /// @brief Deep copy the metadata, transform and tree of another grid whose tree
    /// configuration is the same as this grid's but whose value type is different.
    /// Cast the other grid's values to this grid's value type.
    /// @throw TypeError if the other grid's tree configuration doesn't match this grid's
    /// or if this grid's ValueType is not constructible from the other grid's ValueType.
    template<typename OtherTreeType>
    explicit Grid(const Grid<OtherTreeType>&);
#if OPENVDB_ABI_VERSION_NUMBER <= 3
    /// Deep copy another grid's metadata, but share its tree and transform.
    Grid(const Grid&, ShallowCopy);
#else
    /// Deep copy another grid's metadata and transform, but share its tree.
    Grid(Grid&, ShallowCopy);
#endif
    /// @brief Deep copy another grid's metadata and transform, but construct a new tree
    /// with background value zero.
    explicit Grid(const GridBase&);

    ~Grid() override {}

    /// Disallow assignment, since it wouldn't be obvious whether the copy is deep or shallow.
    Grid& operator=(const Grid&) = delete;


#if OPENVDB_ABI_VERSION_NUMBER <= 3
    //@{
    /// @brief Return a new grid of the same type as this grid whose metadata
    /// is a deep copy of this grid's.
    /// @details If @a treePolicy is @c CP_NEW, the new grid is given a new, empty tree,
    /// and it shares its transform with this grid;
    /// if @c CP_SHARE, the new grid shares this grid's tree and transform;
    /// if @c CP_COPY, the new grid's tree and transform are deep copies of this grid's.
    Ptr copy(CopyPolicy treePolicy = CP_SHARE) const;
    GridBase::Ptr copyGrid(CopyPolicy treePolicy = CP_SHARE) const override;
    //@}
#else
    //@{
    /// @brief Return a new grid of the same type as this grid whose metadata and
    /// transform are deep copies of this grid's and whose tree is shared with this grid.
    Ptr copy();
    ConstPtr copy() const;
    //@}
    /// @brief Return a new grid of the same type as this grid whose metadata and
    /// transform are deep copies of this grid's and whose tree is default-constructed.
    Ptr copyWithNewTree() const;

    GridBase::Ptr copyGrid() override;
    GridBase::ConstPtr copyGrid() const override;
    GridBase::Ptr copyGridWithNewTree() const override;
#endif
    //@{
    /// Return a new grid whose metadata, transform and tree are deep copies of this grid's.
    Ptr deepCopy() const { return Ptr(new Grid(*this)); }
    GridBase::Ptr deepCopyGrid() const override { return this->deepCopy(); }
    //@}

    /// Return the name of this grid's type.
    Name type() const override { return this->gridType(); }
    /// Return the name of this type of grid.
    static Name gridType() { return TreeType::treeType(); }


    //
    // Voxel access methods
    //
    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d").
    Name valueType() const override { return tree().valueType(); }

    /// @brief Return this grid's background value.
    ///
    /// @note Use tools::changeBackground to efficiently modify the background values.
    const ValueType& background() const { return mTree->background(); }

    /// Return @c true if this grid contains only inactive background voxels.
    bool empty() const override { return tree().empty(); }
    /// Empty this grid, so that all voxels become inactive background voxels.
    void clear() override { tree().clear(); }

    /// @brief Return an accessor that provides random read and write access
    /// to this grid's voxels. The accessor is safe in the sense that
    /// it is registered by the tree of this grid.
    Accessor getAccessor() { return Accessor(tree()); }
    /// @brief Return an accessor that provides random read and write access
    /// to this grid's voxels. The accessor is unsafe in the sense that
    /// it is not registered by the tree of this grid. In some rare
    /// cases this can give a performance advantage over a registered
    /// accessor but it is unsafe if the tree topology is modified.
    ///
    /// @warning Only use this method if you're an expert and know the
    /// risks of using an unregistered accessor (see tree/ValueAccessor.h)
    UnsafeAccessor getUnsafeAccessor() { return UnsafeAccessor(tree()); }
    //@{
    /// Return an accessor that provides random read-only access to this grid's voxels.
    ConstAccessor getAccessor() const { return ConstAccessor(tree()); }
    ConstAccessor getConstAccessor() const { return ConstAccessor(tree()); }
    //@}
    /// @brief Return an accessor that provides random read-only access
    /// to this grid's voxels. The accessor is unsafe in the sense that
    /// it is not registered by the tree of this grid. In some rare
    /// cases this can give a performance advantage over a registered
    /// accessor but it is unsafe if the tree topology is modified.
    ///
    /// @warning Only use this method if you're an expert and know the
    /// risks of using an unregistered accessor (see tree/ValueAccessor.h)
    ConstUnsafeAccessor getConstUnsafeAccessor() const { return ConstUnsafeAccessor(tree()); }

    //@{
    /// Return an iterator over all of this grid's active values (tile and voxel).
    ValueOnIter   beginValueOn()       { return tree().beginValueOn(); }
    ValueOnCIter  beginValueOn() const { return tree().cbeginValueOn(); }
    ValueOnCIter cbeginValueOn() const { return tree().cbeginValueOn(); }
    //@}
    //@{
    /// Return an iterator over all of this grid's inactive values (tile and voxel).
    ValueOffIter   beginValueOff()       { return tree().beginValueOff(); }
    ValueOffCIter  beginValueOff() const { return tree().cbeginValueOff(); }
    ValueOffCIter cbeginValueOff() const { return tree().cbeginValueOff(); }
    //@}
    //@{
    /// Return an iterator over all of this grid's values (tile and voxel).
    ValueAllIter   beginValueAll()       { return tree().beginValueAll(); }
    ValueAllCIter  beginValueAll() const { return tree().cbeginValueAll(); }
    ValueAllCIter cbeginValueAll() const { return tree().cbeginValueAll(); }
    //@}

    /// Return the minimum and maximum active values in this grid.
    void evalMinMax(ValueType& minVal, ValueType& maxVal) const;

    //@{
    /// @brief Set all voxels within a given axis-aligned box to a constant value.
    /// @param bbox    inclusive coordinates of opposite corners of an axis-aligned box
    /// @param value   the value to which to set voxels within the box
    /// @param active  if true, mark voxels within the box as active,
    ///                otherwise mark them as inactive
    /// @note This operation generates a sparse, but not always optimally sparse,
    /// representation of the filled box.  Follow fill operations with a prune()
    /// operation for optimal sparseness.
    void sparseFill(const CoordBBox& bbox, const ValueType& value, bool active = true);
    void fill(const CoordBBox& bbox, const ValueType& value, bool active = true);
    //@}

    /// @brief Set all voxels within a given axis-aligned box to a constant value
    /// and ensure that those voxels are all represented at the leaf level.
    /// @param bbox    inclusive coordinates of opposite corners of an axis-aligned box.
    /// @param value   the value to which to set voxels within the box.
    /// @param active  if true, mark voxels within the box as active,
    ///                otherwise mark them as inactive.
    void denseFill(const CoordBBox& bbox, const ValueType& value, bool active = true);

    /// Reduce the memory footprint of this grid by increasing its sparseness.
    void pruneGrid(float tolerance = 0.0) override;

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// @brief Clip this grid to the given index-space bounding box.
    /// @details Voxels that lie outside the bounding box are set to the background.
    /// @warning Clipping a level set will likely produce a grid that is
    /// no longer a valid level set.
    void clip(const CoordBBox&) override;
#endif

    /// @brief Efficiently merge another grid into this grid using one of several schemes.
    /// @details This operation is primarily intended to combine grids that are mostly
    /// non-overlapping (for example, intermediate grids from computations that are
    /// parallelized across disjoint regions of space).
    /// @warning This operation always empties the other grid.
    void merge(Grid& other, MergePolicy policy = MERGE_ACTIVE_STATES);

    /// @brief Union this grid's set of active values with the active values
    /// of the other grid, whose value type may be different.
    /// @details The resulting state of a value is active if the corresponding value
    /// was already active OR if it is active in the other grid. Also, a resulting
    /// value maps to a voxel if the corresponding value already mapped to a voxel
    /// OR if it is a voxel in the other grid. Thus, a resulting value can only
    /// map to a tile if the corresponding value already mapped to a tile
    /// AND if it is a tile value in the other grid.
    ///
    /// @note This operation modifies only active states, not values.
    /// Specifically, active tiles and voxels in this grid are not changed, and
    /// tiles or voxels that were inactive in this grid but active in the other grid
    /// are marked as active in this grid but left with their original values.
    template<typename OtherTreeType>
    void topologyUnion(const Grid<OtherTreeType>& other);

    /// @brief Intersect this grid's set of active values with the active values
    /// of the other grid, whose value type may be different.
    /// @details The resulting state of a value is active only if the corresponding
    /// value was already active AND if it is active in the other tree. Also, a
    /// resulting value maps to a voxel if the corresponding value
    /// already mapped to an active voxel in either of the two grids
    /// and it maps to an active tile or voxel in the other grid.
    ///
    /// @note This operation can delete branches of this grid that overlap with
    /// inactive tiles in the other grid.  Also, because it can deactivate voxels,
    /// it can create leaf nodes with no active values.  Thus, it is recommended
    /// to prune this grid after calling this method.
    template<typename OtherTreeType>
    void topologyIntersection(const Grid<OtherTreeType>& other);

    /// @brief Difference this grid's set of active values with the active values
    /// of the other grid, whose value type may be different.
    /// @details After this method is called, voxels in this grid will be active
    /// only if they were active to begin with and if the corresponding voxels
    /// in the other grid were inactive.
    ///
    /// @note This operation can delete branches of this grid that overlap with
    /// active tiles in the other grid.  Also, because it can deactivate voxels,
    /// it can create leaf nodes with no active values.  Thus, it is recommended
    /// to prune this grid after calling this method.
    template<typename OtherTreeType>
    void topologyDifference(const Grid<OtherTreeType>& other);

    //
    // Statistics
    //
    /// Return the number of active voxels.
    Index64 activeVoxelCount() const override { return tree().activeVoxelCount(); }
    /// Return the axis-aligned bounding box of all active voxels.
    CoordBBox evalActiveVoxelBoundingBox() const override;
    /// Return the dimensions of the axis-aligned bounding box of all active voxels.
    Coord evalActiveVoxelDim() const override;

    /// Return the number of bytes of memory used by this grid.
    /// @todo Add transform().memUsage()
    Index64 memUsage() const override { return tree().memUsage(); }


    //
    // Tree methods
    //
    //@{
    /// @brief Return a pointer to this grid's tree, which might be
    /// shared with other grids.  The pointer is guaranteed to be non-null.
    TreePtrType treePtr() { return mTree; }
    ConstTreePtrType treePtr() const { return mTree; }
    ConstTreePtrType constTreePtr() const { return mTree; }
    TreeBase::ConstPtr constBaseTreePtr() const override { return mTree; }
    //@}
    //@{
    /// @brief Return a reference to this grid's tree, which might be
    /// shared with other grids.
    /// @note Calling setTree() on this grid invalidates all references
    /// previously returned by this method.
    TreeType& tree() { return *mTree; }
    const TreeType& tree() const { return *mTree; }
    const TreeType& constTree() const { return *mTree; }
    //@}

    /// @brief Associate the given tree with this grid, in place of its existing tree.
    /// @throw ValueError if the tree pointer is null
    /// @throw TypeError if the tree is not of type TreeType
    /// @note Invalidates all references previously returned by baseTree(),
    /// constBaseTree(), tree() or constTree().
    void setTree(TreeBase::Ptr) override;

    /// @brief Associate a new, empty tree with this grid, in place of its existing tree.
    /// @note The new tree has the same background value as the existing tree.
    void newTree() override;


    //
    // I/O methods
    //
    /// @brief Read the grid topology from a stream.
    /// This will read only the grid structure, not the actual data buffers.
    void readTopology(std::istream&) override;
    /// @brief Write the grid topology to a stream.
    /// This will write only the grid structure, not the actual data buffers.
    void writeTopology(std::ostream&) const override;

    /// Read all data buffers for this grid.
    void readBuffers(std::istream&) override;
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// Read all of this grid's data buffers that intersect the given index-space bounding box.
    void readBuffers(std::istream&, const CoordBBox&) override;
    /// @brief Read all of this grid's data buffers that are not yet resident in memory
    /// (because delayed loading is in effect).
    /// @details If this grid was read from a memory-mapped file, this operation
    /// disconnects the grid from the file.
    /// @sa io::File::open, io::MappedFile
    void readNonresidentBuffers() const override;
#endif
    /// Write out all data buffers for this grid.
    void writeBuffers(std::ostream&) const override;

    /// Output a human-readable description of this grid.
    void print(std::ostream& = std::cout, int verboseLevel = 1) const override;

    /// @brief Return @c true if grids of this type require multiple I/O passes
    /// to read and write data buffers.
    /// @sa HasMultiPassIO
    static inline bool hasMultiPassIO();


    //
    // Registry methods
    //
    /// Return @c true if this grid type is registered.
    static bool isRegistered() { return GridBase::isRegistered(Grid::gridType()); }
    /// Register this grid type along with a factory function.
    static void registerGrid()
    {
        GridBase::registerGrid(Grid::gridType(), Grid::factory);
        if (!tree::internal::LeafBufferFlags<ValueType>::IsAtomic) {
            OPENVDB_LOG_WARN("delayed loading of grids of type " << Grid::gridType()
                << " might not be threadsafe on this platform");
        }
    }
    /// Remove this grid type from the registry.
    static void unregisterGrid() { GridBase::unregisterGrid(Grid::gridType()); }


private:
    /// Helper function for use with registerGrid()
    static GridBase::Ptr factory() { return Grid::create(); }

    TreePtrType mTree;
}; // class Grid


////////////////////////////////////////


/// @brief Cast a generic grid pointer to a pointer to a grid of a concrete class.
///
/// Return a null pointer if the input pointer is null or if it
/// points to a grid that is not of type @c GridType.
///
/// @note Calling gridPtrCast<GridType>(grid) is equivalent to calling
/// GridBase::grid<GridType>(grid).
template<typename GridType>
inline typename GridType::Ptr
gridPtrCast(const GridBase::Ptr& grid)
{
    return GridBase::grid<GridType>(grid);
}


/// @brief Cast a generic const grid pointer to a const pointer to a grid
/// of a concrete class.
///
/// Return a null pointer if the input pointer is null or if it
/// points to a grid that is not of type @c GridType.
///
/// @note Calling gridConstPtrCast<GridType>(grid) is equivalent to calling
/// GridBase::constGrid<GridType>(grid).
template<typename GridType>
inline typename GridType::ConstPtr
gridConstPtrCast(const GridBase::ConstPtr& grid)
{
    return GridBase::constGrid<GridType>(grid);
}


////////////////////////////////////////


/// @{
/// @brief Return a pointer to a deep copy of the given grid, provided that
/// the grid's concrete type is @c GridType.
///
/// Return a null pointer if the input pointer is null or if it
/// points to a grid that is not of type @c GridType.
template<typename GridType>
inline typename GridType::Ptr
deepCopyTypedGrid(const GridBase::ConstPtr& grid)
{
    if (!grid || !grid->isType<GridType>()) return typename GridType::Ptr();
    return gridPtrCast<GridType>(grid->deepCopyGrid());
}


template<typename GridType>
inline typename GridType::Ptr
deepCopyTypedGrid(const GridBase& grid)
{
    if (!grid.isType<GridType>()) return typename GridType::Ptr();
    return gridPtrCast<GridType>(grid.deepCopyGrid());
}
/// @}


////////////////////////////////////////


//@{
/// @brief This adapter allows code that is templated on a Tree type to
/// accept either a Tree type or a Grid type.
template<typename _TreeType>
struct TreeAdapter
{
    using TreeType             = _TreeType;
    using NonConstTreeType     = typename std::remove_const<TreeType>::type;
    using TreePtrType          = typename TreeType::Ptr;
    using ConstTreePtrType     = typename TreeType::ConstPtr;
    using NonConstTreePtrType  = typename NonConstTreeType::Ptr;
    using GridType             = Grid<TreeType>;
    using NonConstGridType     = Grid<NonConstTreeType>;
    using GridPtrType          = typename GridType::Ptr;
    using NonConstGridPtrType  = typename NonConstGridType::Ptr;
    using ConstGridPtrType     = typename GridType::ConstPtr;
    using ValueType            = typename TreeType::ValueType;
    using AccessorType         = typename tree::ValueAccessor<TreeType>;
    using ConstAccessorType    = typename tree::ValueAccessor<const TreeType>;
    using NonConstAccessorType = typename tree::ValueAccessor<NonConstTreeType>;

    static TreeType& tree(TreeType& t) { return t; }
    static TreeType& tree(GridType& g) { return g.tree(); }
    static const TreeType& tree(const TreeType& t) { return t; }
    static const TreeType& tree(const GridType& g) { return g.tree(); }
    static const TreeType& constTree(TreeType& t) { return t; }
    static const TreeType& constTree(GridType& g) { return g.constTree(); }
    static const TreeType& constTree(const TreeType& t) { return t; }
    static const TreeType& constTree(const GridType& g) { return g.constTree(); }
};


/// Partial specialization for Grid types
template<typename _TreeType>
struct TreeAdapter<Grid<_TreeType> >
{
    using TreeType             = _TreeType;
    using NonConstTreeType     = typename std::remove_const<TreeType>::type;
    using TreePtrType          = typename TreeType::Ptr;
    using ConstTreePtrType     = typename TreeType::ConstPtr;
    using NonConstTreePtrType  = typename NonConstTreeType::Ptr;
    using GridType             = Grid<TreeType>;
    using NonConstGridType     = Grid<NonConstTreeType>;
    using GridPtrType          = typename GridType::Ptr;
    using NonConstGridPtrType  = typename NonConstGridType::Ptr;
    using ConstGridPtrType     = typename GridType::ConstPtr;
    using ValueType            = typename TreeType::ValueType;
    using AccessorType         = typename tree::ValueAccessor<TreeType>;
    using ConstAccessorType    = typename tree::ValueAccessor<const TreeType>;
    using NonConstAccessorType = typename tree::ValueAccessor<NonConstTreeType>;

    static TreeType& tree(TreeType& t) { return t; }
    static TreeType& tree(GridType& g) { return g.tree(); }
    static const TreeType& tree(const TreeType& t) { return t; }
    static const TreeType& tree(const GridType& g) { return g.tree(); }
    static const TreeType& constTree(TreeType& t) { return t; }
    static const TreeType& constTree(GridType& g) { return g.constTree(); }
    static const TreeType& constTree(const TreeType& t) { return t; }
    static const TreeType& constTree(const GridType& g) { return g.constTree(); }
};

/// Partial specialization for ValueAccessor types
template<typename _TreeType>
struct TreeAdapter<tree::ValueAccessor<_TreeType> >
{
    using TreeType             = _TreeType;
    using NonConstTreeType     = typename std::remove_const<TreeType>::type;
    using TreePtrType          = typename TreeType::Ptr;
    using ConstTreePtrType     = typename TreeType::ConstPtr;
    using NonConstTreePtrType  = typename NonConstTreeType::Ptr;
    using GridType             = Grid<TreeType>;
    using NonConstGridType     = Grid<NonConstTreeType>;
    using GridPtrType          = typename GridType::Ptr;
    using NonConstGridPtrType  = typename NonConstGridType::Ptr;
    using ConstGridPtrType     = typename GridType::ConstPtr;
    using ValueType            = typename TreeType::ValueType;
    using AccessorType         = typename tree::ValueAccessor<TreeType>;
    using ConstAccessorType    = typename tree::ValueAccessor<const TreeType>;
    using NonConstAccessorType = typename tree::ValueAccessor<NonConstTreeType>;

    static TreeType& tree(TreeType& t) { return t; }
    static TreeType& tree(GridType& g) { return g.tree(); }
    static TreeType& tree(AccessorType& a) { return a.tree(); }
    static const TreeType& tree(const TreeType& t) { return t; }
    static const TreeType& tree(const GridType& g) { return g.tree(); }
    static const TreeType& tree(const AccessorType& a) { return a.tree(); }
    static const TreeType& constTree(TreeType& t) { return t; }
    static const TreeType& constTree(GridType& g) { return g.constTree(); }
    static const TreeType& constTree(const TreeType& t) { return t; }
    static const TreeType& constTree(const GridType& g) { return g.constTree(); }
};

//@}


////////////////////////////////////////


/// @brief Metafunction that specifies whether a given leaf node, tree, or grid type
/// requires multiple passes to read and write voxel data
/// @details Multi-pass I/O allows one to optimize the data layout of leaf nodes
/// for certain access patterns during delayed loading.
/// @sa io::MultiPass
template<typename LeafNodeType>
struct HasMultiPassIO {
    static const bool value = std::is_base_of<io::MultiPass, LeafNodeType>::value;
};

// Partial specialization for Tree types
template<typename RootNodeType>
struct HasMultiPassIO<tree::Tree<RootNodeType>> {
    // A tree is multi-pass if its (root node's) leaf node type is multi-pass.
    static const bool value = HasMultiPassIO<typename RootNodeType::LeafNodeType>::value;
};

// Partial specialization for Grid types
template<typename TreeType>
struct HasMultiPassIO<Grid<TreeType>> {
    // A grid is multi-pass if its tree's leaf node type is multi-pass.
    static const bool value = HasMultiPassIO<typename TreeType::LeafNodeType>::value;
};


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
GridBase::grid(const GridBase::Ptr& grid)
{
    // The string comparison on type names is slower than a dynamic pointer cast, but
    // it is safer when pointers cross DSO boundaries, as they do in many Houdini nodes.
    if (grid && grid->type() == GridType::gridType()) {
        return StaticPtrCast<GridType>(grid);
    }
    return typename GridType::Ptr();
}


template<typename GridType>
inline typename GridType::ConstPtr
GridBase::grid(const GridBase::ConstPtr& grid)
{
    return ConstPtrCast<const GridType>(
        GridBase::grid<GridType>(ConstPtrCast<GridBase>(grid)));
}


template<typename GridType>
inline typename GridType::ConstPtr
GridBase::constGrid(const GridBase::Ptr& grid)
{
    return ConstPtrCast<const GridType>(GridBase::grid<GridType>(grid));
}


template<typename GridType>
inline typename GridType::ConstPtr
GridBase::constGrid(const GridBase::ConstPtr& grid)
{
    return ConstPtrCast<const GridType>(
        GridBase::grid<GridType>(ConstPtrCast<GridBase>(grid)));
}


inline TreeBase::Ptr
GridBase::baseTreePtr()
{
    return ConstPtrCast<TreeBase>(this->constBaseTreePtr());
}


inline void
GridBase::setTransform(math::Transform::Ptr xform)
{
    if (!xform) OPENVDB_THROW(ValueError, "Transform pointer is null");
    mTransform = xform;
}


////////////////////////////////////////


template<typename TreeT>
inline Grid<TreeT>::Grid(): mTree(new TreeType)
{
}


template<typename TreeT>
inline Grid<TreeT>::Grid(const ValueType &background): mTree(new TreeType(background))
{
}


template<typename TreeT>
inline Grid<TreeT>::Grid(TreePtrType tree): mTree(tree)
{
    if (!tree) OPENVDB_THROW(ValueError, "Tree pointer is null");
}


template<typename TreeT>
inline Grid<TreeT>::Grid(const Grid& other):
    GridBase(other),
    mTree(StaticPtrCast<TreeType>(other.mTree->copy()))
{
}


template<typename TreeT>
template<typename OtherTreeType>
inline Grid<TreeT>::Grid(const Grid<OtherTreeType>& other):
    GridBase(other),
    mTree(new TreeType(other.constTree()))
{
}


#if OPENVDB_ABI_VERSION_NUMBER <= 3
template<typename TreeT>
inline Grid<TreeT>::Grid(const Grid& other, ShallowCopy):
    GridBase(other, ShallowCopy()),
    mTree(other.mTree)
{
}
#else
template<typename TreeT>
inline Grid<TreeT>::Grid(Grid& other, ShallowCopy):
    GridBase(other),
    mTree(other.mTree)
{
}
#endif


template<typename TreeT>
inline Grid<TreeT>::Grid(const GridBase& other):
    GridBase(other),
    mTree(new TreeType)
{
}


//static
template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::create()
{
    return Grid::create(zeroVal<ValueType>());
}


//static
template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::create(const ValueType& background)
{
    return Ptr(new Grid(background));
}


//static
template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::create(TreePtrType tree)
{
    return Ptr(new Grid(tree));
}


//static
template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::create(const GridBase& other)
{
    return Ptr(new Grid(other));
}


////////////////////////////////////////


#if OPENVDB_ABI_VERSION_NUMBER <= 3

template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::copy(CopyPolicy treePolicy) const
{
    Ptr ret;
    switch (treePolicy) {
        case CP_NEW:
            ret.reset(new Grid(*this, ShallowCopy()));
            ret->newTree();
            break;
        case CP_COPY:
            ret.reset(new Grid(*this));
            break;
        case CP_SHARE:
            ret.reset(new Grid(*this, ShallowCopy()));
            break;
    }
    return ret;
}


template<typename TreeT>
inline GridBase::Ptr
Grid<TreeT>::copyGrid(CopyPolicy treePolicy) const
{
    return this->copy(treePolicy);
}

#else // if OPENVDB_ABI_VERSION_NUMBER > 3

template<typename TreeT>
inline typename Grid<TreeT>::ConstPtr
Grid<TreeT>::copy() const
{
    return ConstPtr{new Grid{*const_cast<Grid*>(this), ShallowCopy{}}};
}

template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::copy()
{
    return Ptr{new Grid{*this, ShallowCopy{}}};
}


template<typename TreeT>
inline typename Grid<TreeT>::Ptr
Grid<TreeT>::copyWithNewTree() const
{
    Ptr result{new Grid{*const_cast<Grid*>(this), ShallowCopy{}}};
    result->newTree();
    return result;
}


template<typename TreeT>
inline GridBase::Ptr
Grid<TreeT>::copyGrid()
{
    return this->copy();
}

template<typename TreeT>
inline GridBase::ConstPtr
Grid<TreeT>::copyGrid() const
{
    return this->copy();
}


template<typename TreeT>
inline GridBase::Ptr
Grid<TreeT>::copyGridWithNewTree() const
{
    return this->copyWithNewTree();
}

#endif


////////////////////////////////////////


template<typename TreeT>
inline void
Grid<TreeT>::setTree(TreeBase::Ptr tree)
{
    if (!tree) OPENVDB_THROW(ValueError, "Tree pointer is null");
    if (tree->type() != TreeType::treeType()) {
        OPENVDB_THROW(TypeError, "Cannot assign a tree of type "
            + tree->type() + " to a grid of type " + this->type());
    }
    mTree = StaticPtrCast<TreeType>(tree);
}


template<typename TreeT>
inline void
Grid<TreeT>::newTree()
{
    mTree.reset(new TreeType(this->background()));
}


////////////////////////////////////////


template<typename TreeT>
inline void
Grid<TreeT>::sparseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    tree().sparseFill(bbox, value, active);
}


template<typename TreeT>
inline void
Grid<TreeT>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    this->sparseFill(bbox, value, active);
}

template<typename TreeT>
inline void
Grid<TreeT>::denseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    tree().denseFill(bbox, value, active);
}

template<typename TreeT>
inline void
Grid<TreeT>::pruneGrid(float tolerance)
{
    this->tree().prune(ValueType(zeroVal<ValueType>() + tolerance));
}

#if OPENVDB_ABI_VERSION_NUMBER >= 3
template<typename TreeT>
inline void
Grid<TreeT>::clip(const CoordBBox& bbox)
{
    tree().clip(bbox);
}
#endif


template<typename TreeT>
inline void
Grid<TreeT>::merge(Grid& other, MergePolicy policy)
{
    tree().merge(other.tree(), policy);
}


template<typename TreeT>
template<typename OtherTreeType>
inline void
Grid<TreeT>::topologyUnion(const Grid<OtherTreeType>& other)
{
    tree().topologyUnion(other.tree());
}


template<typename TreeT>
template<typename OtherTreeType>
inline void
Grid<TreeT>::topologyIntersection(const Grid<OtherTreeType>& other)
{
    tree().topologyIntersection(other.tree());
}


template<typename TreeT>
template<typename OtherTreeType>
inline void
Grid<TreeT>::topologyDifference(const Grid<OtherTreeType>& other)
{
    tree().topologyDifference(other.tree());
}


////////////////////////////////////////


template<typename TreeT>
inline void
Grid<TreeT>::evalMinMax(ValueType& minVal, ValueType& maxVal) const
{
    tree().evalMinMax(minVal, maxVal);
}


template<typename TreeT>
inline CoordBBox
Grid<TreeT>::evalActiveVoxelBoundingBox() const
{
    CoordBBox bbox;
    tree().evalActiveVoxelBoundingBox(bbox);
    return bbox;
}


template<typename TreeT>
inline Coord
Grid<TreeT>::evalActiveVoxelDim() const
{
    Coord dim;
    const bool nonempty = tree().evalActiveVoxelDim(dim);
    return (nonempty ? dim : Coord());
}


////////////////////////////////////////


/// @internal Consider using the stream tagging mechanism (see io::Archive)
/// to specify the float precision, but note that the setting is per-grid.

template<typename TreeT>
inline void
Grid<TreeT>::readTopology(std::istream& is)
{
    tree().readTopology(is, saveFloatAsHalf());
}


template<typename TreeT>
inline void
Grid<TreeT>::writeTopology(std::ostream& os) const
{
    tree().writeTopology(os, saveFloatAsHalf());
}


template<typename TreeT>
inline void
Grid<TreeT>::readBuffers(std::istream& is)
{
    if (!hasMultiPassIO() || (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_MULTIPASS_IO)) {
        tree().readBuffers(is, saveFloatAsHalf());
    } else {
        uint16_t numPasses = 1;
        is.read(reinterpret_cast<char*>(&numPasses), sizeof(uint16_t));
        const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(is);
        assert(bool(meta));
        for (uint16_t passIndex = 0; passIndex < numPasses; ++passIndex) {
            uint32_t pass = (uint32_t(numPasses) << 16) | uint32_t(passIndex);
            meta->setPass(pass);
            tree().readBuffers(is, saveFloatAsHalf());
        }
    }
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3

/// @todo Refactor this and the readBuffers() above
/// once support for ABI 2 compatibility is dropped.
template<typename TreeT>
inline void
Grid<TreeT>::readBuffers(std::istream& is, const CoordBBox& bbox)
{
    if (!hasMultiPassIO() || (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_MULTIPASS_IO)) {
        tree().readBuffers(is, bbox, saveFloatAsHalf());
    } else {
        uint16_t numPasses = 1;
        is.read(reinterpret_cast<char*>(&numPasses), sizeof(uint16_t));
        const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(is);
        assert(bool(meta));
        for (uint16_t passIndex = 0; passIndex < numPasses; ++passIndex) {
            uint32_t pass = (uint32_t(numPasses) << 16) | uint32_t(passIndex);
            meta->setPass(pass);
            tree().readBuffers(is, saveFloatAsHalf());
        }
        // Cannot clip inside readBuffers() when using multiple passes,
        // so instead clip afterwards.
        tree().clip(bbox);
    }
}


template<typename TreeT>
inline void
Grid<TreeT>::readNonresidentBuffers() const
{
    tree().readNonresidentBuffers();
}

#endif


template<typename TreeT>
inline void
Grid<TreeT>::writeBuffers(std::ostream& os) const
{
    if (!hasMultiPassIO()) {
        tree().writeBuffers(os, saveFloatAsHalf());
    } else {
        // Determine how many leaf buffer passes are required for this grid
        const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(os);
        assert(bool(meta));
        uint16_t numPasses = 1;
        meta->setCountingPasses(true);
        meta->setPass(0);
        tree().writeBuffers(os, saveFloatAsHalf());
        numPasses = static_cast<uint16_t>(meta->pass());
        os.write(reinterpret_cast<const char*>(&numPasses), sizeof(uint16_t));
        meta->setCountingPasses(false);

        // Save out the data blocks of the grid.
        for (uint16_t passIndex = 0; passIndex < numPasses; ++passIndex) {
            uint32_t pass = (uint32_t(numPasses) << 16) | uint32_t(passIndex);
            meta->setPass(pass);
            tree().writeBuffers(os, saveFloatAsHalf());
        }
    }
}


//static
template<typename TreeT>
inline bool
Grid<TreeT>::hasMultiPassIO()
{
    return HasMultiPassIO<Grid>::value;
}


template<typename TreeT>
inline void
Grid<TreeT>::print(std::ostream& os, int verboseLevel) const
{
    tree().print(os, verboseLevel);

    if (metaCount() > 0) {
        os << "Additional metadata:" << std::endl;
        for (ConstMetaIterator it = beginMeta(), end = endMeta(); it != end; ++it) {
            os << "  " << it->first;
            if (it->second) {
                const std::string value = it->second->str();
                if (!value.empty()) os << ": " << value;
            }
            os << "\n";
        }
    }

    os << "Transform:" << std::endl;
    transform().print(os, /*indent=*/"  ");
    os << std::endl;
}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
createGrid(const typename GridType::ValueType& background)
{
    return GridType::create(background);
}


template<typename GridType>
inline typename GridType::Ptr
createGrid()
{
    return GridType::create();
}


template<typename TreePtrType>
inline typename Grid<typename TreePtrType::element_type>::Ptr
createGrid(TreePtrType tree)
{
    using TreeType = typename TreePtrType::element_type;
    return Grid<TreeType>::create(tree);
}


template<typename GridType>
typename GridType::Ptr
createLevelSet(Real voxelSize, Real halfWidth)
{
    using ValueType = typename GridType::ValueType;

    // GridType::ValueType is required to be a floating-point scalar.
    static_assert(std::is_floating_point<ValueType>::value,
        "level-set grids must be floating-point-valued");

    typename GridType::Ptr grid = GridType::create(
        /*background=*/static_cast<ValueType>(voxelSize * halfWidth));
    grid->setTransform(math::Transform::createLinearTransform(voxelSize));
    grid->setGridClass(GRID_LEVEL_SET);
    return grid;
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_GRID_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
