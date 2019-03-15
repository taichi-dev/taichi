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

/// @file MultiResGrid.h
///
/// @author Ken Museth
///
/// @warning This class is fairly new and as such has not seen a lot of
/// use in production. Please report any issues or request for new
/// features directly to ken.museth@dreamworks.com.
///
/// @brief Multi-resolution grid that contains LoD sequences of trees
/// with powers of two refinements.
///
/// @note While this class can arguably be used to implement a sparse
/// Multi-Grid solver it is currently intended as a means to
/// efficiently compute LoD levels for applications like rendering
///
/// @note Prolongation means interpolation from coarse -> fine
/// @note Restriction means interpolation (or remapping) from fine -> coarse
///
/// @todo Add option to define the level of the input grid (currenlty
/// 0) so as to allow for super-sampling.

#ifndef OPENVDB_TOOLS_MULTIRESGRID_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MULTIRESGRID_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/FiniteDifference.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Operators.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/Metadata.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/NodeManager.h>
#include "Interpolation.h"
#include "Morphology.h"
#include "Prune.h"
#include "SignedFloodFill.h"
#include "ValueTransformer.h"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

template<typename TreeType>
class MultiResGrid: public MetaMap
{
public:
    using Ptr = SharedPtr<MultiResGrid>;
    using ConstPtr = SharedPtr<const MultiResGrid>;

    using ValueType = typename TreeType::ValueType;
    using ValueOnCIter = typename TreeType::ValueOnCIter;
    using ValueOnIter = typename TreeType::ValueOnIter;
    using TreePtr = typename TreeType::Ptr;
    using ConstTreePtr = typename TreeType::ConstPtr;
    using GridPtr = typename Grid<TreeType>::Ptr;
    using ConstGridPtr = typename Grid<TreeType>::ConstPtr;

    //////////////////////////////////////////////////////////////////////

    /// @brief Constructor of empty grids
    /// @param levels The number of trees in this MultiResGrid
    /// @param background Background value
    /// @param voxelSize Size of a (uniform voxel). Defaults to one.
    /// @note The multiple grids are all empty.
    MultiResGrid(size_t levels, ValueType background, double voxelSize = 1.0);

    /// @brief Given an initial high-resolution grid this constructor
    /// generates all the coarser grids by means of restriction.
    /// @param levels The number of trees in this MultiResGrid
    /// @param grid High-resolution input grid
    /// @param useInjection Use restriction by injection, vs
    /// full-weighting. It defaults to false and should rarely be used.
    /// @note This constructor will perform a deep copy of the input
    /// grid and use it as the highest level grid.
    MultiResGrid(size_t levels, const Grid<TreeType> &grid, bool useInjection = false);

    /// @brief Given an initial high-resolution grid this constructor
    /// generates all the coarser grids by means of restriction.
    /// @param levels The number of trees in this MultiResGrid
    /// @param grid High-resolution input grid
    /// @param useInjection Use restriction by injection, vs
    /// full-weighting. It defaults to false and should rarely be used.
    /// @note This constructor will steal the input grid and use it
    /// as the highest level grid. On output the grid is empty.
    MultiResGrid(size_t levels, GridPtr grid, bool useInjection = false);

    //////////////////////////////////////////////////////////////////////

    /// @brief Return the number of levels, i.e. trees, in this MultiResGrid
    /// @note level 0 is the finest level and numLevels()-1 is the coarsest
    /// level.
    size_t numLevels() const { return mTrees.size(); }

    /// @brief Return the level of the finest grid (always 0)
    static size_t finestLevel() { return 0; }

    /// @brief Return the level of the coarsest grid, i.e. numLevels()-1
    size_t coarsestLevel() const { return mTrees.size()-1; }

    //////////////////////////////////////////////////////////////////////

    /// @brief Return a reference to the tree at the specified level
    /// @param level The level of the tree to be returned
    /// @note Level 0 is by definition the finest tree.
    TreeType& tree(size_t level);

    /// @brief Return a const reference to the tree at the specified level
    /// @param level The level of the tree to be returned
    /// @note Level 0 is by definition the finest tree.
    const TreeType& constTree(size_t level) const;

    /// @brief Return a shared pointer to the tree at the specified level
    /// @param level The level of the tree to be returned
    /// @note Level 0 is by definition the finest tree.
    TreePtr treePtr(size_t level);

    /// @brief Return a const shared pointer to the tree at the specified level
    /// @param level The level of the tree to be returned
    /// @note Level 0 is by definition the finest tree.
    ConstTreePtr constTreePtr(size_t level) const;

    /// @brief Return a reference to the tree at the finest level
    TreeType& finestTree() { return *mTrees.front(); }

    /// @brief Return a const reference to the tree at the finest level
    const TreeType& finestConstTree() const { return *mTrees.front(); }

    /// @brief Return a shared pointer to the tree at the finest level
    TreePtr finestTreePtr() { return mTrees.front(); }

    /// @brief Return a const shared pointer to the tree at the finest level
    ConstTreePtr finestConstTreePtr() const { return mTrees.front(); }

    /// @brief Return a reference to the tree at the coarsest level
    TreeType& coarsestTree() { return *mTrees.back(); }

    /// @brief Return a const reference to the tree at the coarsest level
    const TreeType& coarsestConstTree() const { return *mTrees.back(); }

    /// @brief Return a shared pointer to the tree at the coarsest level
    TreePtr coarsestTreePtr() { return mTrees.back(); }

    /// @brief Return a const shared pointer to the tree at the coarsest level
    ConstTreePtr coarsestConstTreePtr() const { return mTrees.back(); }

    //////////////////////////////////////////////////////////////////////

    /// @brief Return a shared pointer to the grid at the specified integer level
    /// @param level Integer level of the grid to be returned
    /// @note Level 0 is by definition the finest grid.
    GridPtr grid(size_t level);

    /// @brief Return a const shared pointer to the grid at the specified level
    /// @param level The level of the grid to be returned
    /// @note Level 0 is by definition the finest grid.
    ConstGridPtr grid(size_t level) const;

    /// @brief Return a shared pointer to a new grid at the specified
    /// floating-point level.
    /// @param level Floating-point level of the grid to be returned
    /// @param grainSize Grain size for the multi-threading
    /// @details Interpolation of the specified order is performed
    /// between the bracketing integer levels.
    /// @note Level 0 is by definition the finest grid.
    template<Index Order>
    GridPtr createGrid(float level, size_t grainSize = 1) const;

    /// @brief Return a shared pointer to a vector of all the base
    /// grids in this instance of the MultiResGrid.
    /// @brief This method is useful for I/O
    GridPtrVecPtr grids();

    /// @brief Return a const shared pointer to a vector of all the base
    /// grids in this instance of the MultiResGrid.
    /// @brief This method is useful for I/O
    GridCPtrVecPtr grids() const;

    //////////////////////////////////////////////////////////////////////

    //@{
    /// @brief Return a reference to the finest grid's transform, which might be
    /// shared with other grids.
    /// @note Calling setTransform() on this grid invalidates all references
    /// previously returned by this method.
    /// @warning The transform is relative to the finest level (=0) grid!
    math::Transform& transform() { return *mTransform; }
    const math::Transform& transform() const { return *mTransform; }
    const math::Transform& constTransform() const { return *mTransform; }
    //@}

    //////////////////////////////////////////////////////////////////////

    //@{
    /// @brief Return the floating-point index coordinate at out_level given
    /// the index coordinate in_xyz at in_level.
    static Vec3R xyz(const Coord& in_ijk, size_t in_level, size_t out_level);
    static Vec3R xyz(const Vec3R& in_xyz, size_t in_level, size_t out_level);
    static Vec3R xyz(const Vec3R& in_xyz, double in_level, double out_level);
    //@}

    //////////////////////////////////////////////////////////////////////



    //@{
    /// @brief Return the value at the specified  coordinate position using
    /// interpolation of the specified order into the tree at the out_level.
    ///
    /// @details First in_ijk is mapped from index space at in_level to
    /// out_level, and then a value is interpolated from the tree at out_level.
    ///
    /// @param in_ijk Index coordinate position relative to tree at in_level
    /// @param in_level Integer level of the input coordinate in_ijk
    /// @param out_level Integer level of the interpolated value
    template<Index Order>
    ValueType sampleValue(const Coord& in_ijk, size_t in_level, size_t out_level) const;
    template<Index Order>
    ValueType sampleValue(const Vec3R& in_ijk, size_t in_level, size_t out_level) const;
    //@}

    /// @brief Return the value at the specified integer coordinate position
    /// and level using interpolation of the specified order.
    /// @param ijk Integer coordinate position relative to the highest level (=0) grid
    /// @param level Floating-point level from which to interpolate the value.
    /// @brief Non-integer values of the level will use linear-interpolation
    /// between the neighboring integer levels.
    template<Index Order>
    ValueType sampleValue(const Coord& ijk, double level) const;

    /// @brief Return the value at the specified floating-point coordinate position
    /// and level using interpolation of the specified order.
    /// @param xyz Floating-point coordinate position relative to the highest level grid
    /// @param level Floating-point level from which to interpolate
    /// the value.
    /// @brief Non-integer values of the level will use linear-interpolation
    /// between the neighboring integer levels.
    template<Index Order>
    ValueType sampleValue(const Vec3R& xyz, double level) const;

    //////////////////////////////////////////////////////////////////////

    /// @brief Return the value at coordinate location in @a level tree
    /// from the coarser tree at @a level+1 using trilinear interpolation
    /// @param coords input coords relative to the fine tree at level
    /// @param level The fine level to receive values from the coarser
    /// level-1
    /// @note Prolongation means to interpolation from coarse -> fine
    ValueType prolongateVoxel(const Coord& coords, const size_t level) const;


    /// (coarse->fine) Populates all the active voxel values in a fine (@a level) tree
    /// from the coarse (@a level+1) tree using linear interpolation
    /// This transforms multiple values of the tree in parallel
    void prolongateActiveVoxels(size_t destlevel, size_t grainSize = 1);

    //////////////////////////////////////////////////////////////////////

    /// Populate a coordinate location in @a level (coarse) tree
    /// from the @a level-1 (fine) tree using trilinear interpolation
    /// input coords are relative to the mTree[level] (coarse)
    /// @note Restriction means remapping from fine -> coarse
    ValueType restrictVoxel(Coord ijk, const size_t level, bool useInjection = false) const;

    /// (fine->coarse) Populates all the active voxel values in the coarse (@a level) tree
    /// from the fine (@a level-1) tree using trilinear interpolation.
    /// For cell-centered data, this is equivalent to an average
    /// For vertex-centered data this is equivalent to transferring the data
    /// from the fine vertex directly above the coarse vertex.
    /// This transforms multiple values of the tree in parallel
    void restrictActiveVoxels(size_t destlevel, size_t grainSize = 1);

    /// Output a human-readable description of this MultiResGrid
    void print(std::ostream& = std::cout, int verboseLevel = 1) const;

    /// @brief Return a string with the name of this MultiResGrid
    std::string getName() const
    {
        if (Metadata::ConstPtr meta = (*this)[GridBase::META_GRID_NAME]) return meta->str();
        return "";
    }

    /// @brief Set the name of this MultiResGrid
    void setName(const std::string& name)
    {
        this->removeMeta(GridBase::META_GRID_NAME);
        this->insertMeta(GridBase::META_GRID_NAME, StringMetadata(name));
    }

    /// Return the class of volumetric data (level set, fog volume, etc.) stored in this grid.
    GridClass getGridClass() const
    {
        typename StringMetadata::ConstPtr s =
            this->getMetadata<StringMetadata>(GridBase::META_GRID_CLASS);
        return s ? GridBase::stringToGridClass(s->value()) : GRID_UNKNOWN;
    }

    /// Specify the class of volumetric data (level set, fog volume, etc.) stored in this grid.
    void setGridClass(GridClass cls)
    {
        this->insertMeta(GridBase::META_GRID_CLASS, StringMetadata(GridBase::gridClassToString(cls)));
    }

    /// Remove the setting specifying the class of this grid's volumetric data.
    void clearGridClass() { this->removeMeta(GridBase::META_GRID_CLASS); }

private:

    MultiResGrid(const MultiResGrid& other);//disallow copy construction
    MultiResGrid& operator=(const MultiResGrid& other);//disallow copy assignment

    // For optimal performance we disable registration of the ValueAccessor
    using Accessor = tree::ValueAccessor<TreeType, false>;
    using ConstAccessor = tree::ValueAccessor<const TreeType, false>;

    void topDownRestrict(bool useInjection);

    inline void initMeta();

    // Private struct that concurrently creates a mask of active voxel
    // in a coarse tree from the active voxels in a fine tree
    struct MaskOp;

    /// Private struct that performs multi-threaded restriction
    struct RestrictOp;

    /// Private struct that performs multi-threaded prolongation
    struct ProlongateOp;

    // Private struct that performs multi-threaded computation of grids a fraction levels
    template<Index Order>
    struct FractionOp;

    /// Private template struct that performs the actual multi-threading
    template<typename OpType> struct CookOp;

    // Array of shared pointer to trees, level 0 has the highest resolution.
    std::vector<TreePtr> mTrees;
    // Shared pointer to a transform associated with the finest level grid
    typename math::Transform::Ptr mTransform;
};// MultiResGrid

template<typename TreeType>
MultiResGrid<TreeType>::
MultiResGrid(size_t levels, ValueType background, double voxelSize)
    : mTrees(levels)
    , mTransform(math::Transform::createLinearTransform( voxelSize ))
{
    this->initMeta();
    for (size_t i=0; i<levels; ++i) mTrees[i] = TreePtr(new TreeType(background));
}

template<typename TreeType>
MultiResGrid<TreeType>::
MultiResGrid(size_t levels, const Grid<TreeType> &grid, bool useInjection)
    : MetaMap(grid)
    , mTrees(levels)
    , mTransform( grid.transform().copy() )
{
    this->initMeta();
    mTrees[0].reset( new TreeType( grid.tree() ) );// deep copy input tree
    mTrees[0]->voxelizeActiveTiles();
    this->topDownRestrict(useInjection);
}

template<typename TreeType>
MultiResGrid<TreeType>::
MultiResGrid(size_t levels, GridPtr grid, bool useInjection)
    : MetaMap(*grid)
    , mTrees(levels)
    , mTransform( grid->transform().copy() )
{
    this->initMeta();
    mTrees[0] = grid->treePtr();// steal tree from input grid
    mTrees[0]->voxelizeActiveTiles();
    grid->newTree();
    this->topDownRestrict(useInjection);
}

template<typename TreeType>
inline TreeType& MultiResGrid<TreeType>::
tree(size_t level)
{
    assert( level < mTrees.size() );
    return *mTrees[level];
}

template<typename TreeType>
inline const TreeType& MultiResGrid<TreeType>::
constTree(size_t level) const
{
    assert( level < mTrees.size() );
    return *mTrees[level];
}

template<typename TreeType>
inline typename TreeType::Ptr MultiResGrid<TreeType>::
treePtr(size_t level)
{
    assert( level < mTrees.size() );
    return mTrees[level];
}

template<typename TreeType>
inline typename TreeType::ConstPtr MultiResGrid<TreeType>::
constTreePtr(size_t level) const
{
    assert( level < mTrees.size() );
    return mTrees[level];
}

template<typename TreeType>
typename Grid<TreeType>::Ptr MultiResGrid<TreeType>::
grid(size_t level)
{
    typename Grid<TreeType>::Ptr grid = Grid<TreeType>::create(this->treePtr(level));
    math::Transform::Ptr xform = mTransform->copy();
    if (level>0) xform->preScale( Real(1 << level) );
    grid->setTransform( xform );
    grid->insertMeta( *this->copyMeta() );
    grid->insertMeta( "MultiResGrid_Level", Int64Metadata(level));
    std::stringstream ss;
    ss << this->getName() << "_level_" << level;
    grid->setName( ss.str() );
    return grid;
}

template<typename TreeType>
inline typename Grid<TreeType>::ConstPtr MultiResGrid<TreeType>::
grid(size_t level) const
{
    return const_cast<MultiResGrid*>(this)->grid(level);
}

template<typename TreeType>
template<Index Order>
typename Grid<TreeType>::Ptr MultiResGrid<TreeType>::
createGrid(float level, size_t grainSize) const
{
    assert( level >= 0.0f && level <= float(mTrees.size()-1) );

    typename Grid<TreeType>::Ptr grid(new Grid<TreeType>(this->constTree(0).background()));
    math::Transform::Ptr xform = mTransform->copy();
    xform->preScale( math::Pow(2.0f, level) );
    grid->setTransform( xform );
    grid->insertMeta( *(this->copyMeta()) );
    grid->insertMeta( "MultiResGrid_Level", FloatMetadata(level) );
    std::stringstream ss;
    ss << this->getName() << "_level_" << level;
    grid->setName( ss.str() );

    if ( size_t(floorf(level)) == size_t(ceilf(level)) ) {
        grid->setTree( this->constTree( size_t(floorf(level))).copy() );
    } else {
        FractionOp<Order> tmp(*this, grid->tree(), level, grainSize);
        if ( grid->getGridClass() == GRID_LEVEL_SET ) {
            signedFloodFill( grid->tree() );
            pruneLevelSet( grid->tree() );//only creates inactive tiles
        }
    }

    return grid;
}

template<typename TreeType>
GridPtrVecPtr MultiResGrid<TreeType>::
grids()
{
    GridPtrVecPtr grids( new GridPtrVec );
    for (size_t level=0; level<mTrees.size(); ++level) grids->push_back(this->grid(level));
    return grids;
}

template<typename TreeType>
GridCPtrVecPtr MultiResGrid<TreeType>::
grids() const
{
    GridCPtrVecPtr grids( new GridCPtrVec );
    for (size_t level=0; level<mTrees.size(); ++level) grids->push_back(this->grid(level));
    return grids;
}

template<typename TreeType>
Vec3R MultiResGrid<TreeType>::
xyz(const Coord& in_ijk, size_t in_level, size_t out_level)
{
    return Vec3R( in_ijk.data() ) * Real(1 << in_level) / Real(1 << out_level);
}

template<typename TreeType>
Vec3R MultiResGrid<TreeType>::
xyz(const Vec3R& in_xyz, size_t in_level, size_t out_level)
{
    return in_xyz * Real(1 << in_level) / Real(1 << out_level);
}

template<typename TreeType>
Vec3R MultiResGrid<TreeType>::
xyz(const Vec3R& in_xyz, double in_level, double out_level)
{
    return in_xyz * math::Pow(2.0, in_level - out_level);

}

template<typename TreeType>
template<Index Order>
typename TreeType::ValueType MultiResGrid<TreeType>::
sampleValue(const Coord& in_ijk, size_t in_level, size_t out_level) const
{
    assert( in_level  >= 0 && in_level  < mTrees.size() );
    assert( out_level >= 0 && out_level < mTrees.size() );
    const ConstAccessor acc(*mTrees[out_level]);// has disabled registration!
    return tools::Sampler<Order>::sample( acc, this->xyz(in_ijk, in_level, out_level) );
}

template<typename TreeType>
template<Index Order>
typename TreeType::ValueType MultiResGrid<TreeType>::
sampleValue(const Vec3R& in_xyz, size_t in_level, size_t out_level) const
{
    assert( in_level  >= 0 && in_level  < mTrees.size() );
    assert( out_level >= 0 && out_level < mTrees.size() );
    const ConstAccessor acc(*mTrees[out_level]);// has disabled registration!
    return tools::Sampler<Order>::sample( acc, this->xyz(in_xyz, in_level, out_level) );
}

template<typename TreeType>
template<Index Order>
typename TreeType::ValueType MultiResGrid<TreeType>::
sampleValue(const Coord& ijk, double level) const
{
    assert( level >= 0.0 && level <= double(mTrees.size()-1) );
    const size_t level0 = size_t(floor(level)), level1 = size_t(ceil(level));
    const ValueType v0 = this->template sampleValue<Order>( ijk, 0, level0 );
    if ( level0 == level1 ) return v0;
    assert( level1 - level0 == 1 );
    const ValueType v1 = this->template sampleValue<Order>( ijk, 0, level1 );
    const ValueType a = ValueType(level1 - level);
    return a * v0 + (ValueType(1) - a) * v1;
}

template<typename TreeType>
template<Index Order>
typename TreeType::ValueType MultiResGrid<TreeType>::
sampleValue(const Vec3R& xyz, double level) const
{
    assert( level >= 0.0 && level <= double(mTrees.size()-1) );
    const size_t level0 = size_t(floor(level)), level1 = size_t(ceil(level));
    const ValueType v0 = this->template sampleValue<Order>( xyz, 0, level0 );
    if ( level0 == level1 ) return v0;
    assert( level1 - level0 == 1 );
    const ValueType v1 = this->template sampleValue<Order>( xyz, 0, level1 );
    const ValueType a = ValueType(level1 - level);
    return a * v0 + (ValueType(1) - a) * v1;
}

template<typename TreeType>
typename TreeType::ValueType MultiResGrid<TreeType>::
prolongateVoxel(const Coord& ijk, const size_t level) const
{
    assert( level+1 < mTrees.size() );
    const ConstAccessor acc(*mTrees[level + 1]);// has disabled registration!
    return ProlongateOp::run(ijk, acc);
}

template<typename TreeType>
void MultiResGrid<TreeType>::
prolongateActiveVoxels(size_t destlevel, size_t grainSize)
{
    assert( destlevel < mTrees.size()-1 );
    TreeType &fineTree = *mTrees[ destlevel ];
    const TreeType &coarseTree = *mTrees[ destlevel+1 ];
    CookOp<ProlongateOp> tmp( coarseTree, fineTree, grainSize );
}

template<typename TreeType>
typename TreeType::ValueType MultiResGrid<TreeType>::
restrictVoxel(Coord ijk, const size_t destlevel, bool useInjection) const
{
    assert( destlevel > 0 && destlevel < mTrees.size() );
    const TreeType &fineTree = *mTrees[ destlevel-1 ];
    if ( useInjection ) return fineTree.getValue(ijk<<1);
    const ConstAccessor acc( fineTree );// has disabled registration!
    return RestrictOp::run( ijk, acc);
}

template<typename TreeType>
void MultiResGrid<TreeType>::
restrictActiveVoxels(size_t destlevel, size_t grainSize)
{
    assert( destlevel > 0 && destlevel < mTrees.size() );
    const TreeType &fineTree = *mTrees[ destlevel-1 ];
    TreeType &coarseTree = *mTrees[ destlevel ];
    CookOp<RestrictOp> tmp( fineTree, coarseTree, grainSize );
}

template<typename TreeType>
void MultiResGrid<TreeType>::
print(std::ostream& os, int verboseLevel) const
{
    os << "MultiResGrid with " << mTrees.size() << " levels\n";
    for (size_t i=0; i<mTrees.size(); ++i) {
        os << "Level " << i << ": ";
        mTrees[i]->print(os, verboseLevel);
    }

    if ( MetaMap::metaCount() > 0) {
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

template<typename TreeType>
void MultiResGrid<TreeType>::
initMeta()
{
    const size_t levels = this->numLevels();
    if (levels < 2) {
        OPENVDB_THROW(ValueError, "MultiResGrid: at least two levels are required");
    }
    this->insertMeta("MultiResGrid_Levels", Int64Metadata( levels ) );
}

template<typename TreeType>
void MultiResGrid<TreeType>::
topDownRestrict(bool useInjection)
{
    const bool isLevelSet = this->getGridClass() == GRID_LEVEL_SET;
    for (size_t n=1; n<mTrees.size(); ++n) {
        const TreeType &fineTree = *mTrees[n-1];
        mTrees[n] = TreePtr(new TreeType( fineTree.background() ) );// empty tree
        TreeType &coarseTree = *mTrees[n];
        if (useInjection) {// Restriction by injection
            for (ValueOnCIter it = fineTree.cbeginValueOn(); it; ++it) {
                const Coord ijk = it.getCoord();
                if ( (ijk[0] & 1) || (ijk[1] & 1) || (ijk[2] & 1) ) continue;
                coarseTree.setValue( ijk >> 1, *it );
            }
        } else {// Restriction by full-weighting
            MaskOp tmp(fineTree, coarseTree, 128);
            this->restrictActiveVoxels(n, 64);
        }
        if ( isLevelSet ) {
            tools::signedFloodFill( coarseTree );
            tools::pruneLevelSet( coarseTree );//only creates inactive tiles
        }
    }// loop over grid levels
}

template<typename TreeType>
struct MultiResGrid<TreeType>::MaskOp
{
    using MaskT = typename TreeType::template ValueConverter<ValueMask>::Type;
    using PoolType = tbb::enumerable_thread_specific<TreeType>;
    using ManagerT = tree::LeafManager<const MaskT>;
    using RangeT = typename ManagerT::LeafRange;
    using VoxelIterT = typename ManagerT::LeafNodeType::ValueOnCIter;

    MaskOp(const TreeType& fineTree, TreeType& coarseTree, size_t grainSize = 1)
        : mPool(new PoolType( coarseTree ) )// empty coarse tree acts as examplar
    {
        assert( coarseTree.empty() );

        // Create Mask of restruction performed on fineTree
        MaskT mask(fineTree, false, true, TopologyCopy() );

        // Muli-threaded dilation which also linearizes the tree to leaf nodes
        tools::dilateActiveValues(mask, 1, NN_FACE_EDGE_VERTEX, EXPAND_TILES);

        // Restriction by injection using thread-local storage of coarse tree masks
        ManagerT leafs( mask );
        tbb::parallel_for(leafs.leafRange( grainSize ), *this);

        // multithreaded union of thread-local coarse tree masks with the coarse tree
        using IterT = typename PoolType::const_iterator;
        for (IterT it=mPool->begin(); it!=mPool->end(); ++it) coarseTree.topologyUnion( *it );
        delete mPool;
    }
    void operator()(const RangeT& range) const
    {
        Accessor coarseAcc( mPool->local() );// disabled registration
        for (typename RangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                Coord ijk = voxelIter.getCoord();
                if ( (ijk[2] & 1) || (ijk[1] & 1) || (ijk[0] & 1) ) continue;//no overlap
                coarseAcc.setValueOn( ijk >> 1 );//injection from fine to coarse level
            }//loop over active voxels in the fine tree
        }// loop over leaf nodes in the fine tree
    }
    PoolType* mPool;
};// MaskOp

template<typename TreeType>
template<Index Order>
struct MultiResGrid<TreeType>::FractionOp
{
    using MaskT = typename TreeType::template ValueConverter<ValueMask>::Type;
    using PoolType = tbb::enumerable_thread_specific<MaskT>;
    using PoolIterT = typename PoolType::iterator;
    using Manager1 = tree::LeafManager<const TreeType>;
    using Manager2 = tree::LeafManager<TreeType>;
    using Range1 = typename Manager1::LeafRange;
    using Range2 = typename Manager2::LeafRange;

    FractionOp(const MultiResGrid& parent,
               TreeType& midTree,
               float level,
               size_t grainSize = 1)
        : mLevel( level )
        , mPool(nullptr)
        , mTree0( &*(parent.mTrees[size_t(floorf(level))]) )//high-resolution
        , mTree1( &*(parent.mTrees[size_t(ceilf(level))]) ) //low-resolution
    {
        assert( midTree.empty() );
        assert( mTree0 != mTree1 );

        // Create a pool of  thread-local masks
        MaskT examplar( false );
        mPool = new PoolType( examplar );

        {// create mask from re-mapping coarse tree to mid-level tree
            tree::LeafManager<const TreeType> manager( *mTree1 );
            tbb::parallel_for( manager.leafRange(grainSize), *this );
        }

        // Multi-threaded dilation of mask
        tbb::parallel_for(tbb::blocked_range<PoolIterT>(mPool->begin(),mPool->end(),1), *this);

        // Union thread-local coarse tree masks into the coarse tree
        for (PoolIterT it=mPool->begin(); it!=mPool->end(); ++it) midTree.topologyUnion( *it );
        delete mPool;

        {// Interpolate values into the static mid level tree
            Manager2 manager( midTree );
            tbb::parallel_for(manager.leafRange(grainSize), *this);
        }
    }
    void operator()(const Range1& range) const
    {
        using VoxelIter = typename Manager1::LeafNodeType::ValueOnCIter;
        // Let mLevel = level + frac, where
        // level is integer part of mLevel and frac is the fractional part
        // low-res voxel size in world units = dx1 = 2^(level + 1)
        // mid-res voxel size in world units = dx  = 2^(mLevel) = 2^(level + frac)
        // low-res index -> world: ijk * dx1
        // world -> mid-res index: world / dx
        // low-res index -> mid-res index: (ijk * dx1) / dx = ijk * scale where
        // scale = dx1/dx = 2^(level+1)/2^(level+frac) = 2^(1-frac)
        const float scale = math::Pow(2.0f, 1.0f - math::FractionalPart(mLevel));
        tree::ValueAccessor<MaskT, false>  acc( mPool->local() );// disabled registration
        for (typename Range1::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            for (VoxelIter voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                Coord ijk = voxelIter.getCoord();
                ijk[0] = int(math::Round(ijk[0] * scale));
                ijk[1] = int(math::Round(ijk[1] * scale));
                ijk[2] = int(math::Round(ijk[2] * scale));
                acc.setValueOn( ijk );
            }//loop over active voxels in the fine tree
        }// loop over leaf nodes in the fine tree
    }
    void operator()(const tbb::blocked_range<PoolIterT>& range) const
    {
        for (PoolIterT it=range.begin(); it!=range.end(); ++it) {
            tools::dilateVoxels( *it, 1, NN_FACE_EDGE_VERTEX);
        }
    }
    void operator()(const Range2 &r) const
    {
        using VoxelIter = typename TreeType::LeafNodeType::ValueOnIter;
        // Let mLevel = level + frac, where
        // level is integer part of mLevel and frac is the fractional part
        // high-res voxel size in world units = dx0 = 2^(level)
        // low-res voxel size in world units = dx1 = 2^(level+1)
        // mid-res voxel size in world units = dx  = 2^(mLevel) = 2^(level + frac)
        // mid-res index -> world: ijk * dx
        // world -> high-res index: world / dx0
        // world -> low-res index: world / dx1
        // mid-res index -> high-res index: (ijk * dx) / dx0 = ijk * scale0 where
        // scale0 = dx/dx0 = 2^(level+frac)/2^(level) = 2^(frac)
        // mid-res index -> low-res index: (ijk * dx) / dx1 = ijk * scale1 where
        // scale1 = dx/dx1 = 2^(level+frac)/2^(level+1) = 2^(frac-1)
        const float b = math::FractionalPart(mLevel), a = 1.0f - b;
        const float scale0 = math::Pow( 2.0f, b );
        const float scale1 = math::Pow( 2.0f,-a );
        ConstAccessor acc0( *mTree0 ), acc1( *mTree1 );
        for (typename Range2::Iterator leafIter = r.begin(); leafIter; ++leafIter) {
            for (VoxelIter voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                const Vec3R xyz =  Vec3R( voxelIter.getCoord().data() );// mid level coord
                const ValueType v0 = tools::Sampler<Order>::sample( acc0, xyz * scale0 );
                const ValueType v1 = tools::Sampler<Order>::sample( acc1, xyz * scale1 );
                voxelIter.setValue( ValueType(a*v0 + b*v1) );
            }
        }
    }
    const float mLevel;
    PoolType* mPool;
    const TreeType *mTree0, *mTree1;
};// FractionOp


template<typename TreeType>
template<typename OperatorType>
struct MultiResGrid<TreeType>::CookOp
{
    using ManagerT = tree::LeafManager<TreeType>;
    using RangeT = typename ManagerT::LeafRange;

    CookOp(const TreeType& srcTree, TreeType& dstTree, size_t grainSize): acc(srcTree)
    {
        ManagerT leafs(dstTree);
        tbb::parallel_for(leafs.leafRange(grainSize), *this);
    }
    CookOp(const CookOp &other): acc(other.acc.tree()) {}

    void operator()(const RangeT& range) const
    {
        for (auto leafIt = range.begin(); leafIt; ++leafIt) {
            auto& phi = leafIt.buffer(0);
            for (auto voxelIt = leafIt->beginValueOn(); voxelIt; ++voxelIt) {
                phi.setValue(voxelIt.pos(), OperatorType::run(voxelIt.getCoord(), acc));
            }
        }
    }

    const ConstAccessor acc;
};// CookOp


template<typename TreeType>
struct MultiResGrid<TreeType>::RestrictOp
{
    /// @brief Static method that performs restriction by full weighting
    /// @param ijk Coordinate location on the coarse tree
    /// @param acc ValueAccessor to the fine tree
    static ValueType run(Coord ijk, const ConstAccessor &acc)
    {
        ijk <<= 1;
        // Overlapping grid point
        ValueType v = 8*acc.getValue(ijk);
        // neighbors in one axial direction
        v += 4*(acc.getValue(ijk.offsetBy(-1, 0, 0)) + acc.getValue(ijk.offsetBy( 1, 0, 0)) +// x
                acc.getValue(ijk.offsetBy( 0,-1, 0)) + acc.getValue(ijk.offsetBy( 0, 1, 0)) +// y
                acc.getValue(ijk.offsetBy( 0, 0,-1)) + acc.getValue(ijk.offsetBy( 0, 0, 1)));// z
        // neighbors in two axial directions
        v += 2*(acc.getValue(ijk.offsetBy(-1,-1, 0)) + acc.getValue(ijk.offsetBy(-1, 1, 0)) +// xy
                acc.getValue(ijk.offsetBy( 1,-1, 0)) + acc.getValue(ijk.offsetBy( 1, 1, 0)) +// xy
                acc.getValue(ijk.offsetBy(-1, 0,-1)) + acc.getValue(ijk.offsetBy(-1, 0, 1)) +// xz
                acc.getValue(ijk.offsetBy( 1, 0,-1)) + acc.getValue(ijk.offsetBy( 1, 0, 1)) +// xz
                acc.getValue(ijk.offsetBy( 0,-1,-1)) + acc.getValue(ijk.offsetBy( 0,-1, 1)) +// yz
                acc.getValue(ijk.offsetBy( 0, 1,-1)) + acc.getValue(ijk.offsetBy( 0, 1, 1)));// yz
        // neighbors in three axial directions
        for (int i=-1; i<=1; i+=2) {
            for (int j=-1; j<=1; j+=2) {
                for (int k=-1; k<=1; k+=2) v += acc.getValue(ijk.offsetBy(i,j,k));// xyz
            }
        }
        v *= ValueType(1.0f/64.0f);
        return v;
    }
};// RestrictOp

template<typename TreeType>
struct MultiResGrid<TreeType>::ProlongateOp
{
    /// @brief Interpolate values from a coarse grid (acc) into the index space (ijk) of a fine grid
    /// @param ijk Coordinate location on the fine tree
    /// @param acc ValueAccessor to the coarse tree
    static ValueType run(const Coord& ijk, const ConstAccessor &acc)
    {
        switch ( (ijk[0] & 1) | ((ijk[1] & 1) << 1) | ((ijk[2] & 1) << 2) ) {
        case 0:// all even
            return acc.getValue(ijk>>1);
        case 1:// x is odd
            return ValueType(0.5)*(acc.getValue(ijk.offsetBy(-1,0,0)>>1) +
                                   acc.getValue(ijk.offsetBy( 1,0,0)>>1));
        case 2:// y is odd
            return ValueType(0.5)*(acc.getValue(ijk.offsetBy(0,-1,0)>>1) +
                                   acc.getValue(ijk.offsetBy(0, 1,0)>>1));
        case 3:// x&y are odd
            return ValueType(0.25)*(acc.getValue(ijk.offsetBy(-1,-1,0)>>1) +
                                    acc.getValue(ijk.offsetBy(-1, 1,0)>>1) +
                                    acc.getValue(ijk.offsetBy( 1,-1,0)>>1) +
                                    acc.getValue(ijk.offsetBy( 1, 1,0)>>1));
        case 4:// z is odd
            return ValueType(0.5)*(acc.getValue(ijk.offsetBy(0,0,-1)>>1) +
                                   acc.getValue(ijk.offsetBy(0,0, 1)>>1));
        case 5:// x&z are odd
            return ValueType(0.25)*(acc.getValue(ijk.offsetBy(-1,0,-1)>>1) +
                                    acc.getValue(ijk.offsetBy(-1,0, 1)>>1) +
                                    acc.getValue(ijk.offsetBy( 1,0,-1)>>1) +
                                    acc.getValue(ijk.offsetBy( 1,0, 1)>>1));
        case 6:// y&z are odd
            return ValueType(0.25)*(acc.getValue(ijk.offsetBy(0,-1,-1)>>1) +
                                    acc.getValue(ijk.offsetBy(0,-1, 1)>>1) +
                                    acc.getValue(ijk.offsetBy(0, 1,-1)>>1) +
                                    acc.getValue(ijk.offsetBy(0, 1, 1)>>1));
        }
        // all are odd
        ValueType v = zeroVal<ValueType>();
        for (int i=-1; i<=1; i+=2) {
            for (int j=-1; j<=1; j+=2) {
                for (int k=-1; k<=1; k+=2) v += acc.getValue(ijk.offsetBy(i,j,k)>>1);// xyz
            }
        }
        return ValueType(0.125) * v;
    }
};// ProlongateOp

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MULTIRESGRID_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
