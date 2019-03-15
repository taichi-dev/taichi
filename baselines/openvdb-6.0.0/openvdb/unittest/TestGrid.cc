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

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/util/Name.h>
#include <openvdb/math/Transform.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/util/CpuTimer.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);

class TestGrid: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestGrid);
    CPPUNIT_TEST(testGridRegistry);
    CPPUNIT_TEST(testConstPtr);
    CPPUNIT_TEST(testGetGrid);
    CPPUNIT_TEST(testIsType);
    CPPUNIT_TEST(testTransform);
    CPPUNIT_TEST(testCopyGrid);
    CPPUNIT_TEST(testValueConversion);
    CPPUNIT_TEST(testClipping);
    CPPUNIT_TEST_SUITE_END();

    void testGridRegistry();
    void testConstPtr();
    void testGetGrid();
    void testIsType();
    void testTransform();
    void testCopyGrid();
    void testValueConversion();
    void testClipping();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGrid);


////////////////////////////////////////


class ProxyTree: public openvdb::TreeBase
{
public:
    using ValueType = int;
    using BuildType = int;
    using LeafNodeType = void;
    using ValueAllCIter = void;
    using ValueAllIter = void;
    using ValueOffCIter = void;
    using ValueOffIter = void;
    using ValueOnCIter = void;
    using ValueOnIter = void;
    using TreeBasePtr = openvdb::TreeBase::Ptr;
    using Ptr = openvdb::SharedPtr<ProxyTree>;
    using ConstPtr = openvdb::SharedPtr<const ProxyTree>;

    static const openvdb::Index DEPTH;
    static const ValueType backg;

    ProxyTree() {}
    ProxyTree(const ValueType&) {}
    ProxyTree(const ProxyTree&) = default;
    ~ProxyTree() override = default;

    static const openvdb::Name& treeType() { static const openvdb::Name s("proxy"); return s; }
    const openvdb::Name& type() const override { return treeType(); }
    openvdb::Name valueType() const override { return "proxy"; }
    const ValueType& background() const { return backg; }

    TreeBasePtr copy() const override { return TreeBasePtr(new ProxyTree(*this)); }

    void readTopology(std::istream& is, bool = false) override { is.seekg(0, std::ios::beg); }
    void writeTopology(std::ostream& os, bool = false) const override { os.seekp(0); }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    void readBuffers(std::istream& is,
        const openvdb::CoordBBox&, bool /*saveFloatAsHalf*/=false) override { is.seekg(0); }
    void readNonresidentBuffers() const override {}
#endif
    void readBuffers(std::istream& is, bool /*saveFloatAsHalf*/=false) override { is.seekg(0); }
    void writeBuffers(std::ostream& os, bool /*saveFloatAsHalf*/=false) const override
        { os.seekp(0, std::ios::beg); }

    bool empty() const { return true; }
    void clear() {}
    void prune(const ValueType& = 0) {}
    void clip(const openvdb::CoordBBox&) {}
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    void clipUnallocatedNodes() override {}
#endif
#if OPENVDB_ABI_VERSION_NUMBER >= 4
    openvdb::Index32 unallocatedLeafCount() const override { return 0; }
#endif

    void getIndexRange(openvdb::CoordBBox&) const override {}
    bool evalLeafBoundingBox(openvdb::CoordBBox& bbox) const override
        { bbox.min() = bbox.max() = openvdb::Coord(0, 0, 0); return false; }
    bool evalActiveVoxelBoundingBox(openvdb::CoordBBox& bbox) const override
        { bbox.min() = bbox.max() = openvdb::Coord(0, 0, 0); return false; }
    bool evalActiveVoxelDim(openvdb::Coord& dim) const override
        { dim = openvdb::Coord(0, 0, 0); return false; }
    bool evalLeafDim(openvdb::Coord& dim) const override
        { dim = openvdb::Coord(0, 0, 0); return false; }

    openvdb::Index treeDepth() const override { return 0; }
    openvdb::Index leafCount() const override { return 0; }
    openvdb::Index nonLeafCount() const override { return 0; }
    openvdb::Index64 activeVoxelCount() const override { return 0UL; }
    openvdb::Index64 inactiveVoxelCount() const override { return 0UL; }
    openvdb::Index64 activeLeafVoxelCount() const override { return 0UL; }
    openvdb::Index64 inactiveLeafVoxelCount() const override { return 0UL; }
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    openvdb::Index64 activeTileCount() const override { return 0UL; }
#endif
};

const openvdb::Index ProxyTree::DEPTH = 0;
const ProxyTree::ValueType ProxyTree::backg = 0;

using ProxyGrid = openvdb::Grid<ProxyTree>;


////////////////////////////////////////

void
TestGrid::testGridRegistry()
{
    using namespace openvdb::tree;

    using TreeType = Tree<RootNode<InternalNode<LeafNode<float, 3>, 2> > >;
    using GridType = openvdb::Grid<TreeType>;

    openvdb::GridBase::clearRegistry();

    CPPUNIT_ASSERT(!GridType::isRegistered());
    GridType::registerGrid();
    CPPUNIT_ASSERT(GridType::isRegistered());
    CPPUNIT_ASSERT_THROW(GridType::registerGrid(), openvdb::KeyError);
    GridType::unregisterGrid();
    CPPUNIT_ASSERT(!GridType::isRegistered());
    CPPUNIT_ASSERT_NO_THROW(GridType::unregisterGrid());
    CPPUNIT_ASSERT(!GridType::isRegistered());
    CPPUNIT_ASSERT_NO_THROW(GridType::registerGrid());
    CPPUNIT_ASSERT(GridType::isRegistered());

    openvdb::GridBase::clearRegistry();
}


void
TestGrid::testConstPtr()
{
    using namespace openvdb;

    GridBase::ConstPtr constgrid = ProxyGrid::create();

    CPPUNIT_ASSERT_EQUAL(Name("proxy"), constgrid->type());
}


void
TestGrid::testGetGrid()
{
    using namespace openvdb;

    GridBase::Ptr grid = FloatGrid::create(/*bg=*/0.0);
    GridBase::ConstPtr constGrid = grid;

    CPPUNIT_ASSERT(grid->baseTreePtr());

    CPPUNIT_ASSERT(!gridPtrCast<DoubleGrid>(grid));
    CPPUNIT_ASSERT(!gridPtrCast<DoubleGrid>(grid));

    CPPUNIT_ASSERT(gridConstPtrCast<FloatGrid>(constGrid));
    CPPUNIT_ASSERT(!gridConstPtrCast<DoubleGrid>(constGrid));
}


void
TestGrid::testIsType()
{
    using namespace openvdb;

    GridBase::Ptr grid = FloatGrid::create();
    CPPUNIT_ASSERT(grid->isType<FloatGrid>());
    CPPUNIT_ASSERT(!grid->isType<DoubleGrid>());
}


void
TestGrid::testTransform()
{
    ProxyGrid grid;

    // Verify that the grid has a valid default transform.
    CPPUNIT_ASSERT(grid.transformPtr());

    // Verify that a null transform pointer is not allowed.
    CPPUNIT_ASSERT_THROW(grid.setTransform(openvdb::math::Transform::Ptr()),
        openvdb::ValueError);

    grid.setTransform(openvdb::math::Transform::createLinearTransform());

    CPPUNIT_ASSERT(grid.transformPtr());

    // Verify that calling Transform-related Grid methods (Grid::voxelSize(), etc.)
    // is the same as calling those methods on the Transform.

    CPPUNIT_ASSERT(grid.transform().voxelSize().eq(grid.voxelSize()));
    CPPUNIT_ASSERT(grid.transform().voxelSize(openvdb::Vec3d(0.1, 0.2, 0.3)).eq(
        grid.voxelSize(openvdb::Vec3d(0.1, 0.2, 0.3))));

    CPPUNIT_ASSERT(grid.transform().indexToWorld(openvdb::Vec3d(0.1, 0.2, 0.3)).eq(
        grid.indexToWorld(openvdb::Vec3d(0.1, 0.2, 0.3))));
    CPPUNIT_ASSERT(grid.transform().indexToWorld(openvdb::Coord(1, 2, 3)).eq(
        grid.indexToWorld(openvdb::Coord(1, 2, 3))));
    CPPUNIT_ASSERT(grid.transform().worldToIndex(openvdb::Vec3d(0.1, 0.2, 0.3)).eq(
        grid.worldToIndex(openvdb::Vec3d(0.1, 0.2, 0.3))));
}


void
TestGrid::testCopyGrid()
{
    using namespace openvdb;

    // set up a grid
    const float fillValue1=5.0f;
    FloatGrid::Ptr grid1 = createGrid<FloatGrid>(/*bg=*/fillValue1);
    FloatTree& tree1 = grid1->tree();
    tree1.setValue(Coord(-10,40,845), 3.456f);
    tree1.setValue(Coord(1,-50,-8), 1.0f);

    // create a new grid, copying the first grid
    GridBase::Ptr grid2 = grid1->deepCopy();

    // cast down to the concrete type to query values
    FloatTree& tree2 = gridPtrCast<FloatGrid>(grid2)->tree();

    // compare topology
    CPPUNIT_ASSERT(tree1.hasSameTopology(tree2));
    CPPUNIT_ASSERT(tree2.hasSameTopology(tree1));

    // trees should be equal
    ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue1, tree2.getValue(Coord(1,2,3)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(3.456f, tree2.getValue(Coord(-10,40,845)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(1.0f, tree2.getValue(Coord(1,-50,-8)));

    // change 1 value in tree2
    Coord changeCoord(1, -500, -8);
    tree2.setValue(changeCoord, 1.0f);

    // topology should no longer match
    CPPUNIT_ASSERT(!tree1.hasSameTopology(tree2));
    CPPUNIT_ASSERT(!tree2.hasSameTopology(tree1));

    // query changed value and make sure it's different between trees
    ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue1, tree1.getValue(changeCoord));
    ASSERT_DOUBLES_EXACTLY_EQUAL(1.0f, tree2.getValue(changeCoord));
}


void
TestGrid::testValueConversion()
{
    using namespace openvdb;

    const Coord c0(-10, 40, 845), c1(1, -50, -8), c2(1, 2, 3);
    const float fval0 = 3.25f, fval1 = 1.0f, fbkgd = 5.0f;

    // Create a FloatGrid.
    FloatGrid fgrid(fbkgd);
    FloatTree& ftree = fgrid.tree();
    ftree.setValue(c0, fval0);
    ftree.setValue(c1, fval1);

    // Copy the FloatGrid to a DoubleGrid.
    DoubleGrid dgrid(fgrid);
    DoubleTree& dtree = dgrid.tree();
    // Compare topology.
    CPPUNIT_ASSERT(dtree.hasSameTopology(ftree));
    CPPUNIT_ASSERT(ftree.hasSameTopology(dtree));
    // Compare values.
    ASSERT_DOUBLES_EXACTLY_EQUAL(double(fbkgd), dtree.getValue(c2));
    ASSERT_DOUBLES_EXACTLY_EQUAL(double(fval0), dtree.getValue(c0));
    ASSERT_DOUBLES_EXACTLY_EQUAL(double(fval1), dtree.getValue(c1));

    // Copy the FloatGrid to a BoolGrid.
    BoolGrid bgrid(fgrid);
    BoolTree& btree = bgrid.tree();
    // Compare topology.
    CPPUNIT_ASSERT(btree.hasSameTopology(ftree));
    CPPUNIT_ASSERT(ftree.hasSameTopology(btree));
    // Compare values.
    CPPUNIT_ASSERT_EQUAL(bool(fbkgd), btree.getValue(c2));
    CPPUNIT_ASSERT_EQUAL(bool(fval0), btree.getValue(c0));
    CPPUNIT_ASSERT_EQUAL(bool(fval1), btree.getValue(c1));

    // Copy the FloatGrid to a Vec3SGrid.
    Vec3SGrid vgrid(fgrid);
    Vec3STree& vtree = vgrid.tree();
    // Compare topology.
    CPPUNIT_ASSERT(vtree.hasSameTopology(ftree));
    CPPUNIT_ASSERT(ftree.hasSameTopology(vtree));
    // Compare values.
    CPPUNIT_ASSERT_EQUAL(Vec3s(fbkgd), vtree.getValue(c2));
    CPPUNIT_ASSERT_EQUAL(Vec3s(fval0), vtree.getValue(c0));
    CPPUNIT_ASSERT_EQUAL(Vec3s(fval1), vtree.getValue(c1));

    // Verify that a Vec3SGrid can't be copied to an Int32Grid
    // (because an Int32 can't be constructed from a Vec3S).
    CPPUNIT_ASSERT_THROW(Int32Grid igrid2(vgrid), openvdb::TypeError);

    // Verify that a grid can't be converted to another type with a different
    // tree configuration.
    using DTree23 = tree::Tree3<double, 2, 3>::Type;
    using DGrid23 = Grid<DTree23>;
    CPPUNIT_ASSERT_THROW(DGrid23 d23grid(fgrid), openvdb::TypeError);
}


////////////////////////////////////////


template<typename GridT>
void
validateClippedGrid(const GridT& clipped, const typename GridT::ValueType& fg)
{
    using namespace openvdb;

    using ValueT = typename GridT::ValueType;

    const CoordBBox bbox = clipped.evalActiveVoxelBoundingBox();
    CPPUNIT_ASSERT_EQUAL(4, bbox.min().x());
    CPPUNIT_ASSERT_EQUAL(4, bbox.min().y());
    CPPUNIT_ASSERT_EQUAL(-6, bbox.min().z());
    CPPUNIT_ASSERT_EQUAL(4, bbox.max().x());
    CPPUNIT_ASSERT_EQUAL(4, bbox.max().y());
    CPPUNIT_ASSERT_EQUAL(6, bbox.max().z());
    CPPUNIT_ASSERT_EQUAL(6 + 6 + 1, int(clipped.activeVoxelCount()));
    CPPUNIT_ASSERT_EQUAL(2, int(clipped.constTree().leafCount()));

    typename GridT::ConstAccessor acc = clipped.getConstAccessor();
    const ValueT bg = clipped.background();
    Coord xyz;
    int &x = xyz[0], &y = xyz[1], &z = xyz[2];
    for (x = -10; x <= 10; ++x) {
        for (y = -10; y <= 10; ++y) {
            for (z = -10; z <= 10; ++z) {
                if (x == 4 && y == 4 && z >= -6 && z <= 6) {
                    CPPUNIT_ASSERT_EQUAL(fg, acc.getValue(Coord(4, 4, z)));
                } else {
                    CPPUNIT_ASSERT_EQUAL(bg, acc.getValue(Coord(x, y, z)));
                }
            }
        }
    }
}


// See also TestTools::testClipping()
void
TestGrid::testClipping()
{
    using namespace openvdb;

    const BBoxd clipBox(Vec3d(4.0, 4.0, -6.0), Vec3d(4.9, 4.9, 6.0));

    {
        const float fg = 5.f;
        FloatGrid cube(0.f);
        cube.fill(CoordBBox(Coord(-10), Coord(10)), /*value=*/fg, /*active=*/true);
#if OPENVDB_ABI_VERSION_NUMBER <= 2
        cube.tree().clip(cube.constTransform().worldToIndexNodeCentered(clipBox));
#else
        cube.clipGrid(clipBox);
#endif
        validateClippedGrid(cube, fg);
    }
    {
        const bool fg = true;
        BoolGrid cube(false);
        cube.fill(CoordBBox(Coord(-10), Coord(10)), /*value=*/fg, /*active=*/true);
#if OPENVDB_ABI_VERSION_NUMBER <= 2
        cube.tree().clip(cube.constTransform().worldToIndexNodeCentered(clipBox));
#else
        cube.clipGrid(clipBox);
#endif
        validateClippedGrid(cube, fg);
    }
    {
        const Vec3s fg(1.f, -2.f, 3.f);
        Vec3SGrid cube(Vec3s(0.f));
        cube.fill(CoordBBox(Coord(-10), Coord(10)), /*value=*/fg, /*active=*/true);
#if OPENVDB_ABI_VERSION_NUMBER <= 2
        cube.tree().clip(cube.constTransform().worldToIndexNodeCentered(clipBox));
#else
        cube.clipGrid(clipBox);
#endif
        validateClippedGrid(cube, fg);
    }
    /*
    {// Benchmark multi-threaded copy construction
        openvdb::util::CpuTimer timer;
        openvdb::initialize();
        openvdb::io::File file("/usr/pic1/Data/OpenVDB/LevelSetModels/crawler.vdb");
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid("ls_crawler");
        file.close();
        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        //grid->tree().print();
        timer.start("\nCopy construction");
        openvdb::FloatTree fTree(grid->tree());
        timer.stop();

        timer.start("\nBoolean topology copy construction");
        openvdb::BoolTree bTree(grid->tree(), false, openvdb::TopologyCopy());
        timer.stop();

        timer.start("\nBoolean topology union");
        bTree.topologyUnion(fTree);
        timer.stop();
        //bTree.print();
    }
    */
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
