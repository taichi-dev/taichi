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

#include <set>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/util/logging.h>
#include "util.h" // for unittest_util::makeSphere()


class TestLeafBool: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestLeafBool);
    CPPUNIT_TEST(testGetValue);
    CPPUNIT_TEST(testSetValue);
    CPPUNIT_TEST(testProbeValue);
    CPPUNIT_TEST(testIterators);
    CPPUNIT_TEST(testIteratorGetCoord);
    CPPUNIT_TEST(testEquivalence);
    CPPUNIT_TEST(testGetOrigin);
    CPPUNIT_TEST(testNegativeIndexing);
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST(testTopologyCopy);
    CPPUNIT_TEST(testMerge);
    CPPUNIT_TEST(testCombine);
    CPPUNIT_TEST(testBoolTree);
    CPPUNIT_TEST(testMedian);
    //CPPUNIT_TEST(testFilter);
    CPPUNIT_TEST_SUITE_END();

    void testGetValue();
    void testSetValue();
    void testProbeValue();
    void testIterators();
    void testEquivalence();
    void testGetOrigin();
    void testIteratorGetCoord();
    void testNegativeIndexing();
    void testIO();
    void testTopologyCopy();
    void testMerge();
    void testCombine();
    void testBoolTree();
    void testMedian();
    //void testFilter();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafBool);

typedef openvdb::tree::LeafNode<bool, 3> LeafType;


////////////////////////////////////////


void
TestLeafBool::testGetValue()
{
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/false);
        for (openvdb::Index n = 0; n < leaf.numValues(); ++n) {
            CPPUNIT_ASSERT_EQUAL(false, leaf.getValue(leaf.offsetToLocalCoord(n)));
        }
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/true);
        for (openvdb::Index n = 0; n < leaf.numValues(); ++n) {
            CPPUNIT_ASSERT_EQUAL(true, leaf.getValue(leaf.offsetToLocalCoord(n)));
        }
    }
    {// test Buffer::data()
        LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/false);
        leaf.fill(true);
        LeafType::Buffer::WordType* w = leaf.buffer().data();
        for (openvdb::Index n = 0; n < LeafType::Buffer::WORD_COUNT; ++n) {
            CPPUNIT_ASSERT_EQUAL(~LeafType::Buffer::WordType(0), w[n]);
        }
    }
    {// test const Buffer::data()
        LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/false);
        leaf.fill(true);
        const LeafType& cleaf = leaf;
        const LeafType::Buffer::WordType* w = cleaf.buffer().data();
        for (openvdb::Index n = 0; n < LeafType::Buffer::WORD_COUNT; ++n) {
            CPPUNIT_ASSERT_EQUAL(~LeafType::Buffer::WordType(0), w[n]);
        }
    }
}


void
TestLeafBool::testSetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0), false);

    openvdb::Coord xyz(0, 0, 0);
    CPPUNIT_ASSERT(!leaf.isValueOn(xyz));
    leaf.setValueOn(xyz);
    CPPUNIT_ASSERT(leaf.isValueOn(xyz));

    xyz.reset(7, 7, 7);
    CPPUNIT_ASSERT(!leaf.isValueOn(xyz));
    leaf.setValueOn(xyz);
    CPPUNIT_ASSERT(leaf.isValueOn(xyz));
    leaf.setValueOn(xyz, /*value=*/true); // value argument should be ignored
    CPPUNIT_ASSERT(leaf.isValueOn(xyz));
    leaf.setValueOn(xyz, /*value=*/false); // value argument should be ignored
    CPPUNIT_ASSERT(leaf.isValueOn(xyz));

    leaf.setValueOff(xyz);
    CPPUNIT_ASSERT(!leaf.isValueOn(xyz));

    xyz.reset(2, 3, 6);
    leaf.setValueOn(xyz);
    CPPUNIT_ASSERT(leaf.isValueOn(xyz));

    leaf.setValueOff(xyz);
    CPPUNIT_ASSERT(!leaf.isValueOn(xyz));
}


void
TestLeafBool::testProbeValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 6, 5));

    bool val;
    CPPUNIT_ASSERT(leaf.probeValue(openvdb::Coord(1, 6, 5), val));
    CPPUNIT_ASSERT(!leaf.probeValue(openvdb::Coord(1, 6, 4), val));
}


void
TestLeafBool::testIterators()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 2, 3));
    leaf.setValueOn(openvdb::Coord(5, 2, 3));
    openvdb::Coord sum;
    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
        sum += iter.getCoord();
    }
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(1 + 5, 2 + 2, 3 + 3), sum);

    openvdb::Index count = 0;
    for (LeafType::ValueOffIter iter = leaf.beginValueOff(); iter; ++iter, ++count);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues() - 2, count);

    count = 0;
    for (LeafType::ValueAllIter iter = leaf.beginValueAll(); iter; ++iter, ++count);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues(), count);

    count = 0;
    for (LeafType::ChildOnIter iter = leaf.beginChildOn(); iter; ++iter, ++count);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), count);

    count = 0;
    for (LeafType::ChildOffIter iter = leaf.beginChildOff(); iter; ++iter, ++count);
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(0), count);

    count = 0;
    for (LeafType::ChildAllIter iter = leaf.beginChildAll(); iter; ++iter, ++count);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues(), count);
}


void
TestLeafBool::testIteratorGetCoord()
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(8, 8, 0));

    CPPUNIT_ASSERT_EQUAL(Coord(8, 8, 0), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(9, 10, 3), xyz);

    ++iter;
    xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(13, 10, 3), xyz);
}


void
TestLeafBool::testEquivalence()
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    {
        LeafType leaf(Coord(0, 0, 0), false); // false and inactive
        LeafType leaf2(Coord(0, 0, 0), true); // true and inactive
        
        CPPUNIT_ASSERT(leaf != leaf2);
        
        leaf.fill(CoordBBox(Coord(0), Coord(LeafType::DIM - 1)), true, /*active=*/false);
        CPPUNIT_ASSERT(leaf == leaf2); // true and inactive
        
        leaf.setValuesOn(); // true and active
        
        leaf2.fill(CoordBBox(Coord(0), Coord(LeafType::DIM - 1)), false); // false and active
        CPPUNIT_ASSERT(leaf != leaf2);
        
        leaf.negate(); // false and active
        CPPUNIT_ASSERT(leaf == leaf2);
        
        // Set some values.
        leaf.setValueOn(Coord(0, 0, 0), true);
        leaf.setValueOn(Coord(0, 1, 0), true);
        leaf.setValueOn(Coord(1, 1, 0), true);
        leaf.setValueOn(Coord(1, 1, 2), true);
        
        leaf2.setValueOn(Coord(0, 0, 0), true);
        leaf2.setValueOn(Coord(0, 1, 0), true);
        leaf2.setValueOn(Coord(1, 1, 0), true);
        leaf2.setValueOn(Coord(1, 1, 2), true);
        
        CPPUNIT_ASSERT(leaf == leaf2);
        
        leaf2.setValueOn(Coord(0, 0, 1), true);
        
        CPPUNIT_ASSERT(leaf != leaf2);
        
        leaf2.setValueOff(Coord(0, 0, 1), false);
        
        CPPUNIT_ASSERT(leaf != leaf2);
        
        leaf2.setValueOn(Coord(0, 0, 1));
        
        CPPUNIT_ASSERT(leaf == leaf2);
    }
    {// test LeafNode<bool>::operator==()
        LeafType leaf1(Coord(0            , 0, 0), true); // true and inactive
        LeafType leaf2(Coord(1            , 0, 0), true); // true and inactive
        LeafType leaf3(Coord(LeafType::DIM, 0, 0), true); // true and inactive
        LeafType leaf4(Coord(0            , 0, 0), true, true);//true and active
        CPPUNIT_ASSERT(leaf1 == leaf2);
        CPPUNIT_ASSERT(leaf1 != leaf3);
        CPPUNIT_ASSERT(leaf2 != leaf3);
        CPPUNIT_ASSERT(leaf1 != leaf4);
        CPPUNIT_ASSERT(leaf2 != leaf4);
        CPPUNIT_ASSERT(leaf3 != leaf4);
    }
        
}


void
TestLeafBool::testGetOrigin()
{
    {
        LeafType leaf(openvdb::Coord(1, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 1, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1024, 1, 3), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(128*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1023, 1, 3), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(127*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(512, 512, 512), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(512, 512, 512), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(2, 52, 515), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 48, 512), leaf.origin());
    }
}


void
TestLeafBool::testNegativeIndexing()
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(-9, -2, -8));

    CPPUNIT_ASSERT_EQUAL(Coord(-16, -8, -8), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3));
    leaf.setValueOn(Coord(5, 2, 3));

    CPPUNIT_ASSERT(leaf.isValueOn(Coord(1, 2, 3)));
    CPPUNIT_ASSERT(leaf.isValueOn(Coord(5, 2, 3)));

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(-15, -6, -5), xyz);

    ++iter;
    xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(-11, -6, -5), xyz);
}


void
TestLeafBool::testIO()
{
    LeafType leaf(openvdb::Coord(1, 3, 5));
    const openvdb::Coord origin = leaf.origin();

    leaf.setValueOn(openvdb::Coord(0, 1, 0));
    leaf.setValueOn(openvdb::Coord(1, 0, 0));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOff(openvdb::Coord(0, 1, 0));
    leaf.setValueOn(openvdb::Coord(0, 1, 1));

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    // Since the input stream doesn't include a VDB header with file format version info,
    // tag the input stream explicitly with the current version number.
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    CPPUNIT_ASSERT_EQUAL(origin, leaf.origin());

    CPPUNIT_ASSERT(leaf.isValueOn(openvdb::Coord(0, 1, 0)));
    CPPUNIT_ASSERT(leaf.isValueOn(openvdb::Coord(1, 0, 0)));

    CPPUNIT_ASSERT(leaf.onVoxelCount() == 2);
}


void
TestLeafBool::testTopologyCopy()
{
    using openvdb::Coord;

    // LeafNode<float, Log2Dim> having the same Log2Dim as LeafType
    typedef LeafType::ValueConverter<float>::Type FloatLeafType;

    FloatLeafType fleaf(Coord(10, 20, 30), /*background=*/-1.0);
    std::set<Coord> coords;
    for (openvdb::Index n = 0; n < fleaf.numValues(); n += 10) {
        Coord xyz = fleaf.offsetToGlobalCoord(n);
        fleaf.setValueOn(xyz, float(n));
        coords.insert(xyz);
    }

    LeafType leaf(fleaf, openvdb::TopologyCopy());
    CPPUNIT_ASSERT_EQUAL(fleaf.onVoxelCount(), leaf.onVoxelCount());

    CPPUNIT_ASSERT(leaf.hasSameTopology(&fleaf));

    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
        coords.erase(iter.getCoord());
    }
    CPPUNIT_ASSERT(coords.empty());
}


void
TestLeafBool::testMerge()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    for (openvdb::Index n = 0; n < leaf.numValues(); n += 10) {
        leaf.setValueOn(n);
    }
    CPPUNIT_ASSERT(!leaf.isValueMaskOn());
    CPPUNIT_ASSERT(!leaf.isValueMaskOff());
    bool val = false, active = false;
    CPPUNIT_ASSERT(!leaf.isConstant(val, active));

    LeafType leaf2(leaf);
    leaf2.getValueMask().toggle();
    CPPUNIT_ASSERT(!leaf2.isValueMaskOn());
    CPPUNIT_ASSERT(!leaf2.isValueMaskOff());
    val = active = false;
    CPPUNIT_ASSERT(!leaf2.isConstant(val, active));

    leaf.merge<openvdb::MERGE_ACTIVE_STATES>(leaf2);
    CPPUNIT_ASSERT(leaf.isValueMaskOn());
    CPPUNIT_ASSERT(!leaf.isValueMaskOff());
    val = active = false;
    CPPUNIT_ASSERT(leaf.isConstant(val, active));
    CPPUNIT_ASSERT(active);
}


void
TestLeafBool::testCombine()
{
    struct Local {
        static void op(openvdb::CombineArgs<bool>& args) {
            args.setResult(false); // result should be ignored
            args.setResultIsActive(args.aIsActive() ^ args.bIsActive());
        }
    };

    LeafType leaf(openvdb::Coord(0, 0, 0));
    for (openvdb::Index n = 0; n < leaf.numValues(); n += 10) {
        leaf.setValueOn(n);
    }
    CPPUNIT_ASSERT(!leaf.isValueMaskOn());
    CPPUNIT_ASSERT(!leaf.isValueMaskOff());
    const LeafType::NodeMaskType savedMask = leaf.getValueMask();
    OPENVDB_LOG_DEBUG_RUNTIME(leaf.str());

    LeafType leaf2(leaf);
    for (openvdb::Index n = 0; n < leaf.numValues(); n += 4) {
        leaf2.setValueOn(n);
    }
    CPPUNIT_ASSERT(!leaf2.isValueMaskOn());
    CPPUNIT_ASSERT(!leaf2.isValueMaskOff());
    OPENVDB_LOG_DEBUG_RUNTIME(leaf2.str());

    leaf.combine(leaf2, Local::op);
    OPENVDB_LOG_DEBUG_RUNTIME(leaf.str());

    CPPUNIT_ASSERT(leaf.getValueMask() == (savedMask ^ leaf2.getValueMask()));
}


void
TestLeafBool::testBoolTree()
{
    using namespace openvdb;

#if 0
    FloatGrid::Ptr inGrid;
    FloatTree::Ptr inTree;
    {
        //io::File vdbFile("/work/rd/fx_tools/vdb_unittest/TestGridCombine::testCsg/large1.vdb2");
        io::File vdbFile("/hosts/whitestar/usr/pic1/VDB/bunny_0256.vdb2");
        vdbFile.open();
        inGrid = gridPtrCast<FloatGrid>(vdbFile.readGrid("LevelSet"));
        CPPUNIT_ASSERT(inGrid.get() != NULL);
        inTree = inGrid->treePtr();
        CPPUNIT_ASSERT(inTree.get() != NULL);
    }
#else
    FloatGrid::Ptr inGrid = FloatGrid::create();
    CPPUNIT_ASSERT(inGrid.get() != NULL);
    FloatTree& inTree = inGrid->tree();
    inGrid->setName("LevelSet");

    unittest_util::makeSphere<FloatGrid>(/*dim   =*/Coord(128),
                                         /*center=*/Vec3f(0, 0, 0),
                                         /*radius=*/5,
                                         *inGrid, unittest_util::SPHERE_DENSE);
#endif

    const Index64
        floatTreeMem = inTree.memUsage(),
        floatTreeLeafCount = inTree.leafCount(),
        floatTreeVoxelCount = inTree.activeVoxelCount();

    TreeBase::Ptr outTree(new BoolTree(inTree, false, true, TopologyCopy()));
    CPPUNIT_ASSERT(outTree.get() != NULL);

    BoolGrid::Ptr outGrid = BoolGrid::create(*inGrid); // copy transform and metadata
    outGrid->setTree(outTree);
    outGrid->setName("Boolean");

    const Index64
        boolTreeMem = outTree->memUsage(),
        boolTreeLeafCount = outTree->leafCount(),
        boolTreeVoxelCount = outTree->activeVoxelCount();

#if 0
    GridPtrVec grids;
    grids.push_back(inGrid);
    grids.push_back(outGrid);
    io::File vdbFile("bool_tree.vdb2");
    vdbFile.write(grids);
    vdbFile.close();
#endif

    CPPUNIT_ASSERT_EQUAL(floatTreeLeafCount, boolTreeLeafCount);
    CPPUNIT_ASSERT_EQUAL(floatTreeVoxelCount, boolTreeVoxelCount);

    //std::cerr << "\nboolTree mem=" << boolTreeMem << " bytes" << std::endl;
    //std::cerr << "floatTree mem=" << floatTreeMem << " bytes" << std::endl;

    // Considering only voxel buffer memory usage, the BoolTree would be expected
    // to use (2 mask bits/voxel / ((32 value bits + 1 mask bit)/voxel)) = ~1/16
    // as much memory as the FloatTree.  Considering total memory usage, verify that
    // the BoolTree is no more than 1/10 the size of the FloatTree.
    CPPUNIT_ASSERT(boolTreeMem * 10 <= floatTreeMem);
}

void
TestLeafBool::testMedian()
{
    using namespace openvdb;
    LeafType leaf(openvdb::Coord(0, 0, 0), /*background=*/false);
    bool state = false;
    
    CPPUNIT_ASSERT_EQUAL(Index(0), leaf.medianOn(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues(), leaf.medianOff(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT(!leaf.medianAll());

    leaf.setValue(Coord(0,0,0), true);
    CPPUNIT_ASSERT_EQUAL(Index(1), leaf.medianOn(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-1, leaf.medianOff(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT(!leaf.medianAll());

    leaf.setValue(Coord(0,0,1), true);
    CPPUNIT_ASSERT_EQUAL(Index(2), leaf.medianOn(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-2, leaf.medianOff(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT(!leaf.medianAll());
    
    leaf.setValue(Coord(5,0,1), true);
    CPPUNIT_ASSERT_EQUAL(Index(3), leaf.medianOn(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-3, leaf.medianOff(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT(!leaf.medianAll());

    leaf.fill(false, false);
    CPPUNIT_ASSERT_EQUAL(Index(0), leaf.medianOn(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues(), leaf.medianOff(state));
    CPPUNIT_ASSERT(state == false);
    CPPUNIT_ASSERT(!leaf.medianAll());

    for (Index i=0; i<leaf.numValues()/2; ++i) {
        leaf.setValueOn(i, true);
        CPPUNIT_ASSERT(!leaf.medianAll());
        CPPUNIT_ASSERT_EQUAL(Index(i+1), leaf.medianOn(state));
        CPPUNIT_ASSERT(state == false);
        CPPUNIT_ASSERT_EQUAL(leaf.numValues()-i-1, leaf.medianOff(state));
        CPPUNIT_ASSERT(state == false);
    }
    for (Index i=leaf.numValues()/2; i<leaf.numValues(); ++i) {
        leaf.setValueOn(i, true);
        CPPUNIT_ASSERT(leaf.medianAll());
        CPPUNIT_ASSERT_EQUAL(Index(i+1), leaf.medianOn(state));
        CPPUNIT_ASSERT(state == true);
        CPPUNIT_ASSERT_EQUAL(leaf.numValues()-i-1, leaf.medianOff(state));
        CPPUNIT_ASSERT(state == false);
    }
}

// void
// TestLeafBool::testFilter()
// {
//     using namespace openvdb;

//     BoolGrid::Ptr grid = BoolGrid::create();
//     CPPUNIT_ASSERT(grid.get() != NULL);
//     BoolTree::Ptr tree = grid->treePtr();
//     CPPUNIT_ASSERT(tree.get() != NULL);
//     grid->setName("filtered");

//     unittest_util::makeSphere<BoolGrid>(/*dim=*/Coord(32),
//                                         /*ctr=*/Vec3f(0, 0, 0),
//                                         /*radius=*/10,
//                                         *grid, unittest_util::SPHERE_DENSE);

//     BoolTree::Ptr copyOfTree(new BoolTree(*tree));
//     BoolGrid::Ptr copyOfGrid = BoolGrid::create(copyOfTree);
//     copyOfGrid->setName("original");

//     tools::Filter<BoolGrid> filter(*grid);
//     filter.offset(1);

// #if 0
//     GridPtrVec grids;
//     grids.push_back(copyOfGrid);
//     grids.push_back(grid);
//     io::File vdbFile("TestLeafBool::testFilter.vdb2");
//     vdbFile.write(grids);
//     vdbFile.close();
// #endif

//     // Verify that offsetting all active voxels by 1 (true) has no effect,
//     // since the active voxels were all true to begin with.
//     CPPUNIT_ASSERT(tree->hasSameTopology(*copyOfTree));
// }

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
