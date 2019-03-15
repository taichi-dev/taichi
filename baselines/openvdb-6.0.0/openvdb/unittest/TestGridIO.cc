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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <cstdio> // for remove()


class TestGridIO: public CppUnit::TestCase
{
public:
    typedef openvdb::tree::Tree<
        openvdb::tree::RootNode<
        openvdb::tree::InternalNode<
        openvdb::tree::InternalNode<
        openvdb::tree::InternalNode<
        openvdb::tree::LeafNode<float, 2>, 3>, 4>, 5> > >
        Float5432Tree;
    typedef openvdb::Grid<Float5432Tree> Float5432Grid;

    virtual void setUp()    { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestGridIO);
    CPPUNIT_TEST(testReadAllBool);
    CPPUNIT_TEST(testReadAllMask);
    CPPUNIT_TEST(testReadAllFloat);
    CPPUNIT_TEST(testReadAllVec3S);
    CPPUNIT_TEST(testReadAllFloat5432);
    CPPUNIT_TEST_SUITE_END();

    void testReadAllBool()  { readAllTest<openvdb::BoolGrid>(); }
    void testReadAllMask()  { readAllTest<openvdb::MaskGrid>(); }
    void testReadAllFloat() { readAllTest<openvdb::FloatGrid>(); }
    void testReadAllVec3S() { readAllTest<openvdb::Vec3SGrid>(); }
    void testReadAllFloat5432() { Float5432Grid::registerGrid(); readAllTest<Float5432Grid>(); }
private:
    template<typename GridType> void readAllTest();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGridIO);


////////////////////////////////////////


template<typename GridType>
void
TestGridIO::readAllTest()
{
    using namespace openvdb;

    typedef typename GridType::TreeType TreeType;
    typedef typename TreeType::Ptr TreePtr;
    typedef typename TreeType::ValueType ValueT;
    typedef typename TreeType::NodeCIter NodeCIter;
    const ValueT zero = zeroVal<ValueT>();

    // For each level of the tree, compute a bit mask for use in converting
    // global coordinates to node origins for nodes at that level.
    // That is, node_origin = global_coordinates & mask[node_level].
    std::vector<Index> mask;
    TreeType::getNodeLog2Dims(mask);
    const size_t height = mask.size();
    for (size_t i = 0; i < height; ++i) {
        Index dim = 0;
        for (size_t j = i; j < height; ++j) dim += mask[j];
        mask[i] = ~((1 << dim) - 1);
    }
    const Index childDim = 1 + ~(mask[0]);

    // Choose sample coordinate pairs (coord0, coord1) and (coord0, coord2)
    // that are guaranteed to lie in different children of the root node
    // (because they are separated by more than the child node dimension).
    const Coord
        coord0(0, 0, 0),
        coord1(int(1.1 * childDim), 0, 0),
        coord2(0, int(1.1 * childDim), 0);

    // Create trees.
    TreePtr
        tree1(new TreeType(zero + 1)),
        tree2(new TreeType(zero + 2));

    // Set some values.
    tree1->setValue(coord0, zero + 5);
    tree1->setValue(coord1, zero + 6);
    tree2->setValue(coord0, zero + 10);
    tree2->setValue(coord2, zero + 11);

    // Create grids with trees and assign transforms.
    math::Transform::Ptr trans1(math::Transform::createLinearTransform(0.1)),
        trans2(math::Transform::createLinearTransform(0.1));
    GridBase::Ptr grid1 = createGrid(tree1), grid2 = createGrid(tree2);
    grid1->setTransform(trans1);
    grid1->setName("density");
    grid2->setTransform(trans2);
    grid2->setName("temperature");

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 5), tree1->getValue(coord0));
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 6), tree1->getValue(coord1));
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 10), tree2->getValue(coord0));
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 11), tree2->getValue(coord2));
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    // count[d] is the number of nodes already visited at depth d.
    // There should be exactly two nodes at each depth (apart from the root).
    std::vector<int> count(height, 0);

    // Verify that tree1 has correct node origins.
    for (NodeCIter iter = tree1->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = {
            coord0 & mask[depth], // origin of the first node at this depth
            coord1 & mask[depth]  // origin of the second node at this depth
        };
        CPPUNIT_ASSERT_EQUAL(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }
    // Verify that tree2 has correct node origins.
    count.assign(height, 0); // reset node counts
    for (NodeCIter iter = tree2->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = { coord0 & mask[depth], coord2 & mask[depth] };
        CPPUNIT_ASSERT_EQUAL(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }

    MetaMap::Ptr meta(new MetaMap);
    meta->insertMeta("author", StringMetadata("Einstein"));
    meta->insertMeta("year", Int32Metadata(2009));

    GridPtrVecPtr grids(new GridPtrVec);
    grids->push_back(grid1);
    grids->push_back(grid2);

    // Write grids and metadata out to a file.
    {
        io::File vdbfile("something.vdb2");
        vdbfile.write(*grids, *meta);
    }
    meta.reset();
    grids.reset();

    io::File vdbfile("something.vdb2");
    CPPUNIT_ASSERT_THROW(vdbfile.getGrids(), openvdb::IoError); // file has not been opened

    // Read the grids back in.
    vdbfile.open();
    CPPUNIT_ASSERT(vdbfile.isOpen());

    grids = vdbfile.getGrids();
    meta = vdbfile.getMetadata();

    // Ensure we have the metadata.
    CPPUNIT_ASSERT(meta.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(2, int(meta->metaCount()));
    CPPUNIT_ASSERT_EQUAL(std::string("Einstein"), meta->metaValue<std::string>("author"));
    CPPUNIT_ASSERT_EQUAL(2009, meta->metaValue<int32_t>("year"));

    // Ensure we got both grids.
    CPPUNIT_ASSERT(grids.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(2, int(grids->size()));

    grid1.reset();
    grid1 = findGridByName(*grids, "density");
    CPPUNIT_ASSERT(grid1.get() != NULL);
    TreePtr density = gridPtrCast<GridType>(grid1)->treePtr();
    CPPUNIT_ASSERT(density.get() != NULL);

    grid2.reset();
    grid2 = findGridByName(*grids, "temperature");
    CPPUNIT_ASSERT(grid2.get() != NULL);
    TreePtr temperature = gridPtrCast<GridType>(grid2)->treePtr();
    CPPUNIT_ASSERT(temperature.get() != NULL);

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 5), density->getValue(coord0));
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 6), density->getValue(coord1));
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 10), temperature->getValue(coord0));
    CPPUNIT_ASSERT_EQUAL(ValueT(zero + 11), temperature->getValue(coord2));
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    // Check if we got the correct node origins.
    count.assign(height, 0);
    for (NodeCIter iter = density->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = { coord0 & mask[depth], coord1 & mask[depth] };
        CPPUNIT_ASSERT_EQUAL(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }
    count.assign(height, 0);
    for (NodeCIter iter = temperature->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = { coord0 & mask[depth], coord2 & mask[depth] };
        CPPUNIT_ASSERT_EQUAL(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }

    vdbfile.close();

    ::remove("something.vdb2");
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
