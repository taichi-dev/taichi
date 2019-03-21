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
#include <openvdb/tree/Tree.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/LevelSetSphere.h> // for tools::createLevelSetSphere()


#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0)


class TestTreeIterators: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestTreeIterators);
    CPPUNIT_TEST(testLeafIterator);
    CPPUNIT_TEST(testEmptyLeafIterator);
    CPPUNIT_TEST(testOnlyNegative);
    CPPUNIT_TEST(testValueAllIterator);
    CPPUNIT_TEST(testValueOnIterator);
    CPPUNIT_TEST(testValueOffIterator);
    CPPUNIT_TEST(testModifyValue);
    CPPUNIT_TEST(testDepthBounds);
    CPPUNIT_TEST_SUITE_END();

    void testLeafIterator();
    void testEmptyLeafIterator();
    void testOnlyNegative();
    void testValueAllIterator();
    void testValueOnIterator();
    void testValueOffIterator();
    void testModifyValue();
    void testDepthBounds();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTreeIterators);


typedef openvdb::FloatTree TreeType;


void
TestTreeIterators::testLeafIterator()
{
    const float fillValue = 256.0f;

    TreeType tree(fillValue);

    tree.setValue(openvdb::Coord(0, 0,  0), 1.0);
    tree.setValue(openvdb::Coord(1, 0,  0), 1.5);
    tree.setValue(openvdb::Coord(0, 0,  8), 2.0);
    tree.setValue(openvdb::Coord(1, 0,  8), 2.5);
    tree.setValue(openvdb::Coord(0, 0, 16), 3.0);
    tree.setValue(openvdb::Coord(1, 0, 16), 3.5);
    tree.setValue(openvdb::Coord(0, 0, 24), 4.0);
    tree.setValue(openvdb::Coord(1, 0, 24), 4.5);

    float val = 1.f;
    for (TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
        const TreeType::LeafNodeType* leaf = iter.getLeaf();
        CPPUNIT_ASSERT(leaf != NULL);
        ASSERT_DOUBLES_EXACTLY_EQUAL(val,       leaf->getValue(openvdb::Coord(0, 0, 0)));
        ASSERT_DOUBLES_EXACTLY_EQUAL(val + 0.5, iter->getValue(openvdb::Coord(1, 0, 0)));
        ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue, iter->getValue(openvdb::Coord(1, 1, 1)));
        val = val + 1.f;
    }
}


// Test the leaf iterator over a tree without any leaf nodes.
void
TestTreeIterators::testEmptyLeafIterator()
{
    using namespace openvdb;

    TreeType tree(/*fillValue=*/256.0);

    std::vector<Index> dims;
    tree.getNodeLog2Dims(dims);
    CPPUNIT_ASSERT_EQUAL(4, int(dims.size()));

    // Start with an iterator over an empty tree.
    TreeType::LeafCIter iter = tree.cbeginLeaf();
    CPPUNIT_ASSERT(!iter);

    // Using sparse fill, add internal nodes but no leaf nodes to the tree.

    // Fill the region subsumed by a level-2 internal node (assuming a four-level tree).
    Index log2Sum = dims[1] + dims[2] + dims[3];
    CoordBBox bbox(Coord(0), Coord((1 << log2Sum) - 1));
    tree.fill(bbox, /*value=*/1.0);
    iter = tree.cbeginLeaf();
    CPPUNIT_ASSERT(!iter);

    // Fill the region subsumed by a level-1 internal node.
    log2Sum = dims[2] + dims[3];
    bbox.reset(Coord(0), Coord((1 << log2Sum) - 1));
    tree.fill(bbox, /*value=*/2.0);
    iter = tree.cbeginLeaf();
    CPPUNIT_ASSERT(!iter);
}


void
TestTreeIterators::testOnlyNegative()
{
    using openvdb::Index64;

    const float fillValue = 5.0f;

    TreeType tree(fillValue);

    CPPUNIT_ASSERT(tree.empty());
    ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue, tree.getValue(openvdb::Coord(5, -10, 20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue, tree.getValue(openvdb::Coord(-500, 200, 300)));

    tree.setValue(openvdb::Coord(-5,  10,  20), 0.1f);
    tree.setValue(openvdb::Coord( 5, -10,  20), 0.2f);
    tree.setValue(openvdb::Coord( 5,  10, -20), 0.3f);
    tree.setValue(openvdb::Coord(-5, -10,  20), 0.4f);
    tree.setValue(openvdb::Coord(-5,  10, -20), 0.5f);
    tree.setValue(openvdb::Coord( 5, -10, -20), 0.6f);
    tree.setValue(openvdb::Coord(-5, -10, -20), 0.7f);
    tree.setValue(openvdb::Coord(-500,  200, -300), 4.5678f);
    tree.setValue(openvdb::Coord( 500, -200, -300), 4.5678f);
    tree.setValue(openvdb::Coord(-500, -200,  300), 4.5678f);

    ASSERT_DOUBLES_EXACTLY_EQUAL(0.1f, tree.getValue(openvdb::Coord(-5,  10,  20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(0.2f, tree.getValue(openvdb::Coord( 5, -10,  20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(0.3f, tree.getValue(openvdb::Coord( 5,  10, -20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(0.4f, tree.getValue(openvdb::Coord(-5, -10,  20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(0.5f, tree.getValue(openvdb::Coord(-5,  10, -20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(0.6f, tree.getValue(openvdb::Coord( 5, -10, -20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(0.7f, tree.getValue(openvdb::Coord(-5, -10, -20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(4.5678f, tree.getValue(openvdb::Coord(-500,  200, -300)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(4.5678f, tree.getValue(openvdb::Coord( 500, -200, -300)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(4.5678f, tree.getValue(openvdb::Coord(-500, -200,  300)));

    int count = 0;
    for (int i = -25; i < 25; ++i) {
        for (int j = -25; j < 25; ++j) {
            for (int k = -25; k < 25; ++k) {
                if (tree.getValue(openvdb::Coord(i, j, k)) < 1.0f) {
                    //fprintf(stderr, "(%i, %i, %i) = %f\n",
                    //    i, j, k, tree.getValue(openvdb::Coord(i, j, k)));
                    ++count;
                }
            }
        }
    }
    CPPUNIT_ASSERT_EQUAL(7, count);

    openvdb::Coord xyz;
    int count2 = 0;
    for (TreeType::ValueOnCIter iter = tree.cbeginValueOn();iter; ++iter) {
        ++count2;
        xyz = iter.getCoord();
        //std::cerr << xyz << " = " << *iter << "\n";
    }
    CPPUNIT_ASSERT_EQUAL(10, count2);
    CPPUNIT_ASSERT_EQUAL(Index64(10), tree.activeVoxelCount());
}


void
TestTreeIterators::testValueAllIterator()
{
    const openvdb::Index DIM0 = 3, DIM1 = 2, DIM2 = 3;

    typedef openvdb::tree::Tree4<float, DIM2, DIM1, DIM0>::Type Tree323f;

    typedef Tree323f::RootNodeType RootT;
    typedef RootT::ChildNodeType Int1T;
    typedef Int1T::ChildNodeType Int2T;
    typedef Int2T::ChildNodeType LeafT;

    Tree323f tree(/*fillValue=*/256.0f);
    tree.setValue(openvdb::Coord(4), 0.0f);
    tree.setValue(openvdb::Coord(-4), -1.0f);

    const size_t expectedNumOff =
          2 * ((1 << (3 * DIM2)) - 1)  // 2 8x8x8 InternalNodes - 1 child pointer each
        + 2 * ((1 << (3 * DIM1)) - 1)  // 2 4x4x4 InternalNodes - 1 child pointer each
        + 2 * ((1 << (3 * DIM0)) - 1); // 2 8x8x8 LeafNodes - 1 active value each

    {
        Tree323f::ValueAllIter iter = tree.beginValueAll();
        CPPUNIT_ASSERT(iter.test());

        // Read all tile and voxel values through a non-const value iterator.
        size_t numOn = 0, numOff = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(iter.getLevel() <= 3);
            const openvdb::Index iterLevel = iter.getLevel();
            for (openvdb::Index lvl = 0; lvl <= 3; ++lvl) {
                RootT* root; Int1T* int1; Int2T* int2; LeafT* leaf;
                iter.getNode(root); CPPUNIT_ASSERT(root != NULL);
                iter.getNode(int1); CPPUNIT_ASSERT(iterLevel < 3 ? int1 != NULL: int1 == NULL);
                iter.getNode(int2); CPPUNIT_ASSERT(iterLevel < 2 ? int2 != NULL: int2 == NULL);
                iter.getNode(leaf); CPPUNIT_ASSERT(iterLevel < 1 ? leaf != NULL: leaf == NULL);
            }

            if (iter.isValueOn()) {
                ++numOn;
                const float f = iter.getValue();
                if (openvdb::math::isZero(f)) {
                    CPPUNIT_ASSERT(iter.getCoord() == openvdb::Coord(4));
                    CPPUNIT_ASSERT(iter.isVoxelValue());
                } else {
                    ASSERT_DOUBLES_EXACTLY_EQUAL(-1.0f, f);
                    CPPUNIT_ASSERT(iter.getCoord() == openvdb::Coord(-4));
                    CPPUNIT_ASSERT(iter.isVoxelValue());
                }
            } else {
                ++numOff;

                // For every tenth inactive value, check that the size of
                // the tile or voxel is as expected.
                if (numOff % 10 == 0) {
                    const int dim[4] = {
                        1, 1 << DIM0, 1 << (DIM1 + DIM0), 1 << (DIM2 + DIM1 + DIM0)
                    };
                    const int lvl = iter.getLevel();
                    CPPUNIT_ASSERT(lvl < 4);
                    openvdb::CoordBBox bbox;
                    iter.getBoundingBox(bbox);
                    CPPUNIT_ASSERT_EQUAL(
                        bbox.extents(), openvdb::Coord(dim[lvl], dim[lvl], dim[lvl]));
                }
            }
        }
        CPPUNIT_ASSERT_EQUAL(2, int(numOn));
        CPPUNIT_ASSERT_EQUAL(expectedNumOff, numOff);
    }
    {
        Tree323f::ValueAllCIter iter = tree.cbeginValueAll();
        CPPUNIT_ASSERT(iter.test());

        // Read all tile and voxel values through a const value iterator.
        size_t numOn = 0, numOff = 0;
        for ( ; iter.test(); iter.next()) {
            if (iter.isValueOn()) ++numOn; else ++numOff;
        }
        CPPUNIT_ASSERT_EQUAL(2, int(numOn));
        CPPUNIT_ASSERT_EQUAL(expectedNumOff, numOff);
    }
    {
        Tree323f::ValueAllIter iter = tree.beginValueAll();
        CPPUNIT_ASSERT(iter.test());

        // Read all tile and voxel values through a non-const value iterator
        // and overwrite all active values.
        size_t numOn = 0, numOff = 0;
        for ( ; iter; ++iter) {
            if (iter.isValueOn()) {
                iter.setValue(iter.getValue() - 5);
                ++numOn;
            } else {
                ++numOff;
            }
        }
        CPPUNIT_ASSERT_EQUAL(2, int(numOn));
        CPPUNIT_ASSERT_EQUAL(expectedNumOff, numOff);
    }
}


void
TestTreeIterators::testValueOnIterator()
{
    typedef openvdb::tree::Tree4<float, 3, 2, 3>::Type Tree323f;

    Tree323f tree(/*fillValue=*/256.0f);

    {
        Tree323f::ValueOnIter iter = tree.beginValueOn();
        CPPUNIT_ASSERT(!iter.test()); // empty tree
    }

    const int STEP = 8/*100*/, NUM_STEPS = 10;
    for (int i = 0; i < NUM_STEPS; ++i) {
        tree.setValue(openvdb::Coord(STEP * i), 0.0f);
    }

    {
        Tree323f::ValueOnIter iter = tree.beginValueOn();
        CPPUNIT_ASSERT(iter.test());

        // Read all active tile and voxel values through a non-const value iterator.
        int numOn = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(iter.isVoxelValue());
            CPPUNIT_ASSERT(iter.isValueOn());
            ASSERT_DOUBLES_EXACTLY_EQUAL(0.0f, iter.getValue());
            CPPUNIT_ASSERT_EQUAL(openvdb::Coord(STEP * numOn), iter.getCoord());
            ++numOn;
        }
        CPPUNIT_ASSERT_EQUAL(NUM_STEPS, numOn);
    }
    {
        Tree323f::ValueOnCIter iter = tree.cbeginValueOn();
        CPPUNIT_ASSERT(iter.test());

        // Read all active tile and voxel values through a const value iterator.
        int numOn = 0;
        for ( ; iter.test(); iter.next()) {
            CPPUNIT_ASSERT(iter.isVoxelValue());
            CPPUNIT_ASSERT(iter.isValueOn());
            ASSERT_DOUBLES_EXACTLY_EQUAL(0.0f, iter.getValue());
            CPPUNIT_ASSERT_EQUAL(openvdb::Coord(STEP * numOn), iter.getCoord());
            ++numOn;
        }
        CPPUNIT_ASSERT_EQUAL(NUM_STEPS, numOn);
    }
    {
        Tree323f::ValueOnIter iter = tree.beginValueOn();
        CPPUNIT_ASSERT(iter.test());

        // Read all active tile and voxel values through a non-const value iterator
        // and overwrite the values.
        int numOn = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(iter.isVoxelValue());
            CPPUNIT_ASSERT(iter.isValueOn());
            ASSERT_DOUBLES_EXACTLY_EQUAL(0.0f, iter.getValue());
            iter.setValue(5.0f);
            ASSERT_DOUBLES_EXACTLY_EQUAL(5.0f, iter.getValue());
            CPPUNIT_ASSERT_EQUAL(openvdb::Coord(STEP * numOn), iter.getCoord());
            ++numOn;
        }
        CPPUNIT_ASSERT_EQUAL(NUM_STEPS, numOn);
    }
}


void
TestTreeIterators::testValueOffIterator()
{
    const openvdb::Index DIM0 = 3, DIM1 = 2, DIM2 = 3;

    typedef openvdb::tree::Tree4<float, DIM2, DIM1, DIM0>::Type Tree323f;

    Tree323f tree(/*fillValue=*/256.0f);
    tree.setValue(openvdb::Coord(4), 0.0f);
    tree.setValue(openvdb::Coord(-4), -1.0f);

    const size_t expectedNumOff =
          2 * ((1 << (3 * DIM2)) - 1)  // 2 8x8x8 InternalNodes - 1 child pointer each
        + 2 * ((1 << (3 * DIM1)) - 1)  // 2 4x4x4 InternalNodes - 1 child pointer each
        + 2 * ((1 << (3 * DIM0)) - 1); // 2 8x8x8 LeafNodes - 1 active value each

    {
        Tree323f::ValueOffIter iter = tree.beginValueOff();
        CPPUNIT_ASSERT(iter.test());

        // Read all inactive tile and voxel values through a non-const value iterator.
        size_t numOff = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(!iter.isValueOn());
            ++numOff;
            // For every tenth inactive value, check that the size of
            // the tile or voxel is as expected.
            if (numOff % 10 == 0) {
                const int dim[4] = {
                    1, 1 << DIM0, 1 << (DIM1 + DIM0), 1 << (DIM2 + DIM1 + DIM0)
                };
                const int lvl = iter.getLevel();
                CPPUNIT_ASSERT(lvl < 4);
                openvdb::CoordBBox bbox;
                iter.getBoundingBox(bbox);
                CPPUNIT_ASSERT_EQUAL(bbox.extents(), openvdb::Coord(dim[lvl], dim[lvl], dim[lvl]));
            }
        }
        CPPUNIT_ASSERT_EQUAL(expectedNumOff, numOff);
    }
    {
        Tree323f::ValueOffCIter iter = tree.cbeginValueOff();
        CPPUNIT_ASSERT(iter.test());

        // Read all inactive tile and voxel values through a const value iterator.
        size_t numOff = 0;
        for ( ; iter.test(); iter.next(), ++numOff) {
            CPPUNIT_ASSERT(!iter.isValueOn());
        }
        CPPUNIT_ASSERT_EQUAL(expectedNumOff, numOff);
    }
    {
        Tree323f::ValueOffIter iter = tree.beginValueOff();
        CPPUNIT_ASSERT(iter.test());

        // Read all inactive tile and voxel values through a non-const value iterator
        // and overwrite the values.
        size_t numOff = 0;
        for ( ; iter; ++iter, ++numOff) {
            iter.setValue(iter.getValue() - 5);
            iter.setValueOff();
        }
        for (iter = tree.beginValueOff(); iter; ++iter, --numOff);
        CPPUNIT_ASSERT_EQUAL(size_t(0), numOff);
    }
}


void
TestTreeIterators::testModifyValue()
{
    using openvdb::Coord;

    const openvdb::Index DIM0 = 3, DIM1 = 2, DIM2 = 3;
    {
        typedef openvdb::tree::Tree4<int32_t, DIM2, DIM1, DIM0>::Type IntTree323f;

        IntTree323f tree(/*background=*/256);
        tree.addTile(/*level=*/3, Coord(-1),                 /*value=*/ 4, /*active=*/true);
        tree.addTile(/*level=*/2, Coord(1 << (DIM0 + DIM1)), /*value=*/-3, /*active=*/true);
        tree.addTile(/*level=*/1, Coord(1 << DIM0),          /*value=*/ 2, /*active=*/true);
        tree.addTile(/*level=*/0, Coord(0),                  /*value=*/-1, /*active=*/true);

        struct Local { static inline void negate(int32_t& n) { n = -n; } };

        for (IntTree323f::ValueAllIter iter = tree.beginValueAll(); iter; ++iter) {
            iter.modifyValue(Local::negate);
        }

        for (IntTree323f::ValueAllCIter iter = tree.cbeginValueAll(); iter; ++iter) {
            const int32_t val = *iter;
            if (val < 0) CPPUNIT_ASSERT((-val) % 2 == 0); // negative values are even
            else CPPUNIT_ASSERT(val % 2 == 1); // positive values are odd
        }

        // Because modifying values through a const iterator is not allowed,
        // uncommenting the following line should result in a static assertion failure:
        //tree.cbeginValueOn().modifyValue(Local::negate);
    }
    {
        typedef openvdb::tree::Tree4<bool, DIM2, DIM1, DIM0>::Type BoolTree323f;

        BoolTree323f tree;
        tree.addTile(/*level=*/3, Coord(-1),                 /*value=*/false, /*active=*/true);
        tree.addTile(/*level=*/2, Coord(1 << (DIM0 + DIM1)), /*value=*/ true, /*active=*/true);
        tree.addTile(/*level=*/1, Coord(1 << DIM0),          /*value=*/false, /*active=*/true);
        tree.addTile(/*level=*/0, Coord(0),                  /*value=*/ true, /*active=*/true);

        struct Local { static inline void negate(bool& b) { b = !b; } };

        for (BoolTree323f::ValueAllIter iter = tree.beginValueAll(); iter; ++iter) {
            iter.modifyValue(Local::negate);
        }

        CPPUNIT_ASSERT(!tree.getValue(Coord(0)));
        CPPUNIT_ASSERT( tree.getValue(Coord(1 << DIM0)));
        CPPUNIT_ASSERT(!tree.getValue(Coord(1 << (DIM0 + DIM1))));
        CPPUNIT_ASSERT( tree.getValue(Coord(-1)));

        // Because modifying values through a const iterator is not allowed,
        // uncommenting the following line should result in a static assertion failure:
        //tree.cbeginValueOn().modifyValue(Local::negate);
    }
    {
        typedef openvdb::tree::Tree4<std::string, DIM2, DIM1, DIM0>::Type StringTree323f;

        StringTree323f tree(/*background=*/"");
        tree.addTile(/*level=*/3, Coord(-1),                 /*value=*/"abc", /*active=*/true);
        tree.addTile(/*level=*/2, Coord(1 << (DIM0 + DIM1)), /*value=*/"abc", /*active=*/true);
        tree.addTile(/*level=*/1, Coord(1 << DIM0),          /*value=*/"abc", /*active=*/true);
        tree.addTile(/*level=*/0, Coord(0),                  /*value=*/"abc", /*active=*/true);

        struct Local { static inline void op(std::string& s) { s.append("def"); } };

        for (StringTree323f::ValueOnIter iter = tree.beginValueOn(); iter; ++iter) {
            iter.modifyValue(Local::op);
        }

        const std::string expectedVal("abcdef");
        for (StringTree323f::ValueOnCIter iter = tree.cbeginValueOn(); iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(expectedVal, *iter);
        }
        for (StringTree323f::ValueOffCIter iter = tree.cbeginValueOff(); iter; ++iter) {
            CPPUNIT_ASSERT((*iter).empty());
        }
    }
}


void
TestTreeIterators::testDepthBounds()
{
    const openvdb::Index DIM0 = 3, DIM1 = 2, DIM2 = 3;

    typedef openvdb::tree::Tree4<float, DIM2, DIM1, DIM0>::Type Tree323f;

    Tree323f tree(/*fillValue=*/256.0f);
    tree.setValue(openvdb::Coord(4), 0.0f);
    tree.setValue(openvdb::Coord(-4), -1.0f);

    const size_t
        numDepth1 = 2 * ((1 << (3 * DIM2)) - 1), // 2 8x8x8 InternalNodes - 1 child pointer each
        numDepth2 = 2 * ((1 << (3 * DIM1)) - 1), // 2 4x4x4 InternalNodes - 1 child pointer each
        numDepth3 = 2 * ((1 << (3 * DIM0)) - 1), // 2 8x8x8 LeafNodes - 1 active value each
        expectedNumOff = numDepth1 + numDepth2 + numDepth3;

    {
        Tree323f::ValueOffCIter iter = tree.cbeginValueOff();
        CPPUNIT_ASSERT(iter.test());

        // Read all inactive tile and voxel values through a non-const value iterator.
        size_t numOff = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(!iter.isValueOn());
            ++numOff;
        }
        CPPUNIT_ASSERT_EQUAL(expectedNumOff, numOff);
    }
    {
        // Repeat, setting the minimum iterator depth to 2.
        Tree323f::ValueOffCIter iter = tree.cbeginValueOff();
        CPPUNIT_ASSERT(iter.test());

        iter.setMinDepth(2);
        CPPUNIT_ASSERT(iter.test());

        size_t numOff = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(!iter.isValueOn());
            ++numOff;
            const int depth = iter.getDepth();
            CPPUNIT_ASSERT(depth > 1);
        }
        CPPUNIT_ASSERT_EQUAL(expectedNumOff - numDepth1, numOff);
    }
    {
        // Repeat, setting the minimum and maximum depths to 2.
        Tree323f::ValueOffCIter iter = tree.cbeginValueOff();
        CPPUNIT_ASSERT(iter.test());

        iter.setMinDepth(2);
        CPPUNIT_ASSERT(iter.test());

        iter.setMaxDepth(2);
        CPPUNIT_ASSERT(iter.test());

        size_t numOff = 0;
        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT(!iter.isValueOn());
            ++numOff;
            const int depth = iter.getDepth();
            CPPUNIT_ASSERT_EQUAL(2, depth);
        }
        CPPUNIT_ASSERT_EQUAL(expectedNumOff - numDepth1 - numDepth3, numOff);
    }
    {
        // FX-7884 regression test
        using namespace openvdb;

        const float radius = 4.3f, voxelSize = 0.1f, width = 2.0f;
        const Vec3f center(15.8f, 13.2f, 16.7f);
        FloatGrid::Ptr sphereGrid = tools::createLevelSetSphere<FloatGrid>(
            radius, center, voxelSize, width);
        const FloatTree& sphereTree = sphereGrid->tree();

        FloatGrid::ValueOffIter iter = sphereGrid->beginValueOff();
        iter.setMaxDepth(2);
        for ( ; iter; ++iter) {
            const Coord ijk = iter.getCoord();
            ASSERT_DOUBLES_EXACTLY_EQUAL(sphereTree.getValue(ijk), *iter);
        }
    }
    {
        // FX-10221 regression test
        // This code generated an infinite loop in OpenVDB 5.1.0 and earlier:
        openvdb::FloatTree emptyTree;
        auto iter = emptyTree.cbeginValueAll();
        iter.setMinDepth(2);
        CPPUNIT_ASSERT(!iter);
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
