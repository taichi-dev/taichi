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


class TestNodeIterator: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestNodeIterator);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST(testSinglePositive);
    CPPUNIT_TEST(testSingleNegative);
    CPPUNIT_TEST(testMultipleBlocks);
    CPPUNIT_TEST(testDepthBounds);
    CPPUNIT_TEST_SUITE_END();

    void testEmpty();
    void testSinglePositive();
    void testSingleNegative();
    void testMultipleBlocks();
    void testDepthBounds();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestNodeIterator);

namespace {
typedef openvdb::tree::Tree4<float, 3, 2, 3>::Type Tree323f;
}


////////////////////////////////////////


void
TestNodeIterator::testEmpty()
{
    Tree323f tree(/*fillValue=*/256.0f);
    {
        Tree323f::NodeCIter iter(tree);
        CPPUNIT_ASSERT(!iter.next());
    }
    {
        tree.setValue(openvdb::Coord(8, 16, 24), 10.f);
        Tree323f::NodeIter iter(tree); // non-const
        CPPUNIT_ASSERT(iter);

        // Try modifying the tree through a non-const iterator.
        Tree323f::RootNodeType* root = NULL;
        iter.getNode(root);
        CPPUNIT_ASSERT(root != NULL);
        root->clear();

        // Verify that the tree is now empty.
        iter = Tree323f::NodeIter(tree);
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT(!iter.next());
    }
}


void
TestNodeIterator::testSinglePositive()
{
    {
        Tree323f tree(/*fillValue=*/256.0f);

        tree.setValue(openvdb::Coord(8, 16, 24), 10.f);

        Tree323f::NodeCIter iter(tree);

        CPPUNIT_ASSERT(Tree323f::LeafNodeType::DIM == 8);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(0U, iter.getDepth());
        CPPUNIT_ASSERT_EQUAL(tree.treeDepth(), 1 + iter.getLevel());
        openvdb::CoordBBox range, bbox;
        tree.getIndexRange(range);
        iter.getBoundingBox(bbox);
        CPPUNIT_ASSERT_EQUAL(bbox.min(), range.min());
        CPPUNIT_ASSERT_EQUAL(bbox.max(), range.max());

        // Descend to the depth-1 internal node with bounding box
        // (0, 0, 0) -> (255, 255, 255) containing voxel (8, 16, 24).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());
        iter.getBoundingBox(bbox);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0), bbox.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord((1 << (3 + 2 + 3)) - 1), bbox.max());

        // Descend to the depth-2 internal node with bounding box
        // (0, 0, 0) -> (31, 31, 31) containing voxel (8, 16, 24).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());
        iter.getBoundingBox(bbox);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0), bbox.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord((1 << (2 + 3)) - 1), bbox.max());

        // Descend to the leaf node with bounding box (8, 16, 24) -> (15, 23, 31)
        // containing voxel (8, 16, 24).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());
        iter.getBoundingBox(bbox);
        range.min().reset(8, 16, 24);
        range.max() = range.min().offsetBy((1 << 3) - 1); // add leaf node size
        CPPUNIT_ASSERT_EQUAL(range.min(), bbox.min());
        CPPUNIT_ASSERT_EQUAL(range.max(), bbox.max());

        iter.next();
        CPPUNIT_ASSERT(!iter);
    }
    {
        Tree323f tree(/*fillValue=*/256.0f);

        tree.setValue(openvdb::Coord(129), 10.f);

        Tree323f::NodeCIter iter(tree);

        CPPUNIT_ASSERT(Tree323f::LeafNodeType::DIM == 8);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(0U, iter.getDepth());
        CPPUNIT_ASSERT_EQUAL(tree.treeDepth(), 1 + iter.getLevel());
        openvdb::CoordBBox range, bbox;
        tree.getIndexRange(range);
        iter.getBoundingBox(bbox);
        CPPUNIT_ASSERT_EQUAL(bbox.min(), range.min());
        CPPUNIT_ASSERT_EQUAL(bbox.max(), range.max());

        // Descend to the depth-1 internal node with bounding box
        // (0, 0, 0) -> (255, 255, 255) containing voxel (129, 129, 129).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());
        iter.getBoundingBox(bbox);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0), bbox.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord((1 << (3 + 2 + 3)) - 1), bbox.max());

        // Descend to the depth-2 internal node with bounding box
        // (128, 128, 128) -> (159, 159, 159) containing voxel (129, 129, 129).
        // (128 is the nearest multiple of 32 less than 129.)
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());
        iter.getBoundingBox(bbox);
        range.min().reset(128, 128, 128);
        CPPUNIT_ASSERT_EQUAL(range.min(), bbox.min());
        CPPUNIT_ASSERT_EQUAL(range.min().offsetBy((1 << (2 + 3)) - 1), bbox.max());

        // Descend to the leaf node with bounding box
        // (128, 128, 128) -> (135, 135, 135) containing voxel (129, 129, 129).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());
        iter.getBoundingBox(bbox);
        range.max() = range.min().offsetBy((1 << 3) - 1); // add leaf node size
        CPPUNIT_ASSERT_EQUAL(range.min(), bbox.min());
        CPPUNIT_ASSERT_EQUAL(range.max(), bbox.max());

        iter.next();
        CPPUNIT_ASSERT(!iter);
    }
}


void
TestNodeIterator::testSingleNegative()
{
    Tree323f tree(/*fillValue=*/256.0f);

    tree.setValue(openvdb::Coord(-1), 10.f);

    Tree323f::NodeCIter iter(tree);

    CPPUNIT_ASSERT(Tree323f::LeafNodeType::DIM == 8);

    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(0U, iter.getDepth());
    CPPUNIT_ASSERT_EQUAL(tree.treeDepth(), 1 + iter.getLevel());
    openvdb::CoordBBox range, bbox;
    tree.getIndexRange(range);
    iter.getBoundingBox(bbox);
    CPPUNIT_ASSERT_EQUAL(bbox.min(), range.min());
    CPPUNIT_ASSERT_EQUAL(bbox.max(), range.max());

    // Descend to the depth-1 internal node with bounding box
    // (-256, -256, -256) -> (-1, -1, -1) containing voxel (-1, -1, -1).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());
    iter.getBoundingBox(bbox);
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-(1 << (3 + 2 + 3))), bbox.min());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-1), bbox.max());

    // Descend to the depth-2 internal node with bounding box
    // (-32, -32, -32) -> (-1, -1, -1) containing voxel (-1, -1, -1).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());
    iter.getBoundingBox(bbox);
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-(1 << (2 + 3))), bbox.min());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-1), bbox.max());

    // Descend to the leaf node with bounding box (-8, -8, -8) -> (-1, -1, -1)
    // containing voxel (-1, -1, -1).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());
    iter.getBoundingBox(bbox);
    range.max().reset(-1, -1, -1);
    range.min() = range.max().offsetBy(-((1 << 3) - 1)); // add leaf node size
    CPPUNIT_ASSERT_EQUAL(range.min(), bbox.min());
    CPPUNIT_ASSERT_EQUAL(range.max(), bbox.max());

    iter.next();
    CPPUNIT_ASSERT(!iter);
}


void
TestNodeIterator::testMultipleBlocks()
{
    Tree323f tree(/*fillValue=*/256.0f);

    tree.setValue(openvdb::Coord(-1), 10.f);
    tree.setValue(openvdb::Coord(129), 10.f);

    Tree323f::NodeCIter iter(tree);

    CPPUNIT_ASSERT(Tree323f::LeafNodeType::DIM == 8);

    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(0U, iter.getDepth());
    CPPUNIT_ASSERT_EQUAL(tree.treeDepth(), 1 + iter.getLevel());

    // Descend to the depth-1 internal node with bounding box
    // (-256, -256, -256) -> (-1, -1, -1) containing voxel (-1, -1, -1).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());

    // Descend to the depth-2 internal node with bounding box
    // (-32, -32, -32) -> (-1, -1, -1) containing voxel (-1, -1, -1).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());

    // Descend to the leaf node with bounding box (-8, -8, -8) -> (-1, -1, -1)
    // containing voxel (-1, -1, -1).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());
    openvdb::Coord expectedMin, expectedMax(-1, -1, -1);
    expectedMin = expectedMax.offsetBy(-((1 << 3) - 1)); // add leaf node size
    openvdb::CoordBBox bbox;
    iter.getBoundingBox(bbox);
    CPPUNIT_ASSERT_EQUAL(expectedMin, bbox.min());
    CPPUNIT_ASSERT_EQUAL(expectedMax, bbox.max());

    // Ascend to the depth-1 internal node with bounding box (0, 0, 0) -> (255, 255, 255)
    // containing voxel (129, 129, 129).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());

    // Descend to the depth-2 internal node with bounding box
    // (128, 128, 128) -> (159, 159, 159) containing voxel (129, 129, 129).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());

    // Descend to the leaf node with bounding box (128, 128, 128) -> (135, 135, 135)
    // containing voxel (129, 129, 129).
    iter.next();
    CPPUNIT_ASSERT(iter);
    CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());
    expectedMin.reset(128, 128, 128);
    expectedMax = expectedMin.offsetBy((1 << 3) - 1); // add leaf node size
    iter.getBoundingBox(bbox);
    CPPUNIT_ASSERT_EQUAL(expectedMin, bbox.min());
    CPPUNIT_ASSERT_EQUAL(expectedMax, bbox.max());

    iter.next();
    CPPUNIT_ASSERT(!iter);
}


void
TestNodeIterator::testDepthBounds()
{
    Tree323f tree(/*fillValue=*/256.0f);

    tree.setValue(openvdb::Coord(-1), 10.f);
    tree.setValue(openvdb::Coord(129), 10.f);

    {
        // Iterate over internal nodes only.
        Tree323f::NodeCIter iter(tree);
        iter.setMaxDepth(2);
        iter.setMinDepth(1);

        // Begin at the depth-1 internal node with bounding box
        // (-256, -256, -256) -> (-1, -1, -1) containing voxel (-1, -1, -1).
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());

        // Descend to the depth-2 internal node with bounding box
        // (-32, -32, -32) -> (-1, -1, -1) containing voxel (-1, -1, -1).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());

        // Skipping the leaf node, ascend to the depth-1 internal node with bounding box
        // (0, 0, 0) -> (255, 255, 255) containing voxel (129, 129, 129).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());

        // Descend to the depth-2 internal node with bounding box
        // (128, 128, 128) -> (159, 159, 159) containing voxel (129, 129, 129).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(2U, iter.getDepth());

        // Verify that no internal nodes remain unvisited.
        iter.next();
        CPPUNIT_ASSERT(!iter);
    }
    {
        // Iterate over depth-1 internal nodes only.
        Tree323f::NodeCIter iter(tree);
        iter.setMaxDepth(1);
        iter.setMinDepth(1);

        // Begin at the depth-1 internal node with bounding box
        // (-256, -256, -256) -> (-1, -1, -1) containing voxel (-1, -1, -1).
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());

        // Skip to the depth-1 internal node with bounding box
        // (0, 0, 0) -> (255, 255, 255) containing voxel (129, 129, 129).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(1U, iter.getDepth());

        // Verify that no depth-1 nodes remain unvisited.
        iter.next();
        CPPUNIT_ASSERT(!iter);
    }
    {
        // Iterate over leaf nodes only.
        Tree323f::NodeCIter iter = tree.cbeginNode();
        iter.setMaxDepth(3);
        iter.setMinDepth(3);

        // Begin at the leaf node with bounding box (-8, -8, -8) -> (-1, -1, -1)
        // containing voxel (-1, -1, -1).
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());

        // Skip to the leaf node with bounding box (128, 128, 128) -> (135, 135, 135)
        // containing voxel (129, 129, 129).
        iter.next();
        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(0U, iter.getLevel());

        // Verify that no leaf nodes remain unvisited.
        iter.next();
        CPPUNIT_ASSERT(!iter);
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
