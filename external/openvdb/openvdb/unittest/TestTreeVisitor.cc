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
//
/// @file TestTreeVisitor.h
///
/// @author Peter Cucka

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>


class TestTreeVisitor: public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestTreeVisitor);
    CPPUNIT_TEST(testVisitTreeBool);
    CPPUNIT_TEST(testVisitTreeInt32);
    CPPUNIT_TEST(testVisitTreeFloat);
    CPPUNIT_TEST(testVisitTreeVec2I);
    CPPUNIT_TEST(testVisitTreeVec3S);
    CPPUNIT_TEST(testVisit2Trees);
    CPPUNIT_TEST_SUITE_END();

    void testVisitTreeBool() { visitTree<openvdb::BoolTree>(); }
    void testVisitTreeInt32() { visitTree<openvdb::Int32Tree>(); }
    void testVisitTreeFloat() { visitTree<openvdb::FloatTree>(); }
    void testVisitTreeVec2I() { visitTree<openvdb::Vec2ITree>(); }
    void testVisitTreeVec3S() { visitTree<openvdb::VectorTree>(); }
    void testVisit2Trees();

private:
    template<typename TreeT> TreeT createTestTree() const;
    template<typename TreeT> void visitTree();
};


CPPUNIT_TEST_SUITE_REGISTRATION(TestTreeVisitor);


////////////////////////////////////////


template<typename TreeT>
TreeT
TestTreeVisitor::createTestTree() const
{
    using ValueT = typename TreeT::ValueType;
    const ValueT zero = openvdb::zeroVal<ValueT>(), one = zero + 1;

    // Create a sparse test tree comprising the eight corners of
    // a 200 x 200 x 200 cube.
    TreeT tree(/*background=*/one);
    tree.setValue(openvdb::Coord(  0,   0,   0),  /*value=*/zero);
    tree.setValue(openvdb::Coord(200,   0,   0),  zero);
    tree.setValue(openvdb::Coord(  0, 200,   0),  zero);
    tree.setValue(openvdb::Coord(  0,   0, 200),  zero);
    tree.setValue(openvdb::Coord(200,   0, 200),  zero);
    tree.setValue(openvdb::Coord(  0, 200, 200),  zero);
    tree.setValue(openvdb::Coord(200, 200,   0),  zero);
    tree.setValue(openvdb::Coord(200, 200, 200),  zero);

    // Verify that the bounding box of all On values is 200 x 200 x 200.
    openvdb::CoordBBox bbox;
    CPPUNIT_ASSERT(tree.evalActiveVoxelBoundingBox(bbox));
    CPPUNIT_ASSERT(bbox.min() == openvdb::Coord(0, 0, 0));
    CPPUNIT_ASSERT(bbox.max() == openvdb::Coord(200, 200, 200));

    return tree;
}


////////////////////////////////////////


namespace {

/// Single-tree visitor that accumulates node counts
class Visitor
{
public:
    using NodeMap = std::map<openvdb::Index, std::set<const void*> >;

    Visitor(): mSkipLeafNodes(false) { reset(); }

    void reset()
    {
        mSkipLeafNodes = false;
        mNodes.clear();
        mNonConstIterUseCount = mConstIterUseCount = 0;
    }

    void setSkipLeafNodes(bool b) { mSkipLeafNodes = b; }

    template<typename IterT>
    bool operator()(IterT& iter)
    {
        incrementIterUseCount(std::is_const<typename IterT::NodeType>::value);
        CPPUNIT_ASSERT(iter.getParentNode() != nullptr);

        if (mSkipLeafNodes && iter.parent().getLevel() == 1) return true;

        using ValueT = typename IterT::NonConstValueType;
        using ChildT = typename IterT::ChildNodeType;
        ValueT value;
        if (const ChildT* child = iter.probeChild(value)) {
            insertChild<ChildT>(child);
        }
        return false;
    }

    openvdb::Index leafCount() const
    {
        NodeMap::const_iterator it = mNodes.find(0);
        return openvdb::Index((it != mNodes.end()) ? it->second.size() : 0);
    }
    openvdb::Index nonLeafCount() const
    {
        openvdb::Index count = 1; // root node
        for (NodeMap::const_iterator i = mNodes.begin(), e = mNodes.end(); i != e; ++i) {
            if (i->first != 0) count = openvdb::Index(count + i->second.size());
        }
        return count;
    }

    bool usedOnlyConstIterators() const
    {
        return (mConstIterUseCount > 0 && mNonConstIterUseCount == 0);
    }
    bool usedOnlyNonConstIterators() const
    {
        return (mConstIterUseCount == 0 && mNonConstIterUseCount > 0);
    }

private:
    template<typename ChildT>
    void insertChild(const ChildT* child)
    {
        if (child != nullptr) {
            const openvdb::Index level = child->getLevel();
            if (!mSkipLeafNodes || level > 0) {
                mNodes[level].insert(child);
            }
        }
    }

    void incrementIterUseCount(bool isConst)
    {
        if (isConst) ++mConstIterUseCount; else ++mNonConstIterUseCount;
    }

    bool mSkipLeafNodes;
    NodeMap mNodes;
    int mNonConstIterUseCount, mConstIterUseCount;
};

/// Specialization for LeafNode iterators, whose ChildNodeType is void
/// (therefore can't call child->getLevel())
template<> inline void Visitor::insertChild<void>(const void*) {}

} // unnamed namespace


template<typename TreeT>
void
TestTreeVisitor::visitTree()
{
    TreeT tree = createTestTree<TreeT>();
    {
        // Traverse the tree, accumulating node counts.
        Visitor visitor;
        const_cast<const TreeT&>(tree).visit(visitor);

        CPPUNIT_ASSERT(visitor.usedOnlyConstIterators());
        CPPUNIT_ASSERT_EQUAL(tree.leafCount(), visitor.leafCount());
        CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.nonLeafCount());
    }
    {
        // Traverse the tree, accumulating node counts as above,
        // but using non-const iterators.
        Visitor visitor;
        tree.visit(visitor);

        CPPUNIT_ASSERT(visitor.usedOnlyNonConstIterators());
        CPPUNIT_ASSERT_EQUAL(tree.leafCount(), visitor.leafCount());
        CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.nonLeafCount());
    }
    {
        // Traverse the tree, accumulating counts of non-leaf nodes only.
        Visitor visitor;
        visitor.setSkipLeafNodes(true);
        const_cast<const TreeT&>(tree).visit(visitor);

        CPPUNIT_ASSERT(visitor.usedOnlyConstIterators());
        CPPUNIT_ASSERT_EQUAL(0U, visitor.leafCount()); // leaf nodes were skipped
        CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.nonLeafCount());
    }
}


////////////////////////////////////////


namespace {

/// Two-tree visitor that accumulates node counts
class Visitor2
{
public:
    using NodeMap = std::map<openvdb::Index, std::set<const void*> >;

    Visitor2() { reset(); }

    void reset()
    {
        mSkipALeafNodes = mSkipBLeafNodes = false;
        mANodeCount.clear();
        mBNodeCount.clear();
    }

    void setSkipALeafNodes(bool b) { mSkipALeafNodes = b; }
    void setSkipBLeafNodes(bool b) { mSkipBLeafNodes = b; }

    openvdb::Index aLeafCount() const { return leafCount(/*useA=*/true); }
    openvdb::Index bLeafCount() const { return leafCount(/*useA=*/false); }
    openvdb::Index aNonLeafCount() const { return nonLeafCount(/*useA=*/true); }
    openvdb::Index bNonLeafCount() const { return nonLeafCount(/*useA=*/false); }

    template<typename AIterT, typename BIterT>
    int operator()(AIterT& aIter, BIterT& bIter)
    {
        CPPUNIT_ASSERT(aIter.getParentNode() != nullptr);
        CPPUNIT_ASSERT(bIter.getParentNode() != nullptr);

        typename AIterT::NodeType& aNode = aIter.parent();
        typename BIterT::NodeType& bNode = bIter.parent();

        const openvdb::Index aLevel = aNode.getLevel(), bLevel = bNode.getLevel();
        mANodeCount[aLevel].insert(&aNode);
        mBNodeCount[bLevel].insert(&bNode);

        int skipBranch = 0;
        if (aLevel == 1 && mSkipALeafNodes) skipBranch = (skipBranch | 1);
        if (bLevel == 1 && mSkipBLeafNodes) skipBranch = (skipBranch | 2);
        return skipBranch;
    }

private:
    openvdb::Index leafCount(bool useA) const
    {
        const NodeMap& theMap = (useA ? mANodeCount : mBNodeCount);
        NodeMap::const_iterator it = theMap.find(0);
        if (it != theMap.end()) return openvdb::Index(it->second.size());
        return 0;
    }
    openvdb::Index nonLeafCount(bool useA) const
    {
        openvdb::Index count = 0;
        const NodeMap& theMap = (useA ? mANodeCount : mBNodeCount);
        for (NodeMap::const_iterator i = theMap.begin(), e = theMap.end(); i != e; ++i) {
            if (i->first != 0) count = openvdb::Index(count + i->second.size());
        }
        return count;
    }

    bool mSkipALeafNodes, mSkipBLeafNodes;
    NodeMap mANodeCount, mBNodeCount;
};

} // unnamed namespace


void
TestTreeVisitor::testVisit2Trees()
{
    using TreeT = openvdb::FloatTree;
    using Tree2T = openvdb::VectorTree;
    using ValueT = TreeT::ValueType;

    // Create a test tree.
    TreeT tree = createTestTree<TreeT>();
    // Create another test tree of a different type but with the same topology.
    Tree2T tree2 = createTestTree<Tree2T>();

    // Traverse both trees.
    Visitor2 visitor;
    tree.visit2(tree2, visitor);

    //CPPUNIT_ASSERT(visitor.usedOnlyConstIterators());
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), visitor.aLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), visitor.bLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.aNonLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.nonLeafCount(), visitor.bNonLeafCount());

    visitor.reset();

    // Change the topology of the first tree.
    tree.setValue(openvdb::Coord(-200, -200, -200), openvdb::zeroVal<ValueT>());

    // Traverse both trees.
    tree.visit2(tree2, visitor);

    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), visitor.aLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), visitor.bLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.aNonLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.nonLeafCount(), visitor.bNonLeafCount());

    visitor.reset();

    // Traverse the two trees in the opposite order.
    tree2.visit2(tree, visitor);

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), visitor.aLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), visitor.bLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.nonLeafCount(), visitor.aNonLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.bNonLeafCount());

    // Repeat, skipping leaf nodes of tree2.
    visitor.reset();
    visitor.setSkipALeafNodes(true);
    tree2.visit2(tree, visitor);

    CPPUNIT_ASSERT_EQUAL(0U, visitor.aLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), visitor.bLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.nonLeafCount(), visitor.aNonLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.bNonLeafCount());

    // Repeat, skipping leaf nodes of tree.
    visitor.reset();
    visitor.setSkipBLeafNodes(true);
    tree2.visit2(tree, visitor);

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), visitor.aLeafCount());
    CPPUNIT_ASSERT_EQUAL(0U, visitor.bLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree2.nonLeafCount(), visitor.aNonLeafCount());
    CPPUNIT_ASSERT_EQUAL(tree.nonLeafCount(), visitor.bNonLeafCount());
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
