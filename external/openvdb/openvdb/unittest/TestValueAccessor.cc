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
#include <tbb/task.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Prune.h>
#include <type_traits>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);


using ValueType = float;
using Tree2Type = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::LeafNode<ValueType, 3> > >;
using Tree3Type = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::InternalNode<
    openvdb::tree::LeafNode<ValueType, 3>, 4> > >;
using Tree4Type = openvdb::tree::Tree4<ValueType, 5, 4, 3>::Type;
using Tree5Type = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::InternalNode<
    openvdb::tree::InternalNode<
    openvdb::tree::InternalNode<
    openvdb::tree::LeafNode<ValueType, 3>, 4>, 5>, 5> > >;
using TreeType = Tree4Type;


using namespace openvdb::tree;

class TestValueAccessor: public CppUnit::TestFixture
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestValueAccessor);

    CPPUNIT_TEST(testTree2Accessor);
    CPPUNIT_TEST(testTree2AccessorRW);
    CPPUNIT_TEST(testTree2ConstAccessor);
    CPPUNIT_TEST(testTree2ConstAccessorRW);

    CPPUNIT_TEST(testTree3Accessor);
    CPPUNIT_TEST(testTree3AccessorRW);
    CPPUNIT_TEST(testTree3ConstAccessor);
    CPPUNIT_TEST(testTree3ConstAccessorRW);

    CPPUNIT_TEST(testTree4Accessor);
    CPPUNIT_TEST(testTree4AccessorRW);
    CPPUNIT_TEST(testTree4ConstAccessor);
    CPPUNIT_TEST(testTree4ConstAccessorRW);

    CPPUNIT_TEST(testTree5Accessor);
    CPPUNIT_TEST(testTree5AccessorRW);
    CPPUNIT_TEST(testTree5ConstAccessor);
    CPPUNIT_TEST(testTree5ConstAccessorRW);

    CPPUNIT_TEST(testTree3Accessor2);
    CPPUNIT_TEST(testTree3ConstAccessor2);
    CPPUNIT_TEST(testTree4Accessor2);
    CPPUNIT_TEST(testTree4ConstAccessor2);
    CPPUNIT_TEST(testTree4Accessor1);
    CPPUNIT_TEST(testTree4ConstAccessor1);
    CPPUNIT_TEST(testTree4Accessor0);
    CPPUNIT_TEST(testTree4ConstAccessor0);
    CPPUNIT_TEST(testTree5Accessor2);
    CPPUNIT_TEST(testTree5ConstAccessor2);
    CPPUNIT_TEST(testTree4Accessor12);//cache node level 2
    CPPUNIT_TEST(testTree5Accessor213);//cache node level 1 and 3

    CPPUNIT_TEST(testMultithreadedAccessor);
    CPPUNIT_TEST(testAccessorRegistration);
    CPPUNIT_TEST(testGetNode);

    CPPUNIT_TEST_SUITE_END();
    // cache all node levels
    void testTree2Accessor()        { accessorTest<ValueAccessor<Tree2Type> >(); }
    void testTree2AccessorRW()      { accessorTest<ValueAccessorRW<Tree2Type> >(); }
    void testTree2ConstAccessor()   { constAccessorTest<ValueAccessor<const Tree2Type> >(); }
    void testTree2ConstAccessorRW() { constAccessorTest<ValueAccessorRW<const Tree2Type> >(); }
    // cache all node levels
    void testTree3Accessor()        { accessorTest<ValueAccessor<Tree3Type> >(); }
    void testTree3AccessorRW()      { accessorTest<ValueAccessorRW<Tree3Type> >(); }
    void testTree3ConstAccessor()   { constAccessorTest<ValueAccessor<const Tree3Type> >(); }
    void testTree3ConstAccessorRW() { constAccessorTest<ValueAccessorRW<const Tree3Type> >(); }
    // cache all node levels
    void testTree4Accessor()        { accessorTest<ValueAccessor<Tree4Type> >(); }
    void testTree4AccessorRW()      { accessorTest<ValueAccessorRW<Tree4Type> >(); }
    void testTree4ConstAccessor()   { constAccessorTest<ValueAccessor<const Tree4Type> >(); }
    void testTree4ConstAccessorRW() { constAccessorTest<ValueAccessorRW<const Tree4Type> >(); }
    // cache all node levels
    void testTree5Accessor()        { accessorTest<ValueAccessor<Tree5Type> >(); }
    void testTree5AccessorRW()      { accessorTest<ValueAccessorRW<Tree5Type> >(); }
    void testTree5ConstAccessor()   { constAccessorTest<ValueAccessor<const Tree5Type> >(); }
    void testTree5ConstAccessorRW() { constAccessorTest<ValueAccessorRW<const Tree5Type> >(); }

    // Test odd combinations of trees and ValueAccessors
    // cache node level 0 and 1
    void testTree3Accessor2()
    {
        accessorTest<ValueAccessor<Tree3Type, true,  2> >();
        accessorTest<ValueAccessor<Tree3Type, false, 2> >();
    }
    void testTree3ConstAccessor2()
    {
        constAccessorTest<ValueAccessor<const Tree3Type, true,  2> >();
        constAccessorTest<ValueAccessor<const Tree3Type, false, 2> >();
    }
    void testTree4Accessor2()
    {
        accessorTest<ValueAccessor<Tree4Type, true,  2> >();
        accessorTest<ValueAccessor<Tree4Type, false, 2> >();
    }
    void testTree4ConstAccessor2()
    {
        constAccessorTest<ValueAccessor<const Tree4Type, true,  2> >();
        constAccessorTest<ValueAccessor<const Tree4Type, false, 2> >();
    }
    void testTree5Accessor2()
    {
        accessorTest<ValueAccessor<Tree5Type, true,  2> >();
        accessorTest<ValueAccessor<Tree5Type, false, 2> >();
    }
    void testTree5ConstAccessor2()
    {
        constAccessorTest<ValueAccessor<const Tree5Type, true,  2> >();
        constAccessorTest<ValueAccessor<const Tree5Type, false, 2> >();
    }
    // only cache leaf level
    void testTree4Accessor1()
    {
        accessorTest<ValueAccessor<Tree5Type, true,  1> >();
        accessorTest<ValueAccessor<Tree5Type, false, 1> >();
    }
    void testTree4ConstAccessor1()
    {
        constAccessorTest<ValueAccessor<const Tree5Type, true,  1> >();
        constAccessorTest<ValueAccessor<const Tree5Type, false, 1> >();
    }
    // disable node caching
    void testTree4Accessor0()
    {
        accessorTest<ValueAccessor<Tree5Type, true,  0> >();
        accessorTest<ValueAccessor<Tree5Type, false, 0> >();
    }
    void testTree4ConstAccessor0()
    {
        constAccessorTest<ValueAccessor<const Tree5Type, true,  0> >();
        constAccessorTest<ValueAccessor<const Tree5Type, false, 0> >();
    }
    //cache node level 2
    void testTree4Accessor12()
    {
        accessorTest<ValueAccessor1<Tree4Type, true,  2> >();
        accessorTest<ValueAccessor1<Tree4Type, false, 2> >();
    }
    //cache node level 1 and 3
    void testTree5Accessor213()
    {
        accessorTest<ValueAccessor2<Tree5Type, true, 1,3> >();
        accessorTest<ValueAccessor2<Tree5Type, false, 1,3> >();
    }

    void testMultithreadedAccessor();
    void testAccessorRegistration();
    void testGetNode();

private:
    template<typename AccessorT> void accessorTest();
    template<typename AccessorT> void constAccessorTest();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestValueAccessor);


////////////////////////////////////////


namespace {

struct Plus
{
    float addend;
    Plus(float f): addend(f) {}
    inline void operator()(float& f) const { f += addend; }
    inline void operator()(float& f, bool& b) const { f += addend; b = false; }
};

}


template<typename AccessorT>
void
TestValueAccessor::accessorTest()
{
    using TreeType = typename AccessorT::TreeType;
    const int leafDepth = int(TreeType::DEPTH) - 1;
    // subtract one because getValueDepth() returns 0 for values at the root

    const ValueType background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20), c1(500000, 200000, 300000);

    {
        TreeType tree(background);
        CPPUNIT_ASSERT(!tree.isValueOn(c0));
        CPPUNIT_ASSERT(!tree.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c1));
        tree.setValue(c0, value);
        CPPUNIT_ASSERT(tree.isValueOn(c0));
        CPPUNIT_ASSERT(!tree.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c1));
    }
    {
        TreeType tree(background);
        AccessorT acc(tree);
        ValueType v;

        CPPUNIT_ASSERT(!tree.isValueOn(c0));
        CPPUNIT_ASSERT(!tree.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c1));
        CPPUNIT_ASSERT(!acc.isCached(c0));
        CPPUNIT_ASSERT(!acc.isCached(c1));
        CPPUNIT_ASSERT(!acc.probeValue(c0,v));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, v);
        CPPUNIT_ASSERT(!acc.probeValue(c1,v));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, v);
        CPPUNIT_ASSERT_EQUAL(-1, acc.getValueDepth(c0));
        CPPUNIT_ASSERT_EQUAL(-1, acc.getValueDepth(c1));
        CPPUNIT_ASSERT(!acc.isVoxel(c0));
        CPPUNIT_ASSERT(!acc.isVoxel(c1));

        acc.setValue(c0, value);

        CPPUNIT_ASSERT(tree.isValueOn(c0));
        CPPUNIT_ASSERT(!tree.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c1));
        CPPUNIT_ASSERT(acc.probeValue(c0,v));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, v);
        CPPUNIT_ASSERT(!acc.probeValue(c1,v));
        ASSERT_DOUBLES_EXACTLY_EQUAL(background, v);
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c0)); // leaf-level voxel value
        CPPUNIT_ASSERT_EQUAL(-1, acc.getValueDepth(c1)); // background value
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(openvdb::Coord(7, 10, 20)));
        const int depth = leafDepth == 1 ? -1 : leafDepth - 1;
        CPPUNIT_ASSERT_EQUAL(depth, acc.getValueDepth(openvdb::Coord(8, 10, 20)));
        CPPUNIT_ASSERT( acc.isVoxel(c0)); // leaf-level voxel value
        CPPUNIT_ASSERT(!acc.isVoxel(c1));
        CPPUNIT_ASSERT( acc.isVoxel(openvdb::Coord(7, 10, 20)));
        CPPUNIT_ASSERT(!acc.isVoxel(openvdb::Coord(8, 10, 20)));

        ASSERT_DOUBLES_EXACTLY_EQUAL(background, acc.getValue(c1));
        CPPUNIT_ASSERT(!acc.isCached(c1)); // uncached background value
        CPPUNIT_ASSERT(!acc.isValueOn(c1)); // inactive background value
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c0));
        CPPUNIT_ASSERT(
            (acc.numCacheLevels()>0) == acc.isCached(c0)); // active, leaf-level voxel value
        CPPUNIT_ASSERT(acc.isValueOn(c0));

        acc.setValue(c1, value);

        CPPUNIT_ASSERT(acc.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c1));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c1));
        CPPUNIT_ASSERT(!acc.isCached(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c0));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c0));
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c0));
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c1));
        CPPUNIT_ASSERT(acc.isVoxel(c0));
        CPPUNIT_ASSERT(acc.isVoxel(c1));

        tree.setValueOff(c1);

        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c1));
        CPPUNIT_ASSERT(!acc.isCached(c0));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c1));
        CPPUNIT_ASSERT( acc.isValueOn(c0));
        CPPUNIT_ASSERT(!acc.isValueOn(c1));

        acc.setValueOn(c1);

        CPPUNIT_ASSERT(!acc.isCached(c0));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c1));
        CPPUNIT_ASSERT( acc.isValueOn(c0));
        CPPUNIT_ASSERT( acc.isValueOn(c1));

        acc.modifyValueAndActiveState(c1, Plus(-value)); // subtract value & mark inactive
        CPPUNIT_ASSERT(!acc.isValueOn(c1));

        acc.modifyValue(c1, Plus(-value)); // subtract value again & mark active

        CPPUNIT_ASSERT(acc.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(-value, tree.getValue(c1));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(-value, acc.getValue(c1));
        CPPUNIT_ASSERT(!acc.isCached(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c0));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c0));
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c0));
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c1));
        CPPUNIT_ASSERT(acc.isVoxel(c0));
        CPPUNIT_ASSERT(acc.isVoxel(c1));

        acc.setValueOnly(c1, 3*value);

        CPPUNIT_ASSERT(acc.isValueOn(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, tree.getValue(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(3*value, tree.getValue(c1));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c1));
        ASSERT_DOUBLES_EXACTLY_EQUAL(3*value, acc.getValue(c1));
        CPPUNIT_ASSERT(!acc.isCached(c0));
        ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c0));
        CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c0));
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c0));
        CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c1));
        CPPUNIT_ASSERT(acc.isVoxel(c0));
        CPPUNIT_ASSERT(acc.isVoxel(c1));

        acc.clear();
        CPPUNIT_ASSERT(!acc.isCached(c0));
        CPPUNIT_ASSERT(!acc.isCached(c1));
    }
}


template<typename AccessorT>
void
TestValueAccessor::constAccessorTest()
{
    using TreeType = typename std::remove_const<typename AccessorT::TreeType>::type;
    const int leafDepth = int(TreeType::DEPTH) - 1;
        // subtract one because getValueDepth() returns 0 for values at the root

    const ValueType background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20), c1(500000, 200000, 300000);
    ValueType v;

    TreeType tree(background);
    AccessorT acc(tree);

    CPPUNIT_ASSERT(!tree.isValueOn(c0));
    CPPUNIT_ASSERT(!tree.isValueOn(c1));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c0));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(c1));
    CPPUNIT_ASSERT(!acc.isCached(c0));
    CPPUNIT_ASSERT(!acc.isCached(c1));
    CPPUNIT_ASSERT(!acc.probeValue(c0,v));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, v);
    CPPUNIT_ASSERT(!acc.probeValue(c1,v));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, v);
    CPPUNIT_ASSERT_EQUAL(-1, acc.getValueDepth(c0));
    CPPUNIT_ASSERT_EQUAL(-1, acc.getValueDepth(c1));
    CPPUNIT_ASSERT(!acc.isVoxel(c0));
    CPPUNIT_ASSERT(!acc.isVoxel(c1));

    tree.setValue(c0, value);

    CPPUNIT_ASSERT(tree.isValueOn(c0));
    CPPUNIT_ASSERT(!tree.isValueOn(c1));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, acc.getValue(c1));
    CPPUNIT_ASSERT(!acc.isCached(c1));
    CPPUNIT_ASSERT(!acc.isCached(c0));
    CPPUNIT_ASSERT(acc.isValueOn(c0));
    CPPUNIT_ASSERT(!acc.isValueOn(c1));
    CPPUNIT_ASSERT(acc.probeValue(c0,v));
    ASSERT_DOUBLES_EXACTLY_EQUAL(value, v);
    CPPUNIT_ASSERT(!acc.probeValue(c1,v));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, v);
    CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c0));
    CPPUNIT_ASSERT_EQUAL(-1, acc.getValueDepth(c1));
    CPPUNIT_ASSERT( acc.isVoxel(c0));
    CPPUNIT_ASSERT(!acc.isVoxel(c1));

    ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c0));
    CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c0));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, acc.getValue(c1));
    CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c0));
    CPPUNIT_ASSERT(!acc.isCached(c1));
    CPPUNIT_ASSERT(acc.isValueOn(c0));
    CPPUNIT_ASSERT(!acc.isValueOn(c1));

    tree.setValue(c1, value);

    ASSERT_DOUBLES_EXACTLY_EQUAL(value, acc.getValue(c1));
    CPPUNIT_ASSERT(!acc.isCached(c0));
    CPPUNIT_ASSERT((acc.numCacheLevels()>0) == acc.isCached(c1));
    CPPUNIT_ASSERT(acc.isValueOn(c0));
    CPPUNIT_ASSERT(acc.isValueOn(c1));
    CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c0));
    CPPUNIT_ASSERT_EQUAL(leafDepth, acc.getValueDepth(c1));
    CPPUNIT_ASSERT(acc.isVoxel(c0));
    CPPUNIT_ASSERT(acc.isVoxel(c1));

    // The next two lines should not compile, because the acc references a const tree:
    //acc.setValue(c1, value);
    //acc.setValueOff(c1);

    acc.clear();
    CPPUNIT_ASSERT(!acc.isCached(c0));
    CPPUNIT_ASSERT(!acc.isCached(c1));
}


void
TestValueAccessor::testMultithreadedAccessor()
{
#define MAX_COORD 5000

    using AccessorT = openvdb::tree::ValueAccessorRW<Tree4Type>;
    // Substituting the following alias typically results in assertion failures:
    //using AccessorT = openvdb::tree::ValueAccessor<Tree4Type>;

    // Task to perform multiple reads through a shared accessor
    struct ReadTask: public tbb::task {
        AccessorT& acc;
        ReadTask(AccessorT& c): acc(c) {}
        tbb::task* execute()
        {
            for (int i = -MAX_COORD; i < MAX_COORD; ++i) {
                ASSERT_DOUBLES_EXACTLY_EQUAL(double(i), acc.getValue(openvdb::Coord(i)));
            }
            return nullptr;
        }
    };
    // Task to perform multiple writes through a shared accessor
    struct WriteTask: public tbb::task {
        AccessorT& acc;
        WriteTask(AccessorT& c): acc(c) {}
        tbb::task* execute()
        {
            for (int i = -MAX_COORD; i < MAX_COORD; ++i) {
                float f = acc.getValue(openvdb::Coord(i));
                ASSERT_DOUBLES_EXACTLY_EQUAL(float(i), f);
                acc.setValue(openvdb::Coord(i), float(i));
                ASSERT_DOUBLES_EXACTLY_EQUAL(float(i), acc.getValue(openvdb::Coord(i)));
            }
            return nullptr;
        }
    };
    // Parent task to spawn multiple parallel read and write tasks
    struct RootTask: public tbb::task {
        AccessorT& acc;
        RootTask(AccessorT& c): acc(c) {}
        tbb::task* execute()
        {
            ReadTask* r[3]; WriteTask* w[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = new(allocate_child()) ReadTask(acc);
                w[i] = new(allocate_child()) WriteTask(acc);
            }
            set_ref_count(6 /*children*/ + 1 /*wait*/);
            for (int i = 0; i < 3; ++i) {
                spawn(*r[i]); spawn(*w[i]);
            }
            wait_for_all();
            return nullptr;
        }
    };

    Tree4Type tree(/*background=*/0.5);
    AccessorT acc(tree);
    // Populate the tree.
    for (int i = -MAX_COORD; i < MAX_COORD; ++i) {
        acc.setValue(openvdb::Coord(i), float(i));
    }

    // Run multiple read and write tasks in parallel.
    RootTask& root = *new(tbb::task::allocate_root()) RootTask(acc);
    tbb::task::spawn_root_and_wait(root);

#undef MAX_COORD
}


void
TestValueAccessor::testAccessorRegistration()
{
    using openvdb::Index;

    const float background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20);

    openvdb::FloatTree::Ptr tree(new openvdb::FloatTree(background));
    openvdb::tree::ValueAccessor<openvdb::FloatTree> acc(*tree);

    // Set a single leaf voxel via the accessor and verify that
    // the cache is populated.
    acc.setValue(c0, value);
    CPPUNIT_ASSERT_EQUAL(Index(1), tree->leafCount());
    CPPUNIT_ASSERT_EQUAL(tree->root().getLevel(), tree->nonLeafCount());
    CPPUNIT_ASSERT(acc.getNode<openvdb::FloatTree::LeafNodeType>() != nullptr);

    // Reset the voxel to the background value and verify that no nodes
    // have been deleted and that the cache is still populated.
    tree->setValueOff(c0, background);
    CPPUNIT_ASSERT_EQUAL(Index(1), tree->leafCount());
    CPPUNIT_ASSERT_EQUAL(tree->root().getLevel(), tree->nonLeafCount());
    CPPUNIT_ASSERT(acc.getNode<openvdb::FloatTree::LeafNodeType>() != nullptr);

    // Prune the tree and verify that only the root node remains and that
    // the cache has been cleared.
    openvdb::tools::prune(*tree);
    //tree->prune();
    CPPUNIT_ASSERT_EQUAL(Index(0), tree->leafCount());
    CPPUNIT_ASSERT_EQUAL(Index(1), tree->nonLeafCount()); // root node only
    CPPUNIT_ASSERT(acc.getNode<openvdb::FloatTree::LeafNodeType>() == nullptr);

    // Set the leaf voxel again and verify that the cache is repopulated.
    acc.setValue(c0, value);
    CPPUNIT_ASSERT_EQUAL(Index(1), tree->leafCount());
    CPPUNIT_ASSERT_EQUAL(tree->root().getLevel(), tree->nonLeafCount());
    CPPUNIT_ASSERT(acc.getNode<openvdb::FloatTree::LeafNodeType>() != nullptr);

    // Delete the tree and verify that the cache has been cleared.
    tree.reset();
    CPPUNIT_ASSERT(acc.getTree() == nullptr);
    CPPUNIT_ASSERT(acc.getNode<openvdb::FloatTree::RootNodeType>() == nullptr);
    CPPUNIT_ASSERT(acc.getNode<openvdb::FloatTree::LeafNodeType>() == nullptr);
}


void
TestValueAccessor::testGetNode()
{
    using LeafT = Tree4Type::LeafNodeType;

    const ValueType background = 5.0f, value = -9.345f;
    const openvdb::Coord c0(5, 10, 20);

    Tree4Type tree(background);
    tree.setValue(c0, value);
    {
        openvdb::tree::ValueAccessor<Tree4Type> acc(tree);
        // Prime the cache.
        acc.getValue(c0);
        // Verify that the cache contains a leaf node.
        LeafT* node = acc.getNode<LeafT>();
        CPPUNIT_ASSERT(node != nullptr);

        // Erase the leaf node from the cache and verify that it is gone.
        acc.eraseNode<LeafT>();
        node = acc.getNode<LeafT>();
        CPPUNIT_ASSERT(node == nullptr);
    }
    {
        // As above, but with a const tree.
        openvdb::tree::ValueAccessor<const Tree4Type> acc(tree);
        acc.getValue(c0);
        const LeafT* node = acc.getNode<const LeafT>();
        CPPUNIT_ASSERT(node != nullptr);

        acc.eraseNode<LeafT>();
        node = acc.getNode<const LeafT>();
        CPPUNIT_ASSERT(node == nullptr);
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
