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
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/math/Transform.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <set>


class TestLeafOrigin: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestLeafOrigin);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST(test2Values);
    CPPUNIT_TEST(testGetValue);
    CPPUNIT_TEST_SUITE_END();

    void test();
    void test2Values();
    void testGetValue();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafOrigin);


////////////////////////////////////////


void
TestLeafOrigin::test()
{
    using namespace openvdb;

    std::set<Coord> indices;
    indices.insert(Coord( 0,   0,  0));
    indices.insert(Coord( 1,   0,  0));
    indices.insert(Coord( 0, 100,  8));
    indices.insert(Coord(-9,   0,  8));
    indices.insert(Coord(32,   0, 16));
    indices.insert(Coord(33,  -5, 16));
    indices.insert(Coord(42,  17, 35));
    indices.insert(Coord(43,  17, 64));

    FloatTree tree(/*bg=*/256.0);
    std::set<Coord>::iterator iter = indices.begin();
    for ( ; iter != indices.end(); ++iter) tree.setValue(*iter, 1.0);

    for (FloatTree::LeafCIter leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        const Int32 mask = ~((1 << leafIter->log2dim()) - 1);
        const Coord leafOrigin = leafIter->origin();
        for (FloatTree::LeafNodeType::ValueOnCIter valIter = leafIter->cbeginValueOn();
            valIter; ++valIter)
        {
            Coord xyz = valIter.getCoord();
            CPPUNIT_ASSERT_EQUAL(leafOrigin, xyz & mask);

            iter = indices.find(xyz);
            CPPUNIT_ASSERT(iter != indices.end());
            indices.erase(iter);
        }
    }
    CPPUNIT_ASSERT(indices.empty());
}


void
TestLeafOrigin::test2Values()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*bg=*/1.0f);
    FloatTree& tree = grid->tree();

    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(100, 0, 0), 6);

    grid->setTransform(math::Transform::createLinearTransform(0.1));

    FloatTree::LeafCIter iter = tree.cbeginLeaf();
    CPPUNIT_ASSERT_EQUAL(Coord(0, 0, 0), iter->origin());
    ++iter;
    CPPUNIT_ASSERT_EQUAL(Coord(96, 0, 0), iter->origin());
}

void
TestLeafOrigin::testGetValue()
{
    const openvdb::Coord c0(0,-10,0), c1(100,13,0);
    const float v0=5.0f, v1=6.0f, v2=1.0f;
    openvdb::FloatTree::Ptr tree(new openvdb::FloatTree(v2));

    tree->setValue(c0, v0);
    tree->setValue(c1, v1);

    openvdb::FloatTree::LeafCIter iter = tree->cbeginLeaf();
    CPPUNIT_ASSERT_EQUAL(v0, iter->getValue(c0));
    ++iter;
    CPPUNIT_ASSERT_EQUAL(v1, iter->getValue(c1));
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
