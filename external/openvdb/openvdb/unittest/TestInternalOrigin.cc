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
#include <set>

class TestInternalOrigin: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestInternalOrigin);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

    void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestInternalOrigin);

void
TestInternalOrigin::test()
{
    std::set<openvdb::Coord> indices;
    indices.insert(openvdb::Coord( 0,  0,  0));
    indices.insert(openvdb::Coord( 1,  0,  0));
    indices.insert(openvdb::Coord( 0,100,  8));
    indices.insert(openvdb::Coord(-9,  0,  8));
    indices.insert(openvdb::Coord(32,  0, 16));
    indices.insert(openvdb::Coord(33, -5, 16));
    indices.insert(openvdb::Coord(42,707,-35));
    indices.insert(openvdb::Coord(43, 17, 64));

    typedef openvdb::tree::Tree4<float,5,4,3>::Type FloatTree4;
    FloatTree4 tree(0.0f);
    std::set<openvdb::Coord>::iterator iter=indices.begin();
    for (int n = 0; iter != indices.end(); ++n, ++iter) {
        tree.setValue(*iter, float(1.0 + double(n) * 0.5));
    }

    openvdb::Coord C3, G;
    typedef FloatTree4::RootNodeType Node0;
    typedef Node0::ChildNodeType     Node1;
    typedef Node1::ChildNodeType     Node2;
    typedef Node2::LeafNodeType      Node3;
    for (Node0::ChildOnCIter iter0=tree.root().cbeginChildOn(); iter0; ++iter0) {//internal 1
        openvdb::Coord C0=iter0->origin();
        iter0.getCoord(G);
        CPPUNIT_ASSERT_EQUAL(C0,G);
        for (Node1::ChildOnCIter iter1=iter0->cbeginChildOn(); iter1; ++iter1) {//internal 2
            openvdb::Coord C1=iter1->origin();
            iter1.getCoord(G);
            CPPUNIT_ASSERT_EQUAL(C1,G);
            CPPUNIT_ASSERT(C0 <= C1);
            CPPUNIT_ASSERT(C1 <= C0 + openvdb::Coord(Node1::DIM,Node1::DIM,Node1::DIM));
            for (Node2::ChildOnCIter iter2=iter1->cbeginChildOn(); iter2; ++iter2) {//leafs
                openvdb::Coord C2=iter2->origin();
                iter2.getCoord(G);
                CPPUNIT_ASSERT_EQUAL(C2,G);
                CPPUNIT_ASSERT(C1 <= C2);
                CPPUNIT_ASSERT(C2 <= C1 + openvdb::Coord(Node2::DIM,Node2::DIM,Node2::DIM));
                for (Node3::ValueOnCIter iter3=iter2->cbeginValueOn(); iter3; ++iter3) {//leaf voxels
                    iter3.getCoord(G);
                    iter = indices.find(G);
                    CPPUNIT_ASSERT(iter != indices.end());
                    indices.erase(iter);
                }
            }
        }
    }
    CPPUNIT_ASSERT(indices.size() == 0);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
