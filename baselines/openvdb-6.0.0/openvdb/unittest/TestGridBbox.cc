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

#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>


class TestGridBbox: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestGridBbox);
    CPPUNIT_TEST(testLeafBbox);
    CPPUNIT_TEST(testGridBbox);
    CPPUNIT_TEST_SUITE_END();

    void testLeafBbox();
    void testGridBbox();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGridBbox);


////////////////////////////////////////


void
TestGridBbox::testLeafBbox()
{
    openvdb::FloatTree tree(/*fillValue=*/256.0f);

    openvdb::CoordBBox bbox;
    CPPUNIT_ASSERT(!tree.evalLeafBoundingBox(bbox));

    // Add values to buffer zero.
    tree.setValue(openvdb::Coord(  0,  9,   9), 2.0);
    tree.setValue(openvdb::Coord(100, 35, 800), 2.5);

    // Coordinates in CoordBBox are inclusive!
    CPPUNIT_ASSERT(tree.evalLeafBoundingBox(bbox));
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0,        8,     8), bbox.min());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(104-1, 40-1, 808-1), bbox.max());

    // Test negative coordinates.
    tree.setValue(openvdb::Coord(-100, -35, -800), 2.5);

    CPPUNIT_ASSERT(tree.evalLeafBoundingBox(bbox));
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-104,   -40,  -800), bbox.min());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(104-1, 40-1, 808-1), bbox.max());
}


void
TestGridBbox::testGridBbox()
{
    openvdb::FloatTree tree(/*fillValue=*/256.0f);

    openvdb::CoordBBox bbox;
    CPPUNIT_ASSERT(!tree.evalActiveVoxelBoundingBox(bbox));

    // Add values to buffer zero.
    tree.setValue(openvdb::Coord(  1,  0,   0), 1.5);
    tree.setValue(openvdb::Coord(  0, 12,   8), 2.0);
    tree.setValue(openvdb::Coord(  1, 35, 800), 2.5);
    tree.setValue(openvdb::Coord(100,  0,  16), 3.0);
    tree.setValue(openvdb::Coord(  1,  0,  16), 3.5);

    // Coordinates in CoordBBox are inclusive!
    CPPUNIT_ASSERT(tree.evalActiveVoxelBoundingBox(bbox));
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(  0,  0,   0), bbox.min());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(100, 35, 800), bbox.max());

    // Test negative coordinates.
    tree.setValue(openvdb::Coord(-100, -35, -800), 2.5);

    CPPUNIT_ASSERT(tree.evalActiveVoxelBoundingBox(bbox));
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-100,   -35,  -800), bbox.min());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(100, 35, 800), bbox.max());
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
