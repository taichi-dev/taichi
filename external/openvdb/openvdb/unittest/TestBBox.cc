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
#include <openvdb/math/BBox.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>

typedef float Real;

class TestBBox: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestBBox);
    CPPUNIT_TEST(testBBox);
    CPPUNIT_TEST(testCenter);
    CPPUNIT_TEST(testExtent);
    CPPUNIT_TEST_SUITE_END();

    void testBBox();
    void testCenter();
    void testExtent();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBBox);


void
TestBBox::testBBox()
{
    typedef openvdb::Vec3R                     Vec3R;
    typedef openvdb::math::BBox<Vec3R>         BBoxType;

    {
        BBoxType B(Vec3R(1,1,1),Vec3R(2,2,2));

        CPPUNIT_ASSERT(B.isSorted());
        CPPUNIT_ASSERT(B.isInside(Vec3R(1.5,2,2)));
        CPPUNIT_ASSERT(!B.isInside(Vec3R(2,3,2)));
        B.expand(Vec3R(3,3,3));
        CPPUNIT_ASSERT(B.isInside(Vec3R(3,3,3)));
    }

    {
        BBoxType B;
        CPPUNIT_ASSERT(B.empty());
        const Vec3R expected(1);
        B.expand(expected);
        CPPUNIT_ASSERT_EQUAL(expected, B.min());
        CPPUNIT_ASSERT_EQUAL(expected, B.max());
    }
}


void
TestBBox::testCenter()
{
    using namespace openvdb::math;

    const Vec3<double> expected(1.5);

    BBox<openvdb::Vec3R> fbox(openvdb::Vec3R(1.0), openvdb::Vec3R(2.0));
    CPPUNIT_ASSERT_EQUAL(expected, fbox.getCenter());

    BBox<openvdb::Vec3i> ibox(openvdb::Vec3i(1), openvdb::Vec3i(2));
    CPPUNIT_ASSERT_EQUAL(expected, ibox.getCenter());

    openvdb::CoordBBox cbox(openvdb::Coord(1), openvdb::Coord(2));
    CPPUNIT_ASSERT_EQUAL(expected, cbox.getCenter());
}

void
TestBBox::testExtent()
{
    typedef openvdb::Vec3R                     Vec3R;
    typedef openvdb::math::BBox<Vec3R>         BBoxType;

    {
        BBoxType B(Vec3R(-20,0,1),Vec3R(2,2,2));
        CPPUNIT_ASSERT_EQUAL(size_t(2), B.minExtent());
        CPPUNIT_ASSERT_EQUAL(size_t(0), B.maxExtent());
    }
    {
        BBoxType B(Vec3R(1,0,1),Vec3R(2,21,20));
        CPPUNIT_ASSERT_EQUAL(size_t(0), B.minExtent());
        CPPUNIT_ASSERT_EQUAL(size_t(1), B.maxExtent());
    }
    {
        BBoxType B(Vec3R(1,0,1),Vec3R(3,1.5,20));
        CPPUNIT_ASSERT_EQUAL(size_t(1), B.minExtent());
        CPPUNIT_ASSERT_EQUAL(size_t(2), B.maxExtent());
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
