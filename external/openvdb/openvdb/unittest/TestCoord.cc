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
#include <openvdb/Types.h>
#include <sstream>
#include <tbb/tbb_stddef.h> // for tbb::split


class TestCoord: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestCoord);
    CPPUNIT_TEST(testCoord);
    CPPUNIT_TEST(testConversion);
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST(testCoordBBox);
    CPPUNIT_TEST_SUITE_END();

    void testCoord();
    void testConversion();
    void testIO();
    void testCoordBBox();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCoord);


void
TestCoord::testCoord()
{
    using openvdb::Coord;

    for (int i=0; i<3; ++i) {
        CPPUNIT_ASSERT_EQUAL(Coord::min()[i], std::numeric_limits<Coord::Int32>::min());
        CPPUNIT_ASSERT_EQUAL(Coord::max()[i], std::numeric_limits<Coord::Int32>::max());
    }

    Coord xyz(-1, 2, 4);
    Coord xyz2 = -xyz;
    CPPUNIT_ASSERT_EQUAL(Coord(1, -2, -4), xyz2);

    CPPUNIT_ASSERT_EQUAL(Coord(1, 2, 4), openvdb::math::Abs(xyz));

    xyz2 = -xyz2;
    CPPUNIT_ASSERT_EQUAL(xyz, xyz2);

    xyz.setX(-xyz.x());
    CPPUNIT_ASSERT_EQUAL(Coord(1, 2, 4), xyz);

    xyz2 = xyz >> 1;
    CPPUNIT_ASSERT_EQUAL(Coord(0, 1, 2), xyz2);

    xyz2 |= 1;
    CPPUNIT_ASSERT_EQUAL(Coord(1, 1, 3), xyz2);

    CPPUNIT_ASSERT(xyz2 != xyz);
    CPPUNIT_ASSERT(xyz2 < xyz);
    CPPUNIT_ASSERT(xyz2 <= xyz);

    xyz2 -= xyz2;
    CPPUNIT_ASSERT_EQUAL(Coord(), xyz2);

    xyz2.reset(0, 4, 4);
    xyz2.offset(-1);
    CPPUNIT_ASSERT_EQUAL(Coord(-1, 3, 3), xyz2);

    // xyz = (1, 2, 4), xyz2 = (-1, 3, 3)
    CPPUNIT_ASSERT_EQUAL(Coord(-1, 2, 3), Coord::minComponent(xyz, xyz2));
    CPPUNIT_ASSERT_EQUAL(Coord(1, 3, 4), Coord::maxComponent(xyz, xyz2));
}


void
TestCoord::testConversion()
{
    using openvdb::Coord;

    openvdb::Vec3I iv(1, 2, 4);
    Coord xyz(iv);
    CPPUNIT_ASSERT_EQUAL(Coord(1, 2, 4), xyz);
    CPPUNIT_ASSERT_EQUAL(iv, xyz.asVec3I());
    CPPUNIT_ASSERT_EQUAL(openvdb::Vec3i(1, 2, 4), xyz.asVec3i());

    iv = (xyz + iv) + xyz;
    CPPUNIT_ASSERT_EQUAL(openvdb::Vec3I(3, 6, 12), iv);
    iv = iv - xyz;
    CPPUNIT_ASSERT_EQUAL(openvdb::Vec3I(2, 4, 8), iv);

    openvdb::Vec3s fv = xyz.asVec3s();
    CPPUNIT_ASSERT(openvdb::math::isExactlyEqual(openvdb::Vec3s(1, 2, 4), fv));
}


void
TestCoord::testIO()
{
    using openvdb::Coord;

    Coord xyz(-1, 2, 4), xyz2;

    std::ostringstream os(std::ios_base::binary);
    CPPUNIT_ASSERT_NO_THROW(xyz.write(os));

    std::istringstream is(os.str(), std::ios_base::binary);
    CPPUNIT_ASSERT_NO_THROW(xyz2.read(is));

    CPPUNIT_ASSERT_EQUAL(xyz, xyz2);

    os.str("");
    os << xyz;
    CPPUNIT_ASSERT_EQUAL(std::string("[-1, 2, 4]"), os.str());
}

void
TestCoord::testCoordBBox()
{
    {// Empty constructor
        openvdb::CoordBBox b;
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::max(), b.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::min(), b.max());
        CPPUNIT_ASSERT(b.empty());
    }
    {// Construct bbox from min and max
        const openvdb::Coord min(-1,-2,30), max(20,30,55);
        openvdb::CoordBBox b(min, max);
        CPPUNIT_ASSERT_EQUAL(min, b.min());
        CPPUNIT_ASSERT_EQUAL(max, b.max());
    }
    {// Construct bbox from components of min and max
        const openvdb::Coord min(-1,-2,30), max(20,30,55);
        openvdb::CoordBBox b(min[0], min[1], min[2],
                             max[0], max[1], max[2]);
        CPPUNIT_ASSERT_EQUAL(min, b.min());
        CPPUNIT_ASSERT_EQUAL(max, b.max());
    }
    {// tbb::split constructor
         const openvdb::Coord min(-1,-2,30), max(20,30,55);
         openvdb::CoordBBox a(min, max), b(a, tbb::split());
         CPPUNIT_ASSERT_EQUAL(min, b.min());
         CPPUNIT_ASSERT_EQUAL(openvdb::Coord(20, 14, 55), b.max());
         CPPUNIT_ASSERT_EQUAL(openvdb::Coord(-1, 15, 30), a.min());
         CPPUNIT_ASSERT_EQUAL(max, a.max());
    }
    {// createCube
        const openvdb::Coord min(0,8,16);
        const openvdb::CoordBBox b = openvdb::CoordBBox::createCube(min, 8);
        CPPUNIT_ASSERT_EQUAL(min, b.min());
        CPPUNIT_ASSERT_EQUAL(min + openvdb::Coord(8-1), b.max());
    }
    {// inf
        const openvdb::CoordBBox b = openvdb::CoordBBox::inf();
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::min(), b.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::max(), b.max());
    }
    {// empty, hasVolume and volume
        const openvdb::Coord c(1,2,3);
        const openvdb::CoordBBox a(c, c), b(c, c.offsetBy(0,-1,0));
        CPPUNIT_ASSERT( a.hasVolume() && !a.empty());
        CPPUNIT_ASSERT(!b.hasVolume() &&  b.empty());
        CPPUNIT_ASSERT_EQUAL(uint64_t(1), a.volume());
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), b.volume());
    }
    {// volume and split constructor
        const openvdb::Coord min(-1,-2,30), max(20,30,55);
        const openvdb::CoordBBox bbox(min,max);
        openvdb::CoordBBox a(bbox), b(a, tbb::split());
        CPPUNIT_ASSERT_EQUAL(bbox.volume(), a.volume() + b.volume());
        openvdb::CoordBBox c(b, tbb::split());
        CPPUNIT_ASSERT_EQUAL(bbox.volume(), a.volume() + b.volume() + c.volume());
    }
    {// getCenter
        const openvdb::Coord min(1,2,3), max(6,10,15);
        const openvdb::CoordBBox b(min, max);
        CPPUNIT_ASSERT_EQUAL(openvdb::Vec3d(3.5, 6.0, 9.0), b.getCenter());
    }
    {// a volume that overflows Int32.
        using Int32 = openvdb::Int32;
        Int32 maxInt32 = std::numeric_limits<Int32>::max();
        const openvdb::Coord min(Int32(0), Int32(0), Int32(0));
        const openvdb::Coord max(maxInt32-Int32(2), Int32(2), Int32(2));

        const openvdb::CoordBBox b(min, max);
        uint64_t volume = UINT64_C(19327352814);
        CPPUNIT_ASSERT_EQUAL(volume, b.volume());
    }
    {// minExtent and maxExtent
        const openvdb::Coord min(1,2,3);
        {
            const openvdb::Coord max = min + openvdb::Coord(1,2,3);
            const openvdb::CoordBBox b(min, max);
            CPPUNIT_ASSERT_EQUAL(int(b.minExtent()), 0);
            CPPUNIT_ASSERT_EQUAL(int(b.maxExtent()), 2);
        }
        {
            const openvdb::Coord max = min + openvdb::Coord(1,3,2);
            const openvdb::CoordBBox b(min, max);
            CPPUNIT_ASSERT_EQUAL(int(b.minExtent()), 0);
            CPPUNIT_ASSERT_EQUAL(int(b.maxExtent()), 1);
        }
        {
            const openvdb::Coord max = min + openvdb::Coord(2,1,3);
            const openvdb::CoordBBox b(min, max);
            CPPUNIT_ASSERT_EQUAL(int(b.minExtent()), 1);
            CPPUNIT_ASSERT_EQUAL(int(b.maxExtent()), 2);
        }
        {
            const openvdb::Coord max = min + openvdb::Coord(2,3,1);
            const openvdb::CoordBBox b(min, max);
            CPPUNIT_ASSERT_EQUAL(int(b.minExtent()), 2);
            CPPUNIT_ASSERT_EQUAL(int(b.maxExtent()), 1);
        }
        {
            const openvdb::Coord max = min + openvdb::Coord(3,1,2);
            const openvdb::CoordBBox b(min, max);
            CPPUNIT_ASSERT_EQUAL(int(b.minExtent()), 1);
            CPPUNIT_ASSERT_EQUAL(int(b.maxExtent()), 0);
        }
        {
            const openvdb::Coord max = min + openvdb::Coord(3,2,1);
            const openvdb::CoordBBox b(min, max);
            CPPUNIT_ASSERT_EQUAL(int(b.minExtent()), 2);
            CPPUNIT_ASSERT_EQUAL(int(b.maxExtent()), 0);
        }
    }

    {//reset
        openvdb::CoordBBox b;
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::max(), b.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::min(), b.max());
        CPPUNIT_ASSERT(b.empty());

        const openvdb::Coord min(-1,-2,30), max(20,30,55);
        b.reset(min, max);
        CPPUNIT_ASSERT_EQUAL(min, b.min());
        CPPUNIT_ASSERT_EQUAL(max, b.max());
        CPPUNIT_ASSERT(!b.empty());

        b.reset();
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::max(), b.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::min(), b.max());
        CPPUNIT_ASSERT(b.empty());
    }

    {// ZYX Iterator 1
        const openvdb::Coord min(-1,-2,3), max(2,3,5);
        const openvdb::CoordBBox b(min, max);
        const size_t count = b.volume();
        size_t n = 0;
        openvdb::CoordBBox::ZYXIterator ijk(b);
        for (int i=min[0]; i<=max[0]; ++i) {
            for (int j=min[1]; j<=max[1]; ++j) {
                for (int k=min[2]; k<=max[2]; ++k, ++ijk, ++n) {
                    CPPUNIT_ASSERT(ijk);
                    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(i,j,k), *ijk);
                }
            }
        }
        CPPUNIT_ASSERT_EQUAL(count, n);
        CPPUNIT_ASSERT(!ijk);
        ++ijk;
        CPPUNIT_ASSERT(!ijk);
    }

    {// ZYX Iterator 2
        const openvdb::Coord min(-1,-2,3), max(2,3,5);
        const openvdb::CoordBBox b(min, max);
        const size_t count = b.volume();
        size_t n = 0;
        openvdb::Coord::ValueType unused = 0;
        for (const auto& ijk: b) {
            unused += ijk[0];
            CPPUNIT_ASSERT(++n <= count);
        }
        CPPUNIT_ASSERT_EQUAL(count, n);
    }

    {// XYZ Iterator 1
        const openvdb::Coord min(-1,-2,3), max(2,3,5);
        const openvdb::CoordBBox b(min, max);
        const size_t count = b.volume();
        size_t n = 0;
        openvdb::CoordBBox::XYZIterator ijk(b);
        for (int k=min[2]; k<=max[2]; ++k) {
            for (int j=min[1]; j<=max[1]; ++j) {
                for (int i=min[0]; i<=max[0]; ++i, ++ijk, ++n) {
                    CPPUNIT_ASSERT( ijk );
                    CPPUNIT_ASSERT_EQUAL( openvdb::Coord(i,j,k), *ijk );
                }
            }
        }
        CPPUNIT_ASSERT_EQUAL(count, n);
        CPPUNIT_ASSERT( !ijk );
        ++ijk;
        CPPUNIT_ASSERT( !ijk );
    }

    {// XYZ Iterator 2
        const openvdb::Coord min(-1,-2,3), max(2,3,5);
        const openvdb::CoordBBox b(min, max);
        const size_t count = b.volume();
        size_t n = 0;
        for (auto ijk = b.beginXYZ(); ijk; ++ijk) {
            CPPUNIT_ASSERT( ++n <= count );
        }
        CPPUNIT_ASSERT_EQUAL(count, n);
    }

    {// bit-wise operations
        const openvdb::Coord min(-1,-2,3), max(2,3,5);
        const openvdb::CoordBBox b(min, max);
        CPPUNIT_ASSERT_EQUAL(openvdb::CoordBBox(min>>1,max>>1), b>>size_t(1));
        CPPUNIT_ASSERT_EQUAL(openvdb::CoordBBox(min>>3,max>>3), b>>size_t(3));
        CPPUNIT_ASSERT_EQUAL(openvdb::CoordBBox(min<<1,max<<1), b<<size_t(1));
        CPPUNIT_ASSERT_EQUAL(openvdb::CoordBBox(min&1,max&1), b&1);
        CPPUNIT_ASSERT_EQUAL(openvdb::CoordBBox(min|1,max|1), b|1);
    }

    {// test getCornerPoints
        const openvdb::CoordBBox bbox(1, 2, 3, 4, 5, 6);
        openvdb::Coord a[10];
        bbox.getCornerPoints(a);
        //for (int i=0; i<8; ++i) {
        //    std::cerr << "#"<<i<<" = ("<<a[i][0]<<","<<a[i][1]<<","<<a[i][2]<<")\n";
        //}
        CPPUNIT_ASSERT_EQUAL( a[0], openvdb::Coord(1, 2, 3) );
        CPPUNIT_ASSERT_EQUAL( a[1], openvdb::Coord(1, 2, 6) );
        CPPUNIT_ASSERT_EQUAL( a[2], openvdb::Coord(1, 5, 3) );
        CPPUNIT_ASSERT_EQUAL( a[3], openvdb::Coord(1, 5, 6) );
        CPPUNIT_ASSERT_EQUAL( a[4], openvdb::Coord(4, 2, 3) );
        CPPUNIT_ASSERT_EQUAL( a[5], openvdb::Coord(4, 2, 6) );
        CPPUNIT_ASSERT_EQUAL( a[6], openvdb::Coord(4, 5, 3) );
        CPPUNIT_ASSERT_EQUAL( a[7], openvdb::Coord(4, 5, 6) );
        for (int i=1; i<8; ++i) CPPUNIT_ASSERT( a[i-1] < a[i] );
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
